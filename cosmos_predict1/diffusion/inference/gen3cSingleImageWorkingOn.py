import open3d as o3d
import argparse
import os
import cv2
from moge.model.v1 import MoGeModel
import torch
import numpy as np
from cosmos_predict1.diffusion.inference.inference_utils import (
    add_common_arguments,
    check_input_frames,
    validate_args,
)
from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
from cosmos_predict1.utils import log, misc
from cosmos_predict1.utils.io import read_prompts_from_file, save_video
from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
import struct

torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    parser.add_argument(
        "--prompt_upsampler_dir",
        type=str,
        default="Pixtral-12B",
        help="Prompt upsampler weights directory relative to checkpoint_dir",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        help="Input image path for generating a single video",
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        choices=[
            "left",
            "right",
            "up",
            "down",
            "zoom_in",
            "zoom_out",
            "clockwise",
            "counterclockwise",
            "none",
        ],
        default="left",
        help="Camera trajectory",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="center_facing",
        help="Controls camera rotation during movement",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of the camera from the center of the scene",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation on warped frames",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="If set, save the warped images (buffer) side by side with the output video.",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="If set, filter the points continuity of the warped images.",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="If set, use foreground masking for the warped images.",
    )
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()


def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, "num_video_frames must be 121, 241, 361, ... (N*120+1)"

def _predict_moge_depth(current_image_path: str | np.ndarray,
                        target_h: int, target_w: int,
                        device: torch.device, moge_model: MoGeModel):
    """Handles MoGe depth prediction for a single image.

    If the image is directly provided as a NumPy array, it should have shape [H, W, C],
    where the channels are RGB and the pixel values are in [0..255].
    """

    if isinstance(current_image_path, str):
        input_image_bgr = cv2.imread(current_image_path) # ci andrebbe current image path 
        if input_image_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path}")
        input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    else:
        input_image_rgb = current_image_path
    del current_image_path

    depth_pred_h, depth_pred_w = 720, 1280

    input_image_for_depth_resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    input_image_for_depth_tensor_chw = torch.tensor(input_image_for_depth_resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    moge_output_full = moge_model.infer(input_image_for_depth_tensor_chw)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_intrinsics_33_full_normalized = moge_output_full["intrinsics"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(moge_mask_hw_full==0, torch.tensor(1000.0, device=moge_depth_hw_full.device), moge_depth_hw_full)
    moge_intrinsics_33_full_pixel = moge_intrinsics_33_full_normalized.clone()
    moge_intrinsics_33_full_pixel[0, 0] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 1] *= depth_pred_h
    moge_intrinsics_33_full_pixel[0, 2] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 2] *= depth_pred_h

    # Calculate scaling factor for height
    height_scale_factor = target_h / depth_pred_h
    width_scale_factor = target_w / depth_pred_w

    # Resize depth map, mask, and image tensor
    # Resizing depth: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_depth_hw = F.interpolate(
        moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    # Resizing mask: (H, W) -> (1, 1, H, W) for interpolate, then squeeze
    moge_mask_hw = F.interpolate(
        moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
        size=(target_h, target_w),
        mode='nearest',  # Using nearest neighbor for binary mask
    ).squeeze(0).squeeze(0).to(torch.bool)

    # Resizing image tensor: (C, H, W) -> (1, C, H, W) for interpolate, then squeeze
    input_image_tensor_chw_target_res = F.interpolate(
        input_image_for_depth_tensor_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1

    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    # Adjust intrinsics for resized height
    moge_intrinsics_33[1, 1] *= height_scale_factor  # fy
    moge_intrinsics_33[1, 2] *= height_scale_factor  # cy
    moge_intrinsics_33[0, 0] *= width_scale_factor  # fx
    moge_intrinsics_33[0, 2] *= width_scale_factor  # cx

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)
    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # Prepare initial intrinsics [B, 1, 3, 3]
    moge_intrinsics_b133 = moge_intrinsics_33.unsqueeze(0).unsqueeze(0)
    initial_w2c_44 = torch.eye(4, dtype=torch.float32, device=device)
    moge_initial_w2c_b144 = initial_w2c_44.unsqueeze(0).unsqueeze(0)

    return (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    )

def _predict_moge_depth_from_tensor(
    image_tensor_chw_0_1: torch.Tensor, # Shape (C, H_input, W_input), range [0,1]
    moge_model: MoGeModel
):
    """Handles MoGe depth prediction from an image tensor."""
    moge_output_full = moge_model.infer(image_tensor_chw_0_1)
    moge_depth_hw_full = moge_output_full["depth"]      # (moge_inf_h, moge_inf_w)
    moge_mask_hw_full = moge_output_full["mask"]        # (moge_inf_h, moge_inf_w)

    moge_depth_11hw = moge_depth_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.nan_to_num(moge_depth_11hw, nan=1e4)
    moge_depth_11hw = torch.clamp(moge_depth_11hw, min=0, max=1e4)
    moge_mask_11hw = moge_mask_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.where(moge_mask_11hw==0, torch.tensor(1000.0, device=moge_depth_11hw.device), moge_depth_11hw)

    return moge_depth_11hw, moge_mask_11hw

def create_zbuffer_projection_improved(ply_path, w2c_matrix, intrinsics, height, width, device='cuda', debug=True):
    """
    Crea proiezione Z-buffer migliorata con diagnostica colori completa
    """
    if debug:
        print("🎨 Z-Buffer Projection with Color Diagnostics")
        print("=" * 60)
    
    # ─── 1) Carica PLY con diagnostica ───
    def load_ply_robust(ply_path):
        """Carica PLY gestendo diversi formati di colori"""
        pcd = o3d.io.read_point_cloud(ply_path)
        pts_np = np.asarray(pcd.points, dtype=np.float32)
        
        if debug:
            print(f"📂 Loaded {len(pts_np)} points from {ply_path}")
        
        if not pcd.has_colors():
            if debug:
                print("❌ No colors in PLY file")
            return pts_np, None
        
        colors_raw = np.asarray(pcd.colors)
        if debug:
            print(f"🎨 Raw colors info:")
            print(f"   Shape: {colors_raw.shape}")
            print(f"   Dtype: {colors_raw.dtype}")
            print(f"   Range: [{colors_raw.min():.6f}, {colors_raw.max():.6f}]")
            print(f"   Sample colors (first 5):")
            for i in range(min(5, len(colors_raw))):
                print(f"     {i}: {colors_raw[i]}")
        
        # Gestione intelligente del range colori
        if colors_raw.max() <= 1.0:
            cols_np = (colors_raw * 255).astype(np.uint8)
            if debug:
                print("✅ Converted from [0-1] to [0-255]")
        elif colors_raw.max() <= 255.0:
            cols_np = colors_raw.astype(np.uint8)
            if debug:
                print("✅ Colors already in [0-255] range")
        else:
            cols_np = np.clip(colors_raw, 0, 255).astype(np.uint8)
            if debug:
                print(f"⚠️  Unusual color range, clipped to [0-255]")
        
        if debug:
            print(f"🎨 Final colors info:")
            print(f"   Range: [{cols_np.min()}, {cols_np.max()}]")
            print(f"   Unique colors: {len(np.unique(cols_np.reshape(-1, 3), axis=0))}")
            print(f"   Sample final colors:")
            for i in range(min(5, len(cols_np))):
                print(f"     {i}: {cols_np[i]}")
        
        return pts_np, cols_np
    
    pts_np, cols_np = load_ply_robust(ply_path)
    if cols_np is None:
        if debug:
            print("⚠️  Using default red color for all points")
        cols_np = np.full((len(pts_np), 3), [255, 0, 0], dtype=np.uint8)
    
    # ─── 2) Setup Z-Buffer ───
    Z_buf = np.full((height, width), np.inf, dtype=np.float32)
    Img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if debug:
        print(f"🖼️  Image buffer: {height}x{width}")
        print(f"📊 Processing {len(pts_np)} points...")
    
    # ─── 3) Proiezione con statistiche ───
    projected_count = 0
    behind_camera = 0
    out_of_bounds = 0
    color_stats = {'min': [255, 255, 255], 'max': [0, 0, 0]}
    
    for k in range(len(pts_np)):
        point_3d = np.append(pts_np[k], 1.0)
        if isinstance(w2c_matrix, torch.Tensor):
            X_cam = (w2c_matrix.cpu().numpy() @ point_3d)[:3]
        else:
            X_cam = (w2c_matrix @ point_3d)[:3]
        if X_cam[2] <= 0:
            behind_camera += 1
            continue
        
        if isinstance(intrinsics, torch.Tensor):
            fx = intrinsics[0, 0].cpu().item()
            fy = intrinsics[1, 1].cpu().item()
            cx = intrinsics[0, 2].cpu().item()
            cy = intrinsics[1, 2].cpu().item()
        else:
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        u = int((fx * X_cam[0] / X_cam[2]) + cx)
        v = int((fy * X_cam[1] / X_cam[2]) + cy)
        if not (0 <= u < width and 0 <= v < height):
            out_of_bounds += 1
            continue
        
        if X_cam[2] < Z_buf[v, u]:
            Z_buf[v, u] = X_cam[2]
            color = cols_np[k]
            Img[v, u] = color
            projected_count += 1
            for i in range(3):
                color_stats['min'][i] = min(color_stats['min'][i], color[i])
                color_stats['max'][i] = max(color_stats['max'][i], color[i])
    
    if debug:
        print(f"📊 Projection Statistics:")
        print(f"   ✅ Successfully projected: {projected_count:,}")
        print(f"   🚫 Behind camera: {behind_camera:,}")
        print(f"   🔲 Out of bounds: {out_of_bounds:,}")
        print(f"   📈 Projection rate: {projected_count/len(pts_np)*100:.1f}%")
        print(f"🎨 Final image statistics:")
        print(f"   Non-zero pixels: {np.count_nonzero(Img):,}")
        print(f"   Image coverage: {np.count_nonzero(Img)/(height*width)*100:.2f}%")
        print(f"   Color range in image: R[{color_stats['min'][0]}-{color_stats['max'][0]}] "
              f"G[{color_stats['min'][1]}-{color_stats['max'][1]}] "
              f"B[{color_stats['min'][2]}-{color_stats['max'][2]}]")
        # 1) Costruisci una maschera 2D di quali pixel hanno almeno un canale > 0
        mask = Img.sum(axis=2) > 0           # forma (H, W), True per pixel “colorato”
        
        # 2) Estrai i pixel non-zero: ora Img[mask] ha forma (N_pixels, 3)
        colored_pixels = Img[mask]           # es. (40437, 3)
        
        # 3) Verifica che *tutti* questi pixel siano [255,0,0]
        if np.all(colored_pixels == np.array([255, 0, 0]), axis=1).all():
            print("⚠️  WARNING: All pixels are red (default color)")
        else:
            print("✅ Image has varied colors")
    
    if debug:
        cv2.imwrite("zbuffer_result.png", cv2.cvtColor(Img, cv2.COLOR_RGB2BGR))
        print("💾 Saved: zbuffer_result.png")
        depth_vis = Z_buf.copy()
        depth_vis[depth_vis == np.inf] = 0
        if depth_vis.max() > 0:
            depth_vis = ((depth_vis / depth_vis.max()) * 255).astype(np.uint8)
            cv2.imwrite("zbuffer_depth.png", depth_vis)
            print("💾 Saved: zbuffer_depth.png")
        mask = (Img.sum(axis=2) > 0).astype(np.uint8) * 255
        cv2.imwrite("zbuffer_mask.png", mask)
        print("💾 Saved: zbuffer_mask.png")
        
    # ─── 4) Inpainting per riempire i “buchi” ───
    # creiamo la maschera dei pixel da ricostruire (1 = mancante, 0 = già colorato)
    mask = (Img.sum(axis=2) == 0).astype(np.uint8) * 255  
    # radius=5 va bene come punto di partenza, INPAINT_TELEA di solito più fluido
    Img = cv2.inpaint(Img, mask, 5, cv2.INPAINT_TELEA)        

    if debug:
        still_missing = np.count_nonzero(Img.sum(axis=2) == 0)
        print(f"✅ Dopo inpainting pixel ancora vuoti: {still_missing}")

        # (eventuali salvataggi di debug come prima)
        cv2.imwrite("zbuffer_result_filled.png", cv2.cvtColor(Img, cv2.COLOR_RGB2BGR))
        print("💾 Saved: zbuffer_result_filled.png")
    
    input_image = (torch.tensor(Img, dtype=torch.float32).permute(2, 0, 1)[None, None] / 127.5) - 1.0
    input_depth = torch.tensor(Z_buf, dtype=torch.float32)[None, None]
    input_depth[input_depth == np.inf] = input_depth[input_depth != np.inf].max() if input_depth[input_depth != np.inf].numel() > 0 else 0
    
    return input_image.to(device), input_depth.to(device), Img

def load_ply_with_colors_robust(ply_path, debug=True):
    """
    Carica un file PLY gestendo correttamente i colori in diversi formati
    """
    if debug:
        print(f"🔍 Analyzing PLY file: {ply_path}")
    # Metodo 1: Open3D
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        pts_np = np.asarray(pcd.points, dtype=np.float32)
        if debug:
            print(f"✅ Open3D loaded {len(pts_np)} points, has_colors={pcd.has_colors()}")
        if pcd.has_colors():
            colors_raw = np.asarray(pcd.colors)
            if debug:
                print(f"   Raw colors range: [{colors_raw.min():.6f}, {colors_raw.max():.6f}]")
            if colors_raw.max() <= 1.0:
                cols_np = (colors_raw * 255).astype(np.uint8)
            else:
                cols_np = np.clip(colors_raw, 0, 255).astype(np.uint8)
            return pts_np, cols_np
    except Exception as e:
        if debug:
            print(f"   ❌ Open3D failed: {e}")
    # Metodo 2: plyfile
    try:
        if debug:
            print("🔄 Trying plyfile library...")
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        pts_np = np.column_stack([
            vertex['x'].astype(np.float32),
            vertex['y'].astype(np.float32),
            vertex['z'].astype(np.float32)
        ])
        cols_np = None
        for keys in [('red','green','blue'), ('r','g','b'), ('diffuse_red','diffuse_green','diffuse_blue')]:
            if all(k in vertex.dtype.names for k in keys):
                arr = np.column_stack([vertex[k] for k in keys])
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)
                    if arr.max() <= 1.0:
                        arr = (arr * 255).astype(np.uint8)
                    else:
                        arr = np.clip(arr, 0, 255).astype(np.uint8)
                cols_np = arr
                if debug:
                    print(f"   ✅ Found colors with keys: {keys}, range: [{cols_np.min()}, {cols_np.max()}]")
                break
        return pts_np, cols_np
    except Exception as e:
        if debug:
            print(f"   ❌ plyfile failed: {e}")
    # Metodo 3: manuale binario
    try:
        if debug:
            print("🔄 Trying manual binary reading...")
        with open(ply_path, 'rb') as f:
            vertex_count = 0
            is_binary = False
            while True:
                line = f.readline().decode('ascii').strip()
                if line == 'end_header': break
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                if 'format binary' in line:
                    is_binary = True
            if is_binary and vertex_count>0:
                pts, cols = [], []
                fmt = '<3d3B'
                size = struct.calcsize(fmt)
                for _ in range(vertex_count):
                    data = f.read(size)
                    if len(data)!=size: break
                    x,y,z,r,g,b = struct.unpack(fmt,data)
                    pts.append([x,y,z]); cols.append([r,g,b])
                return np.array(pts,dtype=np.float32), np.array(cols,dtype=np.uint8)
    except Exception as e:
        if debug:
            print(f"   ❌ Manual reading failed: {e}")
    if debug:
        print("⚠️  Returning points without colors")
    return pts_np, None

def demo(args):
    misc.set_random_seed(args.seed)
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = Gen3cPipeline(
        inference_type="video2world",
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name="Gen3C-Cosmos-7B",
        prompt_upsampler_dir=args.prompt_upsampler_dir,
        enable_prompt_upsampler=not args.disable_prompt_upsampler,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_prompt_upsampler=args.offload_prompt_upsampler,
        offload_guardrail_models=args.offload_guardrail_models,
        disable_guardrail=args.disable_guardrail,
        disable_prompt_encoder=args.disable_prompt_encoder,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=121,
        seed=args.seed,
    )
    frame_buffer_max = pipeline.model.frame_buffer_max
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_n_frames = pipeline.model.chunk_size
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    prompts = read_prompts_from_file(args.batch_input_path) if args.batch_input_path else [{"prompt": args.prompt, "visual_input": args.input_image_path}]
    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)

    for i, input_dict in enumerate(prompts):
        current_prompt = input_dict.get("prompt")
        current_image_path = input_dict.get("visual_input")
        if current_prompt is None or current_image_path is None:
            log.critical("Missing prompt or visual input, skipping.")
            continue
        if not check_input_frames(current_image_path, 1):
            print(f"Input image {current_image_path} not valid, skipping.")
            continue

        # Depth + image init
        moge_image_b1chw_float, moge_depth_b11hw, moge_mask_b11hw, moge_initial_w2c_b144, moge_intrinsics_b133 = \
            _predict_moge_depth("/export/scratch/lvigorel/GEN3C/assets/diffusion/000.png", args.height, args.width, device, moge_model) #CI ANDREBBE CURRENT IMAGE PATH

        # ─── Usa funzioni integrate per debug ───
        ply_path = "/export/scratch/lvigorel/GEN3C/shoehalf.ply"
        intr = moge_intrinsics_b133[0,0]
        w2c  = moge_initial_w2c_b144[0,0]
        input_img, input_depth, dbg_img = create_zbuffer_projection_improved(
            ply_path, w2c, intr, args.height, args.width, device=device, debug=True
        )

        # Prepara buffer Cache3D con i tensori generati
        cache = Cache3D_Buffer(
            frame_buffer_max=frame_buffer_max,
            noise_aug_strength=args.noise_aug_strength,
            generator=generator,
            input_format=None,
            input_image=input_img.squeeze(1).to(device),
            input_depth=input_depth.to(device),
            input_w2c=moge_initial_w2c_b144[:,0],
            input_intrinsics=moge_intrinsics_b133[:,0],
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
        )

        generated_w2cs, generated_intrinsics = generate_camera_trajectory(
            trajectory_type=args.trajectory,
            initial_w2c=w2c,
            initial_intrinsics=intr,
            num_frames=args.num_video_frames,
            movement_distance=args.movement_distance,
            camera_rotation=args.camera_rotation,
            center_depth=1.0,
            device=device.type,
        )
        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, :sample_n_frames],
            generated_intrinsics[:, :sample_n_frames],
        )

        all_rendered_warps = [rendered_warp_images.clone().cpu()] if args.save_buffer else []

        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path=current_image_path,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue
        video, _ = generated_output

        # Loop AR iterations...
        num_ar_iter = (generated_w2cs.shape[1]-1)//(sample_n_frames-1)
        for it in range(1, num_ar_iter):
            start_idx = it*(sample_n_frames-1)
            end_idx   = start_idx+sample_n_frames
            last_frame = torch.tensor(video[-1], device=device)
            depth_pred, mask_pred = _predict_moge_depth_from_tensor(
                last_frame.permute(2,0,1)/255.0, moge_model
            )
            cache.update_cache(
                new_image=last_frame.permute(2,0,1)[None,None]*2-1,
                new_depth=depth_pred,
                new_w2c=generated_w2cs[:, start_idx],
                new_intrinsics=generated_intrinsics[:, start_idx],
            )
            segment_imgs, segment_msks = cache.render_cache(
                generated_w2cs[:, start_idx:end_idx],
                generated_intrinsics[:, start_idx:end_idx],
            )
            if args.save_buffer:
                all_rendered_warps.append(segment_imgs[:,1:].clone().cpu())
            out = pipeline.generate(
                prompt=current_prompt,
                image_path=segment_imgs,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=segment_imgs,
                rendered_warp_masks=segment_msks,
            )
            if out is not None:
                video_new, _ = out
                video = np.concatenate([video, video_new[1:]], axis=0)

        # Salvataggio
        final_video = video
        if args.save_buffer and all_rendered_warps:
            # concat buffer (omesso per brevità)
            pass

        save_path = os.path.join(args.video_save_folder, f"{i if args.batch_input_path else args.video_save_name}.mp4")
        save_video(video=final_video, fps=args.fps, H=args.height, W=args.width, video_save_quality=5, video_save_path=save_path)
        log.info(f"Saved video to {save_path}")

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)



