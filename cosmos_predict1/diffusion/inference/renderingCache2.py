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

from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    AlphaCompositor
)

from tqdm import tqdm
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

def load_ply_robust(ply_path):
    """Load a PLY file, handling various color formats"""
    pcd = o3d.io.read_point_cloud(ply_path)
    pts_np = np.asarray(pcd.points, dtype=np.float32)
    
    # If there are no colors, return points only
    if not pcd.has_colors():
        return pts_np, None
    
    colors_raw = np.asarray(pcd.colors)
    
    # Smart handling of color ranges
    if colors_raw.max() <= 1.0:
        # Colors are in [0,1] float range
        cols_np = (colors_raw * 255).astype(np.uint8)
    elif colors_raw.max() <= 255.0:
        # Colors are already in [0,255] range
        cols_np = colors_raw.astype(np.uint8)
    else:
        # Clamp any values outside [0,255]
        cols_np = np.clip(colors_raw, 0, 255).astype(np.uint8)
     
    return pts_np, cols_np


def renderPointcloud(
    ply_path,
    w2c_matrix,
    intrinsics,
    height,
    width,
    device='cuda',
    out_rgb="render.png",
    out_mask="mask.png",
    out_depth="depth.npy",
    point_radius=0.001,
    saveRendered=False,
):
    # load PLY
    pts_np, cols_np = load_ply_robust(ply_path)
    if cols_np is None:
        cols_np = np.full((len(pts_np), 3), [255, 0, 0], dtype=np.uint8)

    # to tensors on device
    pts = torch.from_numpy(pts_np).float().to(device)           # (N,3)
    rgb = torch.from_numpy(cols_np).float().to(device) / 255.0  # (N,3)

    # compute linear depths in camera-space
    #    X_cam = R @ X_world + t
    if not isinstance(w2c_matrix, torch.Tensor):
        w2c = torch.from_numpy(w2c_matrix).float().to(device)
    else:
        w2c = w2c_matrix.to(device)
    R = w2c[:3, :3]  # (3,3)
    t = w2c[:3, 3]   # (3,)

    # pts_homo: (N,4)
    ones = torch.ones((pts.shape[0], 1), device=device)
    pts_h = torch.cat([pts, ones], dim=1)

    # pts_cam_h: (N,4) 
    pts_cam_h = (w2c @ pts_h.T).T
    linear_depths = pts_cam_h[:, 2]               

    # build Pointclouds
    pointcloud = Pointclouds(points=[pts], features=[rgb])

    # prepare intrinsics for renderer
    if not isinstance(intrinsics, torch.Tensor):
        K = torch.from_numpy(intrinsics).float().to(device)
    else:
        K = intrinsics.to(device)
    R_batch = R.unsqueeze(0)    # (1,3,3)
    t_batch = t.unsqueeze(0)    # (1,3)
    image_size = torch.tensor([[height, width]], device=device)

    cameras = cameras_from_opencv_projection(
        R_batch, t_batch, K.unsqueeze(0), image_size
    )

    # rasterization & rendering setup
    raster_settings = PointsRasterizationSettings(
        image_size=(height, width),
        radius=point_radius,
        points_per_pixel=10,
        bin_size=128,
        max_points_per_bin=2000000,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

    # render RGB & mask
    fragments = rasterizer(pointcloud)
    rendered = renderer(pointcloud)[0]  # (H,W,3)
    rgb_t  = (rendered[..., :3] * 255).byte()
    mask_t = (fragments.idx[0, ..., 0] >= 0).byte()

    # build linear-depth map
    idx_map = fragments.idx[0, ..., 0]            
    depth_lin = torch.zeros_like(idx_map, dtype=torch.float32, device=device)
    valid = idx_map >= 0
    depth_lin[valid] = linear_depths[idx_map[valid]]

    # optional saving
    if saveRendered:
        rgb_np   = rgb_t.cpu().numpy()
        mask_np  = (mask_t.cpu().numpy() * 255).astype(np.uint8)
        depth_np = depth_lin.cpu().numpy().astype(np.float32)
        cv2.imwrite(out_rgb,   cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(out_mask,  mask_np)
        np.save(out_depth,     depth_np)

    # normalize outputs for network input
    rgb_tensor   = rgb_t.permute(2, 0, 1)[None].float() / 127.5 - 1.0
    depth_tensor = depth_lin[None, None]  # (1,1,H,W)
    mask_tensor  = mask_t[None, None].float()

    return rgb_tensor, depth_tensor, mask_tensor


def render_cache_objects(generated_w2cs, generated_intrinsics, height, width, save_folder="/export/scratch/lvigorel/GEN3C/rendered"):
    os.makedirs(save_folder, exist_ok=True)
    rgbs, depths, masks = [], [], []
    for i in tqdm(range(generated_w2cs.shape[1])):
        # Build per-frame file paths
        out_rgbf   = os.path.join(save_folder, f"frame_{i:03d}_rgb.png")
        out_maskf  = os.path.join(save_folder, f"frame_{i:03d}_mask.png")
        out_depthf = os.path.join(save_folder, f"frame_{i:03d}_depth.npy")
        intrinsics = generated_intrinsics[0, i]  
        w2c        = generated_w2cs[0, i]        


        # Render objects using the camera parameters
        rgb_t, depth_t, mask_t = renderPointcloud(
            ply_path = "/export/scratch/lvigorel/GEN3Copy/PointClouds/ShoePositioned/fullSceneWithShoeNewPositioned.ply",
            w2c_matrix=w2c,
            intrinsics=intrinsics,
            height=height,
            width=width,
            device="cuda",
            point_radius=0.003,
            saveRendered=True,
            out_rgb=out_rgbf,
            out_mask=out_maskf,
            out_depth=out_depthf,
        )
        rgbs.append(rgb_t)
        depths.append(depth_t)
        masks.append(mask_t)

    rgbs   = torch.cat(rgbs,   dim=0)  # (N,3,H,W)
    depths = torch.cat(depths, dim=0)  # (N,1,H,W)
    masks  = torch.cat(masks,  dim=0)  # (N,1,H,W)
    
    return rgbs, masks, depths

def fuse_renderings(
    warp_rgb, warp_depth, warp_mask,
    obj_rgb,  obj_depth,  obj_mask,
    save_fusion: bool = False,
    save_folder: str = "./fused"
):
    # Normalize warp inputs
    if warp_rgb.dim() == 4:
        F,C,H,W = warp_rgb.shape
        warp_rgb   = warp_rgb.unsqueeze(0).unsqueeze(2)
        warp_depth = warp_depth.unsqueeze(0).reshape(1,F,1,H,W)
        warp_mask  = warp_mask.unsqueeze(0).unsqueeze(2)

    # Normalize object inputs
    if obj_rgb.dim() == 4:
        F,C,H,W = obj_rgb.shape
        obj_rgb   = obj_rgb.unsqueeze(0).unsqueeze(2)
        if obj_depth.dim() == 3:
            obj_depth = obj_depth.unsqueeze(1)
        obj_depth = obj_depth.unsqueeze(0)
        obj_mask  = obj_mask.unsqueeze(0).unsqueeze(2)

    print("warp_depth: ", warp_depth.min().item(), warp_depth.max().item())
    print("obj_depth:  ", obj_depth.min().item(),  obj_depth.max().item())

    warp_valid = warp_mask.bool()
    obj_valid  = obj_mask.bool()
    B,F,N,C,H,W = warp_rgb.shape
    fused_rgb = torch.full_like(warp_rgb, -1.0)
    fused_mask = torch.zeros((B,F,N,1,H,W), dtype=torch.bool, device=warp_rgb.device)

    both       = warp_valid & obj_valid
    warp_closer = (warp_depth < obj_depth)

    # Correct unsqueeze dimension for mask alignment
    choose_w = both & warp_closer.unsqueeze(3)
    choose_o = both & (~warp_closer).unsqueeze(3)
    only_w   = warp_valid & ~obj_valid
    only_o   = obj_valid  & ~warp_valid

    # Expand masks for channels
    choose_w_e = choose_w.expand(-1,-1,-1,C,-1,-1)
    choose_o_e = choose_o.expand(-1,-1,-1,C,-1,-1)
    only_w_e   = only_w.expand(-1,-1,-1,C,-1,-1)
    only_o_e   = only_o.expand(-1,-1,-1,C,-1,-1)

    fused_rgb[choose_w_e] = warp_rgb[choose_w_e]
    fused_rgb[choose_o_e] = obj_rgb[choose_o_e]
    fused_rgb[only_w_e]   = warp_rgb[only_w_e]
    fused_rgb[only_o_e]   = obj_rgb[only_o_e]

    fused_mask[choose_w|choose_o|only_w|only_o] = True

    if save_fusion:
        os.makedirs(save_folder, exist_ok=True)
        for f in range(F):
            img = fused_rgb[0,f,0]
            msk = fused_mask[0,f,0,0]
            if img.min() < 0 or img.max() <= 1.0:
                img_vis = ((img.permute(1,2,0).cpu()+1.0)*127.5).clamp(0,255).byte().numpy()
            else:
                img_vis = img.permute(1,2,0).cpu().byte().numpy()
            cv2.imwrite(os.path.join(save_folder, f"fused_{f:03d}_rgb.png"), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_folder, f"fused_{f:03d}_mask.png"), msk.cpu().numpy().astype(np.uint8)*255)

    return fused_rgb, fused_mask


def render_cache_full(cache, w2cs, intrinsics):
    """
    Wrap cache.render_cache to get (rgb, depth, mask).

    Returns:
        rgb:   (B,F,N,3,H,W)
        depth: (B,F,N,H,W)
        mask:  (B,F,N,1,H,W)
    """
    rgb, mask = cache.render_cache(w2cs, intrinsics, render_depth=False)
    depth, _  = cache.render_cache(w2cs, intrinsics, render_depth=True)
    return rgb, depth, mask


def make_empty_warp_mask(warp_rgb: torch.Tensor) -> torch.Tensor:
    """
    Create an all-False mask with the same (B,F,N) dims as warp_rgb,
    plus a singleton channel dim at index=3.
    """
    B, F, N, C, H, W = warp_rgb.shape
    return torch.zeros((B, F, N, 1, H, W), dtype=torch.bool, device=warp_rgb.device)


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

        ply_path = "/export/scratch/lvigorel/GEN3Copy/PointClouds/ShoePositioned/fullSceneWithShoeNewPositioned.ply"
        intr = moge_intrinsics_b133[0,0]
        w2c  = moge_initial_w2c_b144[0,0]
        

        # this is done because the point clouds are of a different density of points, and the rendering radius points parameter is better with to diversify   
        input_img, input_depth, input_mask = renderPointcloud(
            ply_path=ply_path,
            w2c_matrix=w2c,
            intrinsics=intr,
            height=args.height,
            width=args.width,
            device='cuda',
            out_rgb="render.png",
            out_mask="mask.png",
            out_depth="depth.npy",
            point_radius=0.004, #0.003 il piu basso
            saveRendered=False,
        )

        input_obj, input_depth_obj, input_mask_obj = renderPointcloud(
            ply_path="/export/scratch/lvigorel/GEN3C/shoePointsHalfPositionedNewPositioned.ply",
            w2c_matrix=w2c,
            intrinsics=intr,
            height=args.height,
            width=args.width,
            device='cuda',
            out_rgb="render.png",
            out_mask="mask.png",
            out_depth="depth.npy",
            point_radius=0.001,
            saveRendered=False,
        )
        input_mask = (~input_mask_obj.bool()).float()  

        fused_rgb, fused_mask = fuse_renderings(
            input_img, input_depth, input_mask,
            input_obj,  input_depth_obj,  input_mask_obj, save_fusion=False,
        )

        print("Building the cache...")
        cache = Cache3D_Buffer(
            frame_buffer_max=frame_buffer_max,
            generator=generator,
            noise_aug_strength=args.noise_aug_strength,
            input_image=moge_image_b1chw_float[:, 0].clone(), # [B, C, H, W]
            input_depth=moge_depth_b11hw[:, 0],       # [B, 1, H, W]
            #input_mask=moge_mask_b11hw[:, 0],         # [B, 1, H, W]
            input_w2c=moge_initial_w2c_b144[:, 0],  # [B, 4, 4]
            input_intrinsics=moge_intrinsics_b133[:, 0],# [B, 3, 3]
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
        )

        print("Generated camera trajectory...")
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
        
        print("Rendering cache...")
        warp_rgb, warp_depth, warp_mask = render_cache_full(
            cache, generated_w2cs[:, :sample_n_frames], generated_intrinsics[:, :sample_n_frames]
        )
        print("cache input image shape:", warp_rgb.shape) # (B, T, n_i, C, H, W)
        print("cache input depth shape:", warp_depth.shape) # (B, T, n_i, 1, H, W)
        print("cache input mask shape:", warp_mask.shape) # (B, T, n_i, 1, H, W)
    
        print("Rendering cache objects...")
        objRgb, objMask, objDepth = render_cache_objects(
            generated_w2cs=generated_w2cs[:, :sample_n_frames],
            generated_intrinsics=generated_intrinsics[:, :sample_n_frames],
            height=args.height,
            width=args.width,
        )

        print("warp_rgb:",  warp_rgb.shape)
        print("warp_depth:", warp_depth.shape)
        print("warp_mask:", warp_mask.shape)
        print("objRgb:",   objRgb.shape)
        print("objDepth:", objDepth.shape)
        print("objMask:",  objMask.shape)

    
        warp_mask = make_empty_warp_mask(warp_rgb)

        print("warp_rgb:",  warp_rgb.shape)
        print("warp_depth:", warp_depth.shape)
        print("warp_mask:", warp_mask.shape)
        print("objRgb:",   objRgb.shape)
        print("objDepth:", objDepth.shape)
        print("objMask:",  objMask.shape)

        print("Fusing renderings...")
        fused_rgb, fused_mask = fuse_renderings(
            warp_rgb, warp_depth, warp_mask,
            objRgb,  objDepth,  objMask, save_fusion=True,
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(fused_rgb.clone().cpu())


        #inpaint_masks = (~fused_mask.bool()).float()  # [B,F,1,H,W]
        print("rendered_warp_images shape:", fused_rgb.shape) # (B, T, n_i, C, H, W)
        print("rendered_warp_masks shape:", fused_mask.shape)   # (B, T, n_i, 1, H, W)
        
        
        

        # Pass fused images/masks to the generation pipeline
        generated_output = pipeline.generate(
            prompt=current_prompt,
            image_path="/export/scratch/lvigorel/GEN3Copy/ShoeRendering.png",
            negative_prompt=args.negative_prompt,
            rendered_warp_images=fused_rgb,
            rendered_warp_masks=fused_mask,
        )
        if generated_output is None:
            log.critical("Guardrail blocked generation.")
            continue

        video, prompt = generated_output
        # Final video processing
        final_video_to_save = video
        final_width = args.width

        if args.save_buffer and all_rendered_warps:
            squeezed_warps = [t.squeeze(0) for t in all_rendered_warps] # Each is (T_chunk, n_i, C, H, W)

            if squeezed_warps:
                n_max = max(t.shape[1] for t in squeezed_warps)

                padded_t_list = []
                for sq_t in squeezed_warps:
                    # sq_t shape: (T_chunk, n_i, C, H, W)
                    current_n_i = sq_t.shape[1]
                    padding_needed_dim1 = n_max - current_n_i

                    pad_spec = (0,0, # W
                                0,0, # H
                                0,0, # C
                                0,padding_needed_dim1, # n_i
                                0,0) # T_chunk
                    padded_t = F.pad(sq_t, pad_spec, mode='constant', value=-1.0)
                    padded_t_list.append(padded_t)

                full_rendered_warp_tensor = torch.cat(padded_t_list, dim=0)

                T_total, _, C_dim, H_dim, W_dim = full_rendered_warp_tensor.shape
                buffer_video_TCHnW = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                buffer_video_TCHWstacked = buffer_video_TCHnW.contiguous().view(T_total, C_dim, H_dim, n_max * W_dim)
                buffer_video_TCHWstacked = (buffer_video_TCHWstacked * 0.5 + 0.5) * 255.0
                buffer_numpy_TCHWstacked = buffer_video_TCHWstacked.cpu().numpy().astype(np.uint8)
                buffer_numpy_THWC = np.transpose(buffer_numpy_TCHWstacked, (0, 2, 3, 1))

                final_video_to_save = np.concatenate([buffer_numpy_THWC, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)
                log.info(f"Concatenating video with {n_max} warp buffers. Final video width will be {final_width}")
            else:
                log.info("No warp buffers to save.")


        video_save_path = os.path.join(
            args.video_save_folder,
            f"{i if args.batch_input_path else args.video_save_name}.mp4"
        )

        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)

        # Save video
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=5,
            video_save_path=video_save_path,
        )
        log.info(f"Saved video to {video_save_path}")



if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)