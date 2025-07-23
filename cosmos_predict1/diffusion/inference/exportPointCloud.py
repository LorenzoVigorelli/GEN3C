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
torch.enable_grad(False)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video to world generation demo script")
    # add shared arguments
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
            "left", "right", "up", "down",
            "zoom_in", "zoom_out",
            "clockwise", "counterclockwise",
            "none",
        ],
        default="left",
        help="Camera path type",
    )
    parser.add_argument(
        "--camera_rotation",
        type=str,
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        default="center_facing",
        help="Camera orientation mode",
    )
    parser.add_argument(
        "--movement_distance",
        type=float,
        default=0.3,
        help="Distance of camera from scene center",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.0,
        help="Strength of noise augmentation",
    )
    parser.add_argument(
        "--save_buffer",
        action="store_true",
        help="Save warped frames alongside output video",
    )
    parser.add_argument(
        "--filter_points_threshold",
        type=float,
        default=0.05,
        help="Threshold for filtering point continuity",
    )
    parser.add_argument(
        "--foreground_masking",
        action="store_true",
        help="Apply foreground masking to warped frames",
    )
    return parser

def parse_arguments() -> argparse.Namespace:
    parser = create_parser()
    return parser.parse_args()

def validate_args(args):
    assert args.num_video_frames is not None, "num_video_frames must be provided"
    assert (args.num_video_frames - 1) % 120 == 0, \
        "num_video_frames must be 121, 241, 361, ..."

def _predict_moge_depth(current_image_path: str | np.ndarray,
                        target_h: int, target_w: int,
                        device: torch.device, moge_model: MoGeModel):
    """Run MoGe depth prediction on one image."""
    if isinstance(current_image_path, str):
        img_bgr = cv2.imread(current_image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Input image not found: {current_image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = current_image_path
    del current_image_path

    # prepare for model
    depth_h, depth_w = 720, 1280
    img_resized = cv2.resize(img_rgb, (depth_w, depth_h))
    tensor = torch.tensor(img_resized / 255.0, dtype=torch.float32,
                          device=device).permute(2, 0, 1)
    out = moge_model.infer(tensor)

    depth_full = out["depth"]
    mask_full  = out["mask"]
    intr_norm  = out["intrinsics"]

    # mask out invalid depth
    depth_full = torch.where(mask_full == 0,
                             torch.tensor(1000.0, device=device),
                             depth_full)

    # convert intrinsics to pixel units
    intr_pix = intr_norm.clone()
    intr_pix[0,0] *= depth_w; intr_pix[0,2] *= depth_w
    intr_pix[1,1] *= depth_h; intr_pix[1,2] *= depth_h

    # resize depth, mask, image
    h_scale = target_h / depth_h
    w_scale = target_w / depth_w

    depth = F.interpolate(depth_full.unsqueeze(0).unsqueeze(0),
                          size=(target_h, target_w),
                          mode='bilinear',
                          align_corners=False).squeeze()
    mask  = F.interpolate(mask_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
                          size=(target_h, target_w),
                          mode='nearest').bool().squeeze()
    img_t = F.interpolate(tensor.unsqueeze(0),
                          size=(target_h, target_w),
                          mode='bilinear',
                          align_corners=False).squeeze(0)

    img_input = img_t.unsqueeze(0).unsqueeze(1) * 2 - 1

    # adjust intrinsics for resized image
    intr = intr_pix.clone()
    intr[0,0] *= w_scale; intr[0,2] *= w_scale
    intr[1,1] *= h_scale; intr[1,2] *= h_scale

    depth_b = torch.nan_to_num(depth.unsqueeze(0).unsqueeze(0),
                               nan=1e4).clamp(0, 1e4)
    intr_b  = intr.unsqueeze(0).unsqueeze(0)
    w2c     = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    return img_input, depth_b, mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), w2c, intr_b

def _predict_moge_depth_from_tensor(
    image_tensor_chw_0_1: torch.Tensor,
    moge_model: MoGeModel
):
    """Run MoGe depth on a CHW tensor."""
    out = moge_model.infer(image_tensor_chw_0_1)
    depth = out["depth"].unsqueeze(0).unsqueeze(0)
    mask  = out["mask"].unsqueeze(0).unsqueeze(0)
    depth = torch.nan_to_num(depth, nan=1e4).clamp(0, 1e4)
    depth = torch.where(mask==0,
                        torch.tensor(1000.0, device=depth.device),
                        depth)
    return depth, mask

def export_point_cloud(points: np.ndarray,
                       colors: np.ndarray,
                       filename: str = "cloud.ply"):
    """Write points+colors to PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors.dtype != np.float32 and colors.max() > 1.0:
        colors = colors.astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")

def demo(args):
    """Main pipeline: predict depth, build cache, export PLY."""
    misc.set_random_seed(args.seed)
    validate_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)

    # load prompts
    if args.batch_input_path:
        prompts = read_prompts_from_file(args.batch_input_path)
    else:
        prompts = [{"prompt": args.prompt, "visual_input": args.input_image_path}]

    os.makedirs(os.path.dirname(args.video_save_folder), exist_ok=True)

    for entry in prompts:
        img_path = entry.get("visual_input")
        if not img_path:
            log.critical("No visual input, skipping.")
            continue

        if not check_input_frames(img_path, 1):
            print(f"Invalid image {img_path}, skipping.")
            continue
        
        # depth prediction
        img_in, depth_b, mask_b, w2c, intr_b = _predict_moge_depth(
            img_path, args.height, args.width, device, moge_model
        )

        # build the cache
        cache = Cache3D_Buffer(
            frame_buffer_max=Gen3cPipeline(
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
            ).model.frame_buffer_max,
            generator=generator,
            noise_aug_strength=args.noise_aug_strength,
            input_image=img_in[:, 0],
            input_depth=depth_b[:, 0],
            input_w2c=w2c[:, 0],
            input_intrinsics=intr_b[:, 0],
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
        )

        # extract and reshape point+color arrays
        pts = cache.input_points[0,0,0,0]  # [H, W, 3]
        cols = cache.input_image[0,0,0,0]  # [C, H, W]
        H, W = pts.shape[:2]
        pts = pts.reshape(-1, 3).cpu().numpy()
        cols = ((cols.permute(1,2,0).cpu().numpy() * 0.5) + 0.5).reshape(-1, 3)

        # save point cloud
        export_point_cloud(pts, cols, "cache_pointcloud.ply")

if __name__ == "__main__":
    args = parse_arguments()
    if args.prompt is None:
        args.prompt = ""
    args.disable_guardrail = True
    args.disable_prompt_upsampler = True
    demo(args)
