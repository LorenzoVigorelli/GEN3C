# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from torch.nn.functional import interpolate
import torch.nn.functional as F

from einops import rearrange

from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    forward_warp,
    reliable_depth_mask_range_batch,
    unproject_points,
)
from cosmos_predict1.diffusion.inference.camera_utils import align_depth

class Cache3D_Base:
    def __init__(
        self,
        input_image,
        input_depth,
        input_w2c,
        input_intrinsics,
        input_mask=None,
        input_format=None,
        input_points=None,
        weight_dtype=torch.float32,
        is_depth=True,
        device="cuda",
        filter_points_threshold=1.0,
        foreground_masking=False,
    ):
        """
        input_image: Tensor with varying dimensions.
        input_format: List of dimension labels corresponding to input_image's dimensions.
                      E.g., ['B', 'C', 'H', 'W'], ['B', 'F', 'C', 'H', 'W'], etc.
        """
        self.weight_dtype = weight_dtype
        self.is_depth = is_depth
        self.device = device
        self.filter_points_threshold = filter_points_threshold
        self.foreground_masking = foreground_masking
        
        # Map dimension names to their indices in input_image
        format_to_indices = {dim: idx for idx, dim in enumerate(input_format)}
        input_shape = input_image.shape
        if input_mask is not None:
            input_image = torch.cat([input_image, input_mask], dim=format_to_indices.get("C"))

        if input_format is None:
            assert input_image.dim() == 4
            input_format = ["B", "C", "H", "W"]

        # B (batch size), F (frame count), N dimensions: no aggregation during warping.
        # Only broadcasting over F to match the target w2c.
        # V: aggregate via concatenation or duster
        B = input_shape[format_to_indices.get("B", 0)] if "B" in format_to_indices else 1  # batch
        F = input_shape[format_to_indices.get("F", 0)] if "F" in format_to_indices else 1  # frame
        N = input_shape[format_to_indices.get("N", 0)] if "N" in format_to_indices else 1  # buffer
        V = input_shape[format_to_indices.get("V", 0)] if "V" in format_to_indices else 1  # view
        H = input_shape[format_to_indices.get("H", 0)] if "H" in format_to_indices else None
        W = input_shape[format_to_indices.get("W", 0)] if "W" in format_to_indices else None

        # Desired dimension order
        desired_dims = ["B", "F", "N", "V", "C", "H", "W"]

        # Build permute order based on input_format
        permute_order = []
        for dim in desired_dims:
            idx = format_to_indices.get(dim)
            if idx is not None:
                permute_order.append(idx)
            else:
                # Placeholder for dimensions to be added later
                permute_order.append(None)

        # Remove None values for permute operation
        permute_indices = [idx for idx in permute_order if idx is not None]
        input_image = input_image.permute(*permute_indices)

        # Insert dimensions of size 1 where necessary
        for i, idx in enumerate(permute_order):
            if idx is None:
                input_image = input_image.unsqueeze(i)

        # Now input_image has the shape B x F x N x V x C x H x W
        if input_mask is not None:
            self.input_image, self.input_mask = input_image[:, :, :, :, :3], input_image[:, :, :, :, 3:]
            self.input_mask = self.input_mask.to("cpu")
        else:
            self.input_mask = None
            self.input_image = input_image
        self.input_image = self.input_image.to(weight_dtype).to("cpu")

        if input_points is not None:
            self.input_points = input_points.reshape(B, F, N, V, H, W, 3).to("cpu")
            self.input_depth = None
        else:
            input_depth = torch.nan_to_num(input_depth, nan=100)
            input_depth = torch.clamp(input_depth, min=0, max=100)
            if weight_dtype == torch.float16:
                input_depth = torch.clamp(input_depth, max=70)
            self.input_points = (
                self._compute_input_points(
                    input_depth.reshape(-1, 1, H, W),
                    input_w2c.reshape(-1, 4, 4),
                    input_intrinsics.reshape(-1, 3, 3),
                )
                .to(weight_dtype)
                .reshape(B, F, N, V, H, W, 3)
                .to("cpu")
            )
            self.input_depth = input_depth

        if self.filter_points_threshold < 1.0 and input_depth is not None:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth, ratio_thresh=self.filter_points_threshold).reshape(B, F, N, V, 1, H, W)
            if self.input_mask is None:
                self.input_mask = depth_mask.to("cpu")
            else:
                self.input_mask = self.input_mask * depth_mask.to(self.input_mask.device)
        self.boundary_mask = None
        if foreground_masking:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth)
            self.boundary_mask = (~depth_mask).reshape(B, F, N, V, 1, H, W).to("cpu")

    def _compute_input_points(self, input_depth, input_w2c, input_intrinsics):
        input_points = unproject_points(
            input_depth,
            input_w2c,
            input_intrinsics,
            is_depth=self.is_depth,
        )
        return input_points

    def update_cache(self):
        raise NotImplementedError

    def input_frame_count(self) -> int:
        return self.input_image.shape[1]

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        bs, F_target, _, _ = target_w2cs.shape

        B, F, N, V, C, H, W = self.input_image.shape
        assert bs == B

        target_w2cs = target_w2cs.reshape(B, F_target, 1, 4, 4).expand(B, F_target, N, 4, 4).reshape(-1, 4, 4)
        target_intrinsics = (
            target_intrinsics.reshape(B, F_target, 1, 3, 3).expand(B, F_target, N, 3, 3).reshape(-1, 3, 3)
        )

        first_images = rearrange(self.input_image[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, C, H, W), "B F N V C H W-> (B F N) V C H W").to(self.device)
        first_points = rearrange(
            self.input_points[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, H, W, 3), "B F N V H W C-> (B F N) V H W C"
        ).to(self.device)
        first_masks = rearrange(
            self.input_mask[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
        ).to(self.device) if self.input_mask is not None else None
        boundary_masks = rearrange(
            self.boundary_mask.expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
        ) if self.boundary_mask is not None else None

        if first_images.shape[1] == 1:
            warp_chunk_size = 2
            rendered_warp_images = []
            rendered_warp_masks = []
            rendered_warp_depth = []
            rendered_warped_flows = []

            first_images = first_images.squeeze(1)
            first_points = first_points.squeeze(1)
            first_masks = first_masks.squeeze(1) if first_masks is not None else None
            for i in range(0, first_images.shape[0], warp_chunk_size):
                (
                    rendered_warp_images_chunk,
                    rendered_warp_masks_chunk,
                    rendered_warp_depth_chunk,
                    rendered_warped_flows_chunk,
                ) = forward_warp(
                    first_images[i : i + warp_chunk_size],
                    mask1=first_masks[i : i + warp_chunk_size] if first_masks is not None else None,
                    depth1=None,
                    transformation1=None,
                    transformation2=target_w2cs[i : i + warp_chunk_size],
                    intrinsic1=target_intrinsics[i : i + warp_chunk_size],
                    intrinsic2=target_intrinsics[i : i + warp_chunk_size],
                    render_depth=render_depth,
                    world_points1=first_points[i : i + warp_chunk_size],
                    foreground_masking=self.foreground_masking,
                    boundary_mask=boundary_masks[i : i + warp_chunk_size, 0, 0] if boundary_masks is not None else None
                )
                rendered_warp_images.append(rendered_warp_images_chunk)
                rendered_warp_masks.append(rendered_warp_masks_chunk)
                rendered_warp_depth.append(rendered_warp_depth_chunk)
                rendered_warped_flows.append(rendered_warped_flows_chunk)
            rendered_warp_images = torch.cat(rendered_warp_images, dim=0)
            rendered_warp_masks = torch.cat(rendered_warp_masks, dim=0)
            if render_depth:
                rendered_warp_depth = torch.cat(rendered_warp_depth, dim=0)
            rendered_warped_flows = torch.cat(rendered_warped_flows, dim=0)

        else:
            raise NotImplementedError

        pixels = rearrange(rendered_warp_images, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        masks = rearrange(rendered_warp_masks, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        if render_depth:
            pixels = rearrange(rendered_warp_depth, "(b f n) h w -> b f n h w", b=bs, f=F_target, n=N)
        return pixels, masks


class Cache3D_Buffer(Cache3D_Base):
    def __init__(self, frame_buffer_max=0, noise_aug_strength=0, generator=None, **kwargs):
        super().__init__(**kwargs)
        self.frame_buffer_max = frame_buffer_max
        self.noise_aug_strength = noise_aug_strength
        self.generator = generator

    def update_cache(self, new_image, new_depth, new_w2c, new_mask=None, new_intrinsics=None, depth_alignment=True, alignment_method="non_rigid"):  # 3D cache
        new_image = new_image.to(self.weight_dtype).to(self.device)
        new_depth = new_depth.to(self.weight_dtype).to(self.device)
        new_w2c = new_w2c.to(self.weight_dtype).to(self.device)
        if new_intrinsics is not None:
            new_intrinsics = new_intrinsics.to(self.weight_dtype).to(self.device)

        new_depth = torch.nan_to_num(new_depth, nan=1e4)
        new_depth = torch.clamp(new_depth, min=0, max=1e4)

        if depth_alignment:
            target_depth, target_mask = self.render_cache(
                new_w2c.unsqueeze(1), new_intrinsics.unsqueeze(1), render_depth=True
            )
            target_depth, target_mask = target_depth[:, :, 0], target_mask[:, :, 0]
            if alignment_method == "rigid":
                new_depth = (
                    align_depth(
                        new_depth.squeeze(),
                        target_depth.squeeze(),
                        target_mask.bool().squeeze(),
                    )
                    .reshape_as(new_depth)
                    .detach()
                )
            elif alignment_method == "non_rigid":
                with torch.enable_grad():
                    new_depth = (
                        align_depth(
                            new_depth.squeeze(),
                            target_depth.squeeze(),
                            target_mask.bool().squeeze(),
                            k=new_intrinsics.squeeze(),
                            c2w=torch.inverse(new_w2c.squeeze()),
                            alignment_method="non_rigid",
                            num_iters=100,
                            lambda_arap=0.1,
                            smoothing_kernel_size=3,
                        )
                        .reshape_as(new_depth)
                        .detach()
                    )
            else:
                raise NotImplementedError
        new_points = unproject_points(new_depth, new_w2c, new_intrinsics, is_depth=self.is_depth).cpu()
        new_image = new_image.cpu()

        if self.filter_points_threshold < 1.0:
            B, F, N, V, C, H, W = self.input_image.shape
            new_depth = new_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(new_depth, ratio_thresh=self.filter_points_threshold).reshape(B, 1, H, W)
            if new_mask is None:
                new_mask = depth_mask.to("cpu")
            else:
                new_mask = new_mask * depth_mask.to(new_mask.device)
        if new_mask is not None:
            new_mask = new_mask.cpu()
        if self.frame_buffer_max > 1:  # newest frame first
            if self.input_image.shape[2] < self.frame_buffer_max:
                self.input_image = torch.cat([new_image[:, None, None, None], self.input_image], 2)
                self.input_points = torch.cat([new_points[:, None, None, None], self.input_points], 2)
                if self.input_mask is not None:
                    self.input_mask = torch.cat([new_mask[:, None, None, None], self.input_mask], 2)
            else:
                self.input_image[:, :, 0] = new_image[:, None, None]
                self.input_points[:, :, 0] = new_points[:, None, None]
                if self.input_mask is not None:
                    self.input_mask[:, :, 0] = new_mask[:, None, None]
        else:
            self.input_image = new_image[:, None, None, None]
            self.input_points = new_points[:, None, None, None]


    def render_cache(
        self,
        target_w2cs,
        target_intrinsics,
        render_depth: bool = False,
        start_frame_idx: int = 0,  # For consistency with Cache4D
    ):
        assert start_frame_idx == 0, "start_frame_idx must be 0 for Cache3D_Buffer"

        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)
        pixels, masks = super().render_cache(
            target_w2cs, target_intrinsics, render_depth
        )
        if not render_depth:
            noise = torch.randn(pixels.shape, generator=self.generator, device=pixels.device, dtype=pixels.dtype)
            per_buffer_noise = (
                torch.arange(start=pixels.shape[2] - 1, end=-1, step=-1, device=pixels.device)
                * self.noise_aug_strength
            )
            pixels = pixels + noise * per_buffer_noise.reshape(1, 1, -1, 1, 1, 1)  # B, F, N, C, H, W
        return pixels.to(output_device), masks.to(output_device)

    def render_with_pseudo_image(
        self,
        target_w2cs,        # [B, F_target, 4, 4], già su device
        target_intrinsics,  # [B, F_target, 3, 3], già su device
        render_depth: bool = False,
        pseudo_h: int = 1
    ):
        """
        Variante di render_cache compatibile con point-cloud sparsa + colori,
        usando pseudo‐immagine + noise‐augmentation.
        """
        # 0) Estrazione shape
        pts  = self.input_points     # [B, Fi, N, V, H, W, 3]  (su CPU)
        cols = self.input_image      # [B, Fi, N, V, C, H, W]  (su CPU)
        B, Fi, N, V, H, W, _ = pts.shape
        C = cols.shape[4]
        Ft = target_w2cs.shape[1]    # numero di frame da renderizzare
        num_pts = H * W
        tot = B * Ft * N * V         # totale “pixel” pseudo‐immagine

        # 1) Ripeti e flatten Fi→Ft
        pts0  = pts .reshape(B * Fi * N * V, H, W,   3)  # [B*Fi*N*V, H, W, 3]
        cols0 = cols.reshape(B * Fi * N * V, C, H,   W)  # [B*Fi*N*V, C, H, W]
        pts_flat  = pts0 .repeat(Ft, 1, 1, 1)            # [B*Ft*N*V, H, W, 3]
        cols_flat = cols0.repeat(Ft, 1, 1, 1)            # [B*Ft*N*V, C, H, W]

        ####OLDDDD 2) Costruisci pseudo-immagine######
        #pseudo_W   = num_pts // pseudo_h
        #pseudo_img = cols_flat.reshape(tot, C, pseudo_h, pseudo_W)
        #pseudo_pts = pts_flat.reshape(tot, pseudo_h, pseudo_W, 3)



            # 2) Scegli pseudo_h / pseudo_W in modo sicuro e pad
        num_pts = H * W
        # non superare il numero di punti
        pseudo_h = min(pseudo_h, num_pts)
        # ceil per coprire tutti i punti
        pseudo_W = math.ceil(num_pts / pseudo_h)
        # se c'è spazio extra, padding a zero
        pad = pseudo_h * pseudo_W - num_pts
        if pad > 0:
            # cols_flat: [tot, C, num_pts] → pad lungo ultima dim
            cols_flat = torch.cat([
                cols_flat,
                torch.zeros(tot, C, pad, device=cols_flat.device, dtype=cols_flat.dtype)
            ], dim=2)
            # pts_flat: [tot, num_pts, 3] → pad lungo dim 1
            pts_flat = torch.cat([
                pts_flat,
                torch.zeros(tot, pad, 3, device=pts_flat.device, dtype=pts_flat.dtype)
            ], dim=1)
        # ora reshape sicuro
        pseudo_img = cols_flat.reshape(tot, C, pseudo_h, pseudo_W)
        pseudo_pts = pts_flat.reshape(tot, pseudo_h, pseudo_W, 3)

        # 3) Sposta pseudo-img/pts su GPU
        device = self.device
        pseudo_img = pseudo_img.to(device)
        pseudo_pts = pseudo_pts.to(device)

        # 4) Replica extrinseche/intrinseche su [B,Ft,N,V] e flatten
        tw2c = (
            target_w2cs
             .unsqueeze(2).unsqueeze(3)     # [B,Ft,1,1,4,4]
             .expand(  B, Ft, N, V, 4, 4)    # [B,Ft,N,V,4,4]
             .reshape(tot,   4, 4)          # [tot,4,4]
        )
        tK = (
            target_intrinsics
             .unsqueeze(2).unsqueeze(3)     # [B,Ft,1,1,3,3]
             .expand(  B, Ft, N, V, 3, 3)    # [B,Ft,N,V,3,3]
             .reshape(tot,   3, 3)          # [tot,3,3]
        )

        # 5) Chiama forward_warp (tutto su GPU)
        warped_img, warped_mask, warped_depth, _ = forward_warp(
            pseudo_img,
            mask1=None,
            depth1=None,
            transformation1=None,
            transformation2=tw2c,
            intrinsic1=tK,
            intrinsic2=tK,
            render_depth=render_depth,
            world_points1=pseudo_pts,
            foreground_masking=self.foreground_masking,
            boundary_mask=None,
        )

        # 6) Flatten output di warp
        warped_img_flat = warped_img.reshape(tot, C, num_pts)
        if render_depth:
            warped_depth_flat = warped_depth.reshape(tot, num_pts)

        # 7) Scatter-back su H×W
        y = torch.arange(num_pts, device=device) // W
        x = torch.arange(num_pts, device=device) %  W
        out_img   = torch.zeros(tot, C, H, W, device=device)
        out_mask  = torch.zeros(tot, 1, H, W, device=device)
        out_depth = torch.zeros(tot, H, W, device=device) if render_depth else None

        for i in range(num_pts):
            out_img [:, :, y[i], x[i]] = warped_img_flat[:, :, i]
            out_mask[:, :, y[i], x[i]] = warped_mask[:, :, 0, i]
            if render_depth:
                out_depth[:, y[i], x[i]] = warped_depth_flat[:, i]

        # 8) Rimodella in [B, Ft, N, …]
        out_img  = out_img .reshape(B, Ft, N, C, H, W)
        out_mask = out_mask.reshape(B, Ft, N, 1, H, W)
        if render_depth:
            out_depth = out_depth.reshape(B, Ft, N, H, W)

        # 9) Aggiungi noise come in render_cache
        if not render_depth and self.noise_aug_strength > 0:
            noise   = torch.randn_like(out_img)
            per_buf = torch.arange(N-1, -1, -1, device=device) * self.noise_aug_strength
            out_img = out_img + noise * per_buf.view(1,1,N,1,1,1)

        # ——————————————————————————————————————————————————————
        
        return (out_img, out_mask, out_depth) if render_depth else (out_img, out_mask)



class Cache4D(Cache3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_cache(self, **kwargs):
        raise NotImplementedError

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        rendered_warp_images, rendered_warp_masks = super().render_cache(target_w2cs, target_intrinsics, render_depth, start_frame_idx)
        return rendered_warp_images, rendered_warp_masks