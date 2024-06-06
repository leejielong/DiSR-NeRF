from dataclasses import dataclass, field
from functools import partial

import nerfacc
import torch
import torch.nn.functional as F

import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.estimators import ImportanceEstimator
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import create_network_with_input_encoding
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.systems.utils import parse_optimizer, parse_scheduler_to_instance
from threestudio.utils.ops import chunk_batch, get_activation, validate_empty_rays
from threestudio.utils.typing import *

from nerfacc import render_weight_from_density, accumulate_along_rays


@threestudio.register("ngp-volume-renderer")
class NGPVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        eval_chunk_size: int = 160000
        randomized: bool = True
        learned_background: bool = False

        near_plane: float = 0.0
        far_plane: float = 1e10

        return_comp_normal: bool = False
        return_normal_perturb: bool = False

        # occupancy grid parameters
        grid_prune: bool = False
        prune_alpha_threshold: bool = True
        occgrid_resolution: int = 128
        occgrid_levels: int = 1

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)


        # initialize occgrid
        self.scene_aabb = self.bbox.view(-1)
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb, 
            resolution=self.cfg.occgrid_resolution, 
            levels=self.cfg.occgrid_levels
        )

        if not self.cfg.grid_prune: # grid should always start full whether we prune or not
            self.estimator.occs.fill_(True)
            self.estimator.binaries.fill_(True)
        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized

    def forward(self, rays_o,rays_d,supersample, **kwargs):

        self.supersample = supersample

        if self.training:
            out = self.forward_(rays_o,rays_d)
        else:
            out = chunk_batch(self.forward_, self.cfg.eval_chunk_size, rays_o,rays_d)
        return {
            **out,
        }

    def forward_(
        self,
        rays_o: Float[Tensor, "Nr 3"],
        rays_d: Float[Tensor, "Nr 3"],
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:

        # flatten rays
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        n_rays = rays_o_flatten.shape[0]

        # occupancy sampling
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            sigma = self.geometry.forward_density(positions)[..., 0]
            return sigma

        with torch.no_grad(): # config follows instant-nsr-pl
            ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                rays_o_flatten,
                rays_d_flatten,
                sigma_fn=sigma_fn,
                near_plane=0.0,
                far_plane=1e10,
                render_step_size=self.render_step_size,
                alpha_thre=0.0,
                stratified=self.randomized,
                cone_angle=0.0,
            )

        # create dummy sample for empty ray batch
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )

        # get sample positions
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        # evaluate density of sampled points
        geo_out = self.geometry(positions)
        density, feature = geo_out['density'], geo_out['features']

        # evaluate radiance of sampled points
        rgb = self.material(viewdirs=t_dirs,features=feature)

        # compute sample weights and transmitatance
        weights_, trans_, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            density[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]

        # accumulate composite opacity, depth and color of rays
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )

        # aggregate rays in supersampling mode
        if self.supersample:
            opacity = opacity.reshape(-1,4,1).mean(dim=1)
            depth = depth.reshape(-1,4,1).mean(dim=1)

        # populate depth and opacity to each point
        # t_depth = depth[ray_indices]
        # z_variance = nerfacc.accumulate_along_rays( # L2 distance between sample point and surface
        #     weights[..., 0],
        #     values=(t_positions - t_depth) ** 2,
        #     ray_indices=ray_indices,
        #     n_rays=n_rays,
        # )

        # print(comp_rgb.shape) #[Nr,3]
        # print(comp_rgb_fg.shape) #[Nr,3]
        # print(comp_rgb_bg.shape) #[Nr,3]
        # print(opacity.shape) #[Nr,1]
        # print(depth.shape) #[Nr,1]
        # print(z_variance.shape) #[Nr,1]

        pcl_xyz = rays_o + depth*rays_d

        out = {
            "opacity": opacity,
            "depth": depth,
            'rays_valid': opacity > 0,                    
            'num_samples': torch.as_tensor([len(positions)], dtype=torch.int32, device=self.device),
            'pcl_xyz': pcl_xyz,
        }

        # render image
        comp_rgb: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=n_rays
        )

        if self.supersample:
            comp_rgb = comp_rgb.reshape(-1,4,3).mean(dim=1)

        out.update({"comp_rgb": comp_rgb})


        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update({"comp_normal": comp_normal})
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else: # test/val
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update({"comp_normal": comp_normal})

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        # print(f"in update step!")
        if self.cfg.grid_prune:
            # print(f"updating occgrid densities")
            def occ_eval_fn(x):
                density = self.geometry.forward_density(x)
                # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                return density * self.render_step_size

            if self.training and not on_load_weights:
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )

    def update_step_end(self, epoch: int, global_step: int) -> None:
        pass

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
