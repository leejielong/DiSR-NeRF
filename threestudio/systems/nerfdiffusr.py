import os
from dataclasses import dataclass, field
from tqdm import tqdm
import copy
import pathlib
import random

import torch
import torch.nn.functional as F
import torch.nn as nn

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *
from threestudio.utils.criterion import PSNR, NIQE

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from threestudio.utils.sr_esrnet import SFTNet, default_init_weights

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    AlphaCompositor,
    rasterize_points,
)
from pytorch3d.renderer.points.rasterizer import PointFragments
import lpips

@threestudio.register("nerfdiffusr-system")
class NerfDiffuSR(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        dynamic_ray_sampling: bool = True
        batch_size: int = 256
        max_batch_size: int = 8192
        vis: bool = False
        vis_interval: int = 100
        vis_save_interval: int = 5000
        use_ray_correlator: bool = False
        supersample: bool = False
        patch_size: int = 128
        use_lr_renders: bool = False

        start_sr_step: int = 20000
        num_sr_steps: int = 10000
        num_sync_steps: int = 10000
        sr_batch_size: int = 1

    cfg: Config

    def configure(self):
        torch.set_float32_matmul_precision('medium') # test

        print(f"Configuring geometry,material,background,renderer")
        # create geometry, material, background, renderer
        super().configure()
        self.criterions = {
            'psnr': PSNR(),
            'niqe': NIQE()
        }

        self.validation_step_psnr = []
        self.test_step_psnr = []
        self.validation_step_niqe = []
        self.test_step_niqe = []
        self.test_step_lpips3 = []
        self.test_step_lpips15 = []

        # load prompt processor
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(self.cfg.prompt_processor)
        self.prompt_processor.configure_text_encoder()
        self.prompt_processor.destroy_text_encoder()
        self.prompt_processor_output = self.prompt_processor()
        print(f"VRAM usage after removing text_encoder: {torch.cuda.memory_allocated(0) / (1024 ** 3)}GB")

        # load guidance
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.vae.enable_tiling()
        # self.guidance.vae.enable_slicing()

        # freeze modules
        # for p in self.geometry.parameters():
        #     p.requires_grad=False
        # print(f"Geometry weights frozen.")
        # for p in self.material.parameters():
        #     p.requires_grad=False
        # print(f"Material weights frozen.")
        for p in self.guidance.parameters():
            p.requires_grad=False
        print(f"Guidance weights frozen.")

        self.stage_interval = self.cfg.num_sr_steps+self.cfg.num_sync_steps

        # lpips loss
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').to(self.device)


    def forward(self, batch: Dict[str, Any], render_latent=False, render_image=True, supersample=False) -> Dict[str, Any]:
        # print(f"In forward step!")

        assert (render_latent or render_image), f"Must render either latents or image or both!"

        render_out = self.renderer(render_latent=render_latent, render_image=render_image, supersample=supersample, **batch)
        return {
            **render_out,
        }
    
    def setup(self,stage):
        if stage == 'fit':

            self.sr_frames_idxs = set()

            self.train_dataset = self.trainer.datamodule.train_dataset
            w,h = self.train_dataset.img_wh_base
            n = len(self.train_dataset.rays)

            class Param(nn.Module):
                def __init__(self):
                    super(Param, self).__init__()
                    self.latents = nn.ParameterList([nn.Parameter(torch.zeros(h,w,4)) for i in range(n)])
                def forward(self,idxs):
                    param_latents = []
                    for i in idxs:
                        param_latents.append(self.latents[i])
                    param_latents = torch.stack(param_latents)
                    return param_latents

            self.param = Param()
            self.base_param_state_dict = self.param.state_dict()
            
            # background colors
            self.default_image_background = torch.zeros([3],device=self.train_dataset.rank).flatten() #[3]]
            black_img = -torch.ones(1,3,128,128,device=self.train_dataset.rank)
            black_latent = self.guidance.encode_images(black_img) #(1,4,32,32)
            self.default_latent_background = black_latent[0,:,16,16].flatten() #[4]

            # get initial ray/sample sizes
            self.train_num_rays = self.cfg.batch_size
            self.train_num_samples = self.cfg.batch_size * self.cfg.renderer.num_samples_per_ray

            self.stage = 'sync'
            self.render_latents = None
            self.sr_stage = 0

            self.apply_mask = self.train_dataset.apply_mask

    def apply_background(self, render, opacity, gt, mask, bg_color):
        render = render + bg_color * (1.0 - opacity)
        gt = gt * mask + bg_color * (1.0 - mask)
        return render, gt

    def training_step(self, batch, batch_idx):

        # stage determination
        if self.global_step >= self.cfg.start_sr_step:

            stage_step = (self.global_step-self.cfg.start_sr_step)%self.stage_interval 
            self.log('stage_step', torch.tensor(stage_step,dtype=torch.float), prog_bar=True)

            # begin sr phase if at sr interval or if no rendered latents are present
            if stage_step<self.cfg.num_sr_steps or self.render_latents is None: 
                self.stage = 'sr'

            # begin sync phase
            if stage_step >= self.cfg.num_sr_steps:
                self.stage = 'sync'
        
        # sync stage
        if self.stage == 'sync':

            idx = random.randint(0,len(self.train_dataset.poses)-1)  

            # load a batch of rays
            batch = self.train_dataset.sample_rays(strategy='random', 
                                            idx=idx,
                                            batch_size=self.train_num_rays, 
                                            supersample=self.cfg.supersample)

            # render adapted images in lr, render latents in sr
            out = self(batch, supersample=self.cfg.supersample)

            # apply latent bg masking
            if self.apply_mask:
                # sample random latent background color
                random_bg = torch.rand(3,device=self.device) #[0,1]
                
                out['comp_rgb'], batch['rgb'] = self.apply_background(render=out['comp_rgb'], #{NHWC}
                                        opacity=out['opacity'],
                                        gt=batch['rgb'], #[NHWC]
                                        mask=batch['fg_mask'],
                                        bg_color=random_bg,
                                        )

            # dynamic ray sampling (only for random ray sampler)
            if self.cfg.dynamic_ray_sampling:
                train_num_rays = int(self.train_num_rays * self.train_num_samples / out['num_samples'])        
                self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.cfg.max_batch_size)
                self.log('train/num_rays', torch.tensor([self.train_num_rays],dtype=torch.float32), prog_bar=True)

            # backward
            loss = 0.0
            loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
            # loss_rgb = F.smooth_l1_loss(out['comp_rgb'], batch['rgb'])

            self.log('train/loss_rgb', loss_rgb, prog_bar=True)
            loss += loss_rgb * self.C(self.cfg.loss.lambda_rgb)


        # sr stage
        if self.stage == 'sr':

            bw,bh = self.train_dataset.img_wh_base
            w,h = self.train_dataset.img_wh
            N = len(self.train_dataset.rays)
            B = self.cfg.sr_batch_size

            # first sr step
            if stage_step == 0:
                print(f"\nStarting SR Stage. Round:{self.sr_stage}")

                # 1. clear param_latents
                self.param.load_state_dict(self.base_param_state_dict) 

                # 2. render images from all training views
                self.render_latents = torch.zeros(size=(N,bh,bw,4),device=self.device) #[N,bh,bw,4]

                print(f"Rendering training views from NeRF...")
                for idx, _ in enumerate(tqdm(self.render_latents)):
                    with torch.no_grad():
                        full_batch = self.dataset.sample_rays(strategy='full',
                                                            idx=idx,
                                                            supersample=self.cfg.supersample)
                        self.renderer.training=False
                        full_out = self(full_batch, supersample=self.cfg.supersample)
                        self.renderer.training=True
                        comp_rgb = full_out['comp_rgb'].reshape(1,h,w,3) #[1HWC]
                        comp_rgb = comp_rgb * 2.0 - 1.0
                        comp_latent = self.guidance.encode_images(comp_rgb.movedim(-1,1)).movedim(1,-1) #[1 bh bw 4]
                        self.render_latents[idx] = comp_latent.reshape(bh,bw,4)
                        
                # reset idxs
                self.idxs = torch.arange(0,N)


            # normal sr step
            # 0. timestep annealing
            if stage_step%(N//B)==0: # update by epoch
                self.guidance.timestep_annealing(stage_step)

            # 1. select a set of B latents from different training views in rolling order
            self.idxs = torch.roll(self.idxs, shifts=B, dims=0)
            idxs = self.idxs[:B]

            # 2. let latents = self.render_latents + self.param.latents
            latent = self.render_latents[idxs] + self.param(idxs)
            orig_image = self.train_dataset.orig_rays[idxs].reshape(B,h,w,3).to(self.device) #[1HWC]

            if self.apply_mask:
                mask = self.train_dataset.fg_masks[idxs].reshape(B,h,w,1).to(self.device)
                lr_mask = self.train_dataset.lr_masks[idxs].reshape(B,bh,bw,1).to(self.device)
                orig_image = mask * orig_image + (1.0-mask) * self.default_image_background # mask orig image with default bg
                latent = lr_mask * latent + (1.0-lr_mask) * self.default_latent_background
            else:
                lr_mask = None

            lr_image = F.interpolate(orig_image.movedim(-1,1), scale_factor=0.25, mode='bicubic', align_corners=False).movedim(1,-1)

            # sample patch from each image
            latent, lr_image, mask = self.sample_patch(latent=latent,
                                                    image=lr_image,
                                                    mask=lr_mask,
                                                    patch_size=self.cfg.patch_size)

            # 3. run guidance to get gradient and loss
            latents_noisy, pred_latents_noisy = self.guidance(latent, lr_image, self.prompt_processor_output) #[NCHW]
            latents_noisy = latents_noisy.movedim(1,-1)
            pred_latents_noisy = pred_latents_noisy.movedim(1,-1)

            # logging
            self.log('t', self.guidance.last_timestep.to(torch.float), prog_bar=True)
            self.log('cfg', self.guidance.cfg.guidance_scale, prog_bar=True)
            lightning_optimizer = self.optimizers()  # self = your model
            for param_group in lightning_optimizer.optimizer.param_groups:
                lr = param_group['lr']
            self.log('lr', lr, prog_bar=True)


            loss = 0.0

            # RSD loss
            # w = (1-self.guidance.alphas[self.guidance.last_timestep])**0.5
            loss_rsd = F.l1_loss(latents_noisy, pred_latents_noisy, reduction="mean")*B
            loss_rsd *= self.C(self.cfg.loss.lambda_rsd)
            loss += loss_rsd
            self.log('train/loss_rsd', loss_rsd.item(), prog_bar=True)


            # last sr step
            if stage_step == self.cfg.num_sr_steps-1:

                # 1. save optimized latents into self.train_dataset.latents
                all_idxs = torch.arange(0,len(self.render_latents))
                self.render_latents += self.param(all_idxs).clone().detach()

                # 2. decode all params to images and save in self.train_dataset.rays
                print(f"\nDecoding SR latents to images...")
                for idx, latent in enumerate(tqdm(self.render_latents)):
                    with torch.no_grad():
                        latent = latent.reshape(1,bh,bw,4)
                        if self.dataset.apply_mask:
                            lr_mask = self.train_dataset.lr_masks[idx].to(self.device).reshape(1,bh,bw,1)
                            latent = lr_mask * latent + (1.0-lr_mask) * self.default_latent_background
                        image = self.guidance.decode_latents(latent.movedim(-1,1)).movedim(1,-1) # NHWC
                        self.train_dataset.rays[idx] = image.reshape(h*w,3).to('cpu')

                        # save images
                        # image = image.reshape(h,w,3).detach().cpu().numpy()
                        # image_filename=f"sr_train/sr_stage_{self.sr_stage}/it{self.global_step}-{idx}.png"
                        # plt.imsave(self.get_save_path(image_filename), image)
                
                self.sr_stage += 1

                # test: shift timestep range for next 
                self.guidance.cfg.min_step_percent -= self.guidance.cfg.t_min_shift_per_stage
                self.guidance.cfg.max_step_percent -= self.guidance.cfg.t_max_shift_per_stage
                self.guidance.cfg.guidance_scale += self.guidance.cfg.cfg_shift_per_stage


                print(f"\nSR Stage Complete. Starting Nerf Sync.")


            # run an additional eval method when in sr stage to view optimized latents
            if stage_step%self.cfg.vis_interval==0:
                bw,bh = self.train_dataset.img_wh_base
                train_idx = torch.randint(0,len(self.train_dataset.poses),size=(1,),dtype=torch.int32)
                # train_idx = torch.tensor([2], dtype=torch.int32)
                latent = self.render_latents[train_idx] + self.param(train_idx)
                if self.apply_mask:
                    lr_mask = self.train_dataset.lr_masks[train_idx].reshape(1,bh,bw,1).to(self.device)
                    latent = lr_mask * latent + (1.0-lr_mask) * self.default_latent_background

                with torch.no_grad():
                    image = self.guidance.decode_latents(latent.movedim(-1,1)).movedim(1,-1) # NHWC
                
                sr_niqe = self.criterions['niqe'](image) # sr space niqe
                self.log('sr/niqe', sr_niqe, prog_bar=True, rank_zero_only=True)  

                image_filename=f"sr_train/it{self.global_step}-{train_idx[0].item()}.png"
                plt.imsave(self.get_save_path(image_filename), image[0].detach().cpu().numpy())



        return {
            'loss': loss
        }
    
    def evaluate(self, batch, batch_idx, image_filename, stage):

        self.default_image_background = torch.zeros([3],device=self.device).flatten() #[3]]
        batch = self.dataset.sample_rays(idx=batch_idx, supersample=self.cfg.supersample)
        out = self(batch, render_latent=False, render_image=True, supersample=self.cfg.supersample)
        W, H = self.dataset.img_wh
        # W4, H4 = int(W*4), int(H*4)

        # decode rendered latents
        comp_rgb = out['comp_rgb'].reshape(1,H,W,3)
        depth = out['depth'].reshape(1,H,W,1) #[1,128,128,1]

        # apply bg mask
        # lr_comp_rgb = F.interpolate(comp_rgb.movedim(-1,1), scale_factor=0.25, mode='bilinear', align_corners=False).movedim(1,-1)
        comp_rgb = comp_rgb.clamp(0,1) 
        # interpolate to SR resolution
        gt_rgb = batch['rgb'].reshape(1,H,W,3)
        # lr_gt_rgb = F.interpolate(gt_rgb.movedim(-1,1), scale_factor=0.25, mode='bilinear', align_corners=False).movedim(1,-1)
        gt_rgb = gt_rgb.clamp(0,1)

        if self.dataset.apply_mask:
            mask = batch['fg_mask'].reshape(1,H,W,1)
            comp_rgb = mask * comp_rgb + (1.0-mask) * self.default_image_background     
            gt_rgb = mask * gt_rgb + (1.0-mask) * self.default_image_background     

        psnr = self.criterions['psnr'](comp_rgb, gt_rgb) # sr space psnr
        # psnr = self.criterions['psnr'](lr_comp_rgb, lr_gt_rgb) # lr space psnr
        niqe = self.criterions['niqe'](comp_rgb) # sr space niqe
        # niqe = self.criterions['niqe'](lr_comp_rgb) # sr space niqe

        # psnr = torch.zeros(1)
        # niqe =torch.zeros(1)
        
 
        if stage == 'val':
            plt.imsave(self.get_save_path("sr_"+image_filename), comp_rgb[0].detach().cpu().numpy())

            return psnr, niqe
        
        if stage == 'test':
        
            self.save_image_grid(image_filename, [
                # {'type': 'rgb', 'img': gt_rgb.view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': comp_rgb.view(H,W,3), 'kwargs': {'data_format': 'HWC'}},
                # {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            ])

            with torch.no_grad():
                # Warped LPIPS ()
                pcl_xyz = out['pcl_xyz']
                pcl_rgb = out['comp_rgb']

                if self.dataset.apply_mask:
                    inlier_mask = mask.reshape(-1).to(torch.bool) #out['rays_valid'][...,0]
                    pcl_xyz = pcl_xyz[inlier_mask]
                    pcl_rgb = pcl_rgb[inlier_mask]
                    
                pcl_xyz = pcl_xyz.reshape(-1,3).to(torch.float32)
                pcl_rgb = pcl_rgb.reshape(-1,3).to(torch.float32)

                lpips3 = self.warped_lpips(pcl_xyz, pcl_rgb,batch_idx,3)[...,0]
                # lpips15 = self.warped_lpips(pcl_xyz, pcl_rgb,batch_idx,15)[...,0]

                # lpips3 = torch.zeros(1)
                lpips15 = torch.zeros(1)

                return psnr, niqe, lpips3, lpips15


    def validation_step(self, batch, batch_idx):
        # sample random index
        idx = random.randint(0,len(self.dataset.poses)-1)  

        psnr, niqe = self.evaluate(batch, idx, image_filename=f"val/it{self.global_step}-{batch_idx}.png", stage='val')
        self.validation_step_psnr.append(psnr)
        self.validation_step_niqe.append(niqe)


    def test_step(self, batch, batch_idx): 
        psnr, niqe, lpips3, lpips15 = self.evaluate(batch, batch_idx, image_filename=f"test/it{self.global_step}-{batch_idx}.png", stage='test')
        self.test_step_lpips3.append(lpips3)
        self.test_step_lpips15.append(lpips15)      
        self.log('test/lpips3', lpips3, prog_bar=True, rank_zero_only=True)         
        self.log('test/lpips15', lpips15, prog_bar=True, rank_zero_only=True)        

        self.test_step_psnr.append(psnr)
        self.test_step_niqe.append(niqe)
        self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)         
        self.log('test/niqe', niqe, prog_bar=True, rank_zero_only=True)     


    def on_validation_epoch_end(self):
        psnr = torch.stack(self.validation_step_psnr)
        psnr = torch.mean(psnr)
        self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)      
        self.validation_step_psnr.clear()  # free memory

        niqe = torch.stack(self.validation_step_niqe)
        niqe = torch.mean(niqe)
        self.log('val/niqe', niqe, prog_bar=True, rank_zero_only=True)      
        self.validation_step_niqe.clear()  # free memory


    def on_test_epoch_end(self):
        psnr = torch.stack(self.test_step_psnr)
        psnr = torch.mean(psnr)
        self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    
        self.test_step_psnr.clear()  # free memory

        niqe = torch.stack(self.test_step_niqe)
        niqe = torch.mean(niqe)
        self.log('test/niqe', niqe, prog_bar=True, rank_zero_only=True)    
        self.test_step_niqe.clear()  # free memory

        lpips3 = torch.stack(self.test_step_lpips3)
        lpips3 = torch.mean(lpips3)
        self.log('test/lpips3', lpips3, prog_bar=True, rank_zero_only=True)    
        self.test_step_lpips3.clear()  # free memory

        lpips15 = torch.stack(self.test_step_lpips15)
        lpips15 = torch.mean(lpips15)
        self.log('test/lpips15', lpips15, prog_bar=True, rank_zero_only=True)    
        self.test_step_lpips15.clear()  # free memory

        self.save_img_sequence(
            f"it{self.global_step}",
            f"test/",
            '(\d+)\.png',
            save_format='mp4',
            fps=30
        )

        # self.export()

    # def export(self):
    #     mesh = self.model.export(self.config.export)
    #     self.save_mesh(
    #         f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
    #         **mesh
    #   )    

    def sample_patch(self, latent, image, mask, patch_size):
        
        B,H,W,_ = latent.shape

        ph=pw=patch_size

        # get masked crop coords
        coord_y, coord_x = torch.meshgrid(torch.arange(H,device=self.device), torch.arange(W,device=self.device), indexing='ij')
        coord_x = coord_x.flatten()
        coord_y = coord_y.flatten()
        x_mask = torch.all(torch.stack([coord_x>(pw//2), coord_x<(W-pw//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW
        y_mask = torch.all(torch.stack([coord_y>(ph//2), coord_y<(H-ph//2)]),dim=0).unsqueeze(0).expand(B,-1) #B,HW

        if mask is not None:
            fg_mask = mask.reshape(B,-1) # B,HW
            comp_masks = torch.all(torch.stack([x_mask,y_mask,fg_mask]),dim=0)
            fg_masks = fg_mask.view(B,H,W,1)
        else:
            comp_masks = torch.all(torch.stack([x_mask,y_mask]),dim=0)

        rays = image.view(B,H,W,3)
        latent = latent.view(B,H,W,4)
    
        patch_rays_list = []
        patch_latent_list = []
        patch_mask_list = []


        for i, comp_mask in enumerate(comp_masks):
            valid_coord_x = coord_x[comp_mask]
            valid_coord_y = coord_y[comp_mask]
            assert len(valid_coord_x) == len(valid_coord_y)
            sample_idx = random.randint(0,len(valid_coord_x)-1)
            h_sample = valid_coord_y[sample_idx]
            w_sample = valid_coord_x[sample_idx]

            # edge sampling for LLFF dataset
            if not self.dataset.apply_mask:
                h_sample = random.randint(0,H)
                w_sample = random.randint(0,W)
                if h_sample < (ph//2):
                    h_sample = ph//2
                if w_sample < (pw//2):
                    w_sample = pw//2
                if h_sample > (H-ph//2):
                    h_sample = (H-ph//2)
                if w_sample > (W-pw//2):
                    w_sample = (W-pw//2)

            center = (h_sample,w_sample)
            crop_size = (ph,pw)
            crop_box = (int(center[1]-crop_size[1]/2), int(center[0]-crop_size[0]/2), int(center[1]+crop_size[1]/2), int(center[0]+crop_size[0]/2)) #left,up,right,down
        
            patch_rays = rays[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,3) #(128*128,3)
            patch_latent = latent[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,4) #(128*128,1)
   

            patch_rays_list.append(patch_rays)
            patch_latent_list.append(patch_latent)

            if self.dataset.apply_mask:
                patch_mask = fg_masks[i,crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1,1) #(128*128,1)
                patch_mask_list.append(patch_mask)
        
        patch_rays = torch.stack(patch_rays_list)
        patch_latent = torch.stack(patch_latent_list)

        patch_rays = patch_rays.reshape(B,ph,pw,3)
        patch_latent = patch_latent.reshape(B,ph,pw,4)

        if self.dataset.apply_mask:
            patch_mask = torch.stack(patch_mask_list)
            patch_mask = patch_mask.reshape(B,ph,pw,1)
        else:
            patch_mask = None

        return patch_latent, patch_rays, patch_mask


    def warped_lpips(self, pcl_xyz,pcl_rgb,i,j): # jth nearest view
        
        W,H = self.dataset.img_wh
        bw,bh = self.dataset.img_wh_base

        # find relative camera distances
        ts = self.dataset.poses[...,-1] #[N,3]
        dist = ts.unsqueeze(0) - ts.unsqueeze(1) # [1,N,3]-[N,1,3]=[N,N,3]
        dist = dist ** 2
        dist = dist.sum(-1) ** 0.5

        # find dists for current view
        dists = dist[i] #[N,3]
        inds = torch.argsort(dists) # ascending distances
        view_j_idx = inds[j] # take jth nearest view

        warped_img, warped_mask = self.warp_pcl(pcl_xyz, pcl_rgb,view_j_idx,bh,bw)
        # view_j_mask = self.dataset.fg_masks[view_j_idx].reshape(1,H,W,1).to(self.device)
        # warped_img = view_j_mask * warped_img + (1.0-view_j_mask) * self.default_image_background # apply view j mask to warped img

        # render from view_j instead of taking from rays.
        full_batch = self.dataset.sample_rays(strategy='full',
                                            idx=view_j_idx,
                                            supersample=self.cfg.supersample)
        full_out = self(full_batch, supersample=self.cfg.supersample)
        gt_rgb = full_out['comp_rgb'].reshape(1,H,W,3) #[1HWC]
        # gt_rgb = self.dataset.rays[view_j_idx].to(self.device).reshape(1,H,W,3)

        # visualize warped img
        # plt.figure(figsize=(20, 10))
        # plt.subplot(121)
        # plt.imshow(warped_img[0, ..., :3].cpu().numpy())
        # plt.subplot(122)
        # plt.imshow(gt_rgb[0, ..., :3].cpu().numpy())
        # plt.show()
        # assert False

        # downscale both images to base dimensions
        # warped_img = F.interpolate(warped_img.movedim(-1,1),size=(bh,bw),mode='bilinear',align_corners=False).movedim(1,-1)
        gt_rgb = F.interpolate(gt_rgb.movedim(-1,1),size=(bh,bw),mode='bilinear',align_corners=False).movedim(1,-1)
        gt_rgb = warped_mask * gt_rgb + (1.0-warped_mask) * self.default_image_background

        # compute lpips
         # closer to "traditional" perceptual loss, when used for optimization
        warped_img = warped_img *2.0 - 1.0
        warped_img = warped_img.movedim(-1,1)
        gt_rgb = gt_rgb *2.0 - 1.0
        gt_rgb = gt_rgb.movedim(-1,1)
        lpips_j = self.loss_fn_lpips(warped_img, gt_rgb)

        # print(f"raw lpips: {lpips_j}")
        # # rescale lpips by ratio of warped mask
        # print(warped_mask.shape)
        # masked_area = warped_mask[...,0].sum()
        # print(f"masked_area: {masked_area}")
        # rescale = (H*W)/masked_area
        # lpips_j *= rescale
        # print(f"rescaled lpips: {lpips_j}")

        return lpips_j
    
    def warp_pcl(self,pcl_xyz,pcl_rgb,idx,H,W):

        with torch.autocast(device_type="cuda",dtype=torch.float32):
            
            # project pcl to pytorch3d NDC space
            pcl_xyz = self.dataset.mvp_projection(pcl_xyz,idx)
            pcl_rgb = pcl_rgb.clamp(0,1)

            # scale coordinates by aspect ratio (for non-square images)
            if W>H:
                aspect_ratio = W/H
                pcl_xyz[:,0] *= aspect_ratio
            else:
                aspect_ratio = H/W
                pcl_xyz[:,1] *= aspect_ratio

            # filter outliers
            outlier_mask = pcl_xyz[:,-1]>0.1
            pcl_xyz = pcl_xyz[outlier_mask]
            pcl_rgb = pcl_rgb[outlier_mask]

            points_proj = Pointclouds([pcl_xyz],features=[pcl_rgb])

            # visualize NDC pcl
            # fig = plot_scene(
            #     {
            #         "Pointcloud": {
            #             "scene": points_proj,
            #         }
            #     },
            #     xaxis={"backgroundcolor":"rgb(200, 200, 230)"},
            #     yaxis={"backgroundcolor":"rgb(230, 200, 200)"},
            #     zaxis={"backgroundcolor":"rgb(200, 230, 200)"}, 
            #     axis_args=AxisArgs(showgrid=True),
            # )
            # fig.show()

            # rasterization
            raster_settings = PointsRasterizationSettings(
                image_size=(H,W), 
                radius = 0.003,
                points_per_pixel = 10,
                bin_size=0
            )

            idx, zbuf, dists2 = rasterize_points(
                points_proj,
                image_size=raster_settings.image_size,
                radius=raster_settings.radius,
                points_per_pixel=raster_settings.points_per_pixel,
                bin_size=raster_settings.bin_size,
                max_points_per_bin=raster_settings.max_points_per_bin,
            )
            fragments=PointFragments(idx=idx, zbuf=zbuf, dists=dists2)

            compositor=AlphaCompositor()

            # rendering
            r = raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            weights = 1 - dists2 / (r * r)
            images = compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                weights,
                points_proj.features_packed().permute(1, 0),
            )
            warped_img = images.permute(0, 2, 3, 1) #1,H,W,3

            # render warped mask
            points_proj = Pointclouds([pcl_xyz],features=[torch.ones_like(pcl_rgb)])

            idx, zbuf, dists2 = rasterize_points(
                points_proj,
                image_size=raster_settings.image_size,
                radius=raster_settings.radius,
                points_per_pixel=raster_settings.points_per_pixel,
                bin_size=raster_settings.bin_size,
                max_points_per_bin=raster_settings.max_points_per_bin,
            )
            fragments=PointFragments(idx=idx, zbuf=zbuf, dists=dists2)

            # rendering
            r = raster_settings.radius
            dists2 = fragments.dists.permute(0, 3, 1, 2)
            weights = 1 - dists2 / (r * r)
            images = compositor(
                fragments.idx.long().permute(0, 3, 1, 2),
                weights,
                points_proj.features_packed().permute(1, 0),
            )
            warped_mask = images.permute(0, 2, 3, 1) #1,H,W,1

            return warped_img, warped_mask