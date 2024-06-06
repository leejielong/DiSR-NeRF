import json
import math
import os
import random
from dataclasses import dataclass
import glob
from PIL import Image
import torchvision.transforms.functional as TF

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
# from threestudio.utils.ops import get_mvp_matrix, get_ray_directions, get_rays
from threestudio.utils.typing import *
from threestudio.utils.misc import get_rank

import cv2
import imageio
import numpy as np
from einops import rearrange

# Note: we try to align dataparsing with nerfstudio and use their datasets



def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32) / 255.0
    img = cv2.resize(img, img_wh)

    if img.shape[-1] == 4: # get mask
        mask = img[..., -1:]
        mask = rearrange(mask, 'h w c -> (h w) c')

        if blend_a: 
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
        else:
            img = img[..., :3] * img[..., -1:]

    else:
        mask = None

    img = img[..., :3]
    img = rearrange(img, 'h w c -> (h w) c')

    return img, mask

def get_proj_matrix(K: Float[Tensor, "3 3"], N: Int, n: Float, f: Float, h:Int, w:Int) -> Float[Tensor, "N 4 4"]:
    proj = [
            [2*K[0, 0]/w, -2*K[0, 1]/w  , (w-2*K[0, 2])/w   , 0             ],
            [0          , -2*K[1, 1]/h  , (h-2*K[1, 2])/h   , 0             ],
            [0          , 0             , (-f-n)/(f-n)      , -2*f*n/(f-n)  ],
            [0          , 0             , -1                , 0             ],
           ]
    proj = torch.FloatTensor(proj).unsqueeze(0).expand(N,-1,-1)
    return proj

def get_mvp_matrix(c2w: Float[Tensor, "B 3 4"], proj_mtx: Float[Tensor, "B 4 4"]) -> Float[Tensor, "B 4 4"]:

    # add 4th row to make c2w a 4x4 matrix
    c2w = torch.cat([c2w, torch.zeros_like(c2w)[:,-1:,:]], dim=1)
    c2w[:,3,3] = 1.0 # (N,4,4)

    # calculate w2c from c2w: R' = Rt, t' = -Rt * t
    # mathematically equivalent to (c2w)^-1
    w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
    w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
    w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
    w2c[:, 3, 3] = 1.0
    # calculate mvp matrix by proj_mtx @ w2c (mv_mtx)
    mvp_mtx = proj_mtx @ w2c
    return mvp_mtx

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


class BlenderDatasetBase():
    def __init__(self, cfg: Any, split: str) -> None:
        self.cfg: BlenderDataModuleConfig = cfg
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        with open(os.path.join(self.cfg.dataroot, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
        
        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        if 'img_wh' in self.cfg:
            w, h = self.cfg.img_wh
            assert round(W / w * h) == H
        elif 'downsample' in self.cfg:
            w, h = W // self.cfg.downsample, H // self.cfg.downsample
        else:
            raise KeyError("Either img_wh or downsample should be specified.")

        self.img_wh_base = (w, h) # img_wh before zoom
        self.near, self.far = 2.0, 6.0
        self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(w, h, self.focal, self.focal, w//2, h//2).to(self.rank) # (h, w, 3)

        # apply zoom
        wz = int(w*self.cfg.zoom)
        hz = int(h*self.cfg.zoom)
        self.img_wh = (wz, hz) # img_wh after zoom
        self.directions = F.interpolate(self.directions.unsqueeze(0).permute(0,3,1,2), (hz,wz), mode="bilinear", align_corners=False)
        self.directions = self.directions.permute(0,2,3,1)[0] #[h,w,3]

        # supersampling ray directions
        wz2, hz2 = int(wz*2), int(hz*2)
        self.directions_x4 = F.interpolate(self.directions.unsqueeze(0).permute(0,3,1,2), (hz2,wz2), mode="bilinear", align_corners=False)
        self.directions_x4 = torch.nn.Unfold(2,1,0,2)(self.directions_x4).reshape(3,4,hz,wz).permute(2,3,1,0) #[HW43]

        # visualize ray directions in xy plane
        # directions = self.directions.reshape(-1,3).detach().cpu().numpy()
        # directions_x4 = self.directions_x4.permute(2,0,1,3).reshape(4,-1,3).detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.scatter(directions[:,0],directions[:,1])
        # plt.scatter(directions_x4[0,:,0],directions_x4[0,:,1])
        # plt.scatter(directions_x4[1,:,0],directions_x4[1,:,1])
        # plt.scatter(directions_x4[2,:,0],directions_x4[2,:,1])
        # plt.scatter(directions_x4[3,:,0],directions_x4[3,:,1])
        # plt.show()
        # assert False

        self.all_c2w, self.all_images, self.all_fg_masks, self.all_lr_masks = [], [], [], []

        print(f"Loading {self.split} dataset images...")
        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.cfg.dataroot, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh_base, Image.BICUBIC)
            lr_img = img
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            lr_img = TF.to_tensor(lr_img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(img[..., -1]) # (h, w)
            self.all_images.append(img[...,:3])
            self.all_lr_masks.append(lr_img[..., -1]) #(bh,bw)

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to('cpu'), \
            torch.stack(self.all_fg_masks, dim=0).float().to('cpu')
        
        self.all_lr_masks = torch.stack(self.all_lr_masks, dim=0).float().to('cpu')

        # get intrinsics and proj matrix
        poses = self.all_c2w
        K = np.float32([[-self.focal, 0, w/2], [0, -self.focal, h/2], [0, 0, 1]]) # negative focal to convert opengl ndc to pytorch3d ndc
        K = torch.FloatTensor(K).to(self.rank)
        N = len(self.all_c2w)
        near, far = 0.1,1000.0
        proj = get_proj_matrix(K,N ,near,far,h,w).to(self.rank)

        self.mvp_mtx = get_mvp_matrix(poses, proj)
        self.poses = self.all_c2w
        self.rays = self.all_images.reshape(-1,hz*wz,3)
        self.fg_masks = self.all_fg_masks.reshape(-1,hz*wz,1)
        self.directions = self.directions.reshape(hz*wz,-1)
        self.latents: Float[Tensor, "N HW 4"] = None
        self.directions_x4 = self.directions_x4.reshape(hz*wz,-1) #[HW,4*3]
        self.lr_masks = self.all_lr_masks.reshape(-1,w*h,1)

        # store original images
        self.orig_rays = self.rays.clone()

    def mvp_projection(self,pcl_xyz,idx):
        mvp_mtx = self.mvp_mtx[idx].unsqueeze(0) #1,4,4
        pcl_homo = torch.cat([pcl_xyz,torch.ones(size=(len(pcl_xyz),1),device=pcl_xyz.device)],dim=-1).unsqueeze(-1) #N,4,1
        pcl_proj = torch.matmul(mvp_mtx,pcl_homo).squeeze(-1)
        pcl_proj = pcl_proj/pcl_proj[...,-1:]
        pcl_proj = pcl_proj.squeeze(-1)[...,:3]
        return pcl_proj

    def sample_rays(self, strategy=None, batch_size=None, idx=0, supersample=False, prob_uniform_patch=0.0):

        if supersample:
            directions = self.directions_x4 #[HW,4*3]
        else:
            directions = self.directions #[HW,3]


        if self.split == "train" and strategy != 'full':
            assert strategy in ['random', 'patch', 'full'], f"Invalid ray sampling strategy {strategy}"
            self.strategy=strategy

            assert batch_size is not None 
            self.batch_size = batch_size

            # random ray sampling
            # sample rays in batch across different images
            if self.strategy == 'random':

                img_idxs = torch.tensor(idx).expand(self.batch_size)

                # select random pixels from directions
                pix_idxs = torch.randint(
                    0, 
                    self.img_wh[0]*self.img_wh[1], 
                    size=(self.batch_size,), 
                    device=self.rays.device
                )

                rays = self.rays[img_idxs, pix_idxs].to(self.rank) # select different pixels in different images

                pose = self.poses[img_idxs]
                if supersample:
                    pose = pose.unsqueeze(1).repeat(1,4,1,1).reshape(-1,3,4) #N*4,3,4

                sample = {
                    'idx': idx, #[1]
                    'pose': pose, #[8192*4,3,4]
                    'direction': directions[pix_idxs], #[8192,3] or [8192,4*3]
                    'rgb': rays[:, :3], #[8192,3]
                }

                if self.fg_masks is not None:
                    fg_mask = self.fg_masks[img_idxs, pix_idxs].to(self.rank)
                    sample.update({'fg_mask':fg_mask})

                if self.latents is not None:
                    latent = self.latents[img_idxs, pix_idxs].to(self.rank)
                    sample.update({'latent':latent})

        else: # val/test or 'full'
            sample = {
                'idx': idx, #[1] #int
                'pose': self.poses[idx].unsqueeze(0), #[1,3,4]
                'direction': directions, #[HW,3] or [HW,4*3]
                'rgb': self.rays[idx].to(self.rank), #[HW,3]
            }
            if self.fg_masks is not None:
                sample.update({'fg_mask':self.fg_masks[idx].to(self.rank)}) #[HW,1]
        
        # get rays_o and rays_d
        direction = sample['direction'].reshape(-1,3) #[HW,3] or [HW4,3]
        pose = sample['pose']

        rays_o, rays_d = self._get_rays(direction, pose) # [B,3],[B,3,4] -> [B,3], [B,3]

        # apply white bg    
        if self.apply_mask:
            background_color = torch.zeros((3,), dtype=torch.float32, device=self.rank)
            sample['rgb'] = sample['rgb'] * sample['fg_mask'] + background_color * (1 - sample['fg_mask']) 

        # return in threestudio compatible format
        out = {
            "idx": sample['idx'],
            "rays_o": rays_o, #[B,3] or #[B4,3]
            "rays_d": rays_d, #[B,3] or #[B4,3]
            "c2w": sample['pose'], #[B,3,4]
            "camera_positions": sample['pose'][...,-1], #[B,3]
            "light_positions": sample['pose'][...,-1], #[B,3]
            "rgb": sample['rgb'], #[B,3]
            'height': self.img_wh[1],
            'width': self.img_wh[0],
        }
        if self.fg_masks is not None:
            out.update({'fg_mask':sample['fg_mask']}) #[B,HW,1]
        if self.latents is not None:
            out.update({'latent':sample['latent']}) #[B,HW,4]

        return out
    
    def _get_rays(self, directions, c2w, keepdim=False):

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # Rotate ray directions from camera coordinate to the world coordinate
            # rays_d = directions @ c2w[:, :3].T # (H, W, 3) # slow?
            assert directions.shape[-1] == 3

            if directions.ndim == 2: # (N_rays, 3)
                assert c2w.ndim == 3 # (N_rays, 4, 4) / (1, 4, 4)
                rays_d = (directions[:,None,:] * c2w[:,:3,:3]).sum(-1) # (N_rays, 3)
                rays_o = c2w[:,:,3].expand(rays_d.shape)
            elif directions.ndim == 3: # (H, W, 3)
                if c2w.ndim == 2: # (4, 4)
                    rays_d = (directions[:,:,None,:] * c2w[None,None,:3,:3]).sum(-1) # (H, W, 3)
                    rays_o = c2w[None,None,:,3].expand(rays_d.shape)
                elif c2w.ndim == 3: # (B, 4, 4)
                    rays_d = (directions[None,:,:,None,:] * c2w[:,None,None,:3,:3]).sum(-1) # (B, H, W, 3)
                    rays_o = c2w[:,None,None,:,3].expand(rays_d.shape)

            if not keepdim:
                rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

            rays_d = F.normalize(rays_d, p=2, dim=-1)

            return rays_o, rays_d



class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __len__(self):
        return len(self.poses)    

    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __iter__(self):
        while True:
            yield {}


@dataclass
class BlenderDataModuleConfig:
    dataroot: str
    data_format : str #['synthetic-nerf']
    downsample: int # downsample training images
    zoom: int # upscale training images


@register("blender-camera-datamodule")
class BlenderDataModule(pl.LightningDataModule):
    cfg: BlenderDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(BlenderDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = BlenderIterableDataset(self.cfg , "train")
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.cfg, "val")
        if stage in [None, 'fit','test']:
            self.test_dataset = BlenderDataset(self.cfg, "test")
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.cfg, "predict")

    def prepare_data(self): # used for downloading dataset,
        pass

    def general_loader(self, dataset, batch_size) -> DataLoader:
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=None
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)