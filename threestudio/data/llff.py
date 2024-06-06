import os
import math
import numpy as np
from PIL import Image
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
from kornia import create_meshgrid

import pytorch_lightning as pl

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
from threestudio.utils.misc import get_rank


from threestudio.data.ray_utils import *
from threestudio.data.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

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

def create_circular_poses(all_c2w, n_steps=120):
    '''
    1. Find avg cam R and T
    2. project all cam T to avg cam frame
    3. Find 90th percentile x,y distances
    4. Create circle around avg cam frame
    5. Transform circle to world space
    6. Create c2w poses with same R different T
    7. Check video to see if views are circular.
    '''

    # compute initial avg pose
    avg_c2w = all_c2w.mean(dim=0,keepdim=True) #[1,3,4]
    avg_r = avg_c2w[0,:3,:3] # [3,3]
    avg_t = avg_c2w[0,:3,3] #[3]
    up = F.normalize(avg_c2w[:,:,1],dim=-1).flatten() #[3]
    # up = torch.as_tensor([0., 0., 1.], dtype=all_c2w.dtype, device=all_c2w.device)

    # re-estimate true scene center by extrapolating z_vec of closest pose (pts3d center may not be true scene center)
    cam_t = all_c2w[...,3] #[N,3]
    closest_idx = torch.argmin(torch.norm(cam_t-avg_t, dim=-1))
    closest_cam = all_c2w[closest_idx]
    center = closest_cam[...,3] + 1.0*closest_cam[...,2]

    # get all camera pos in world space
    cam_t = all_c2w[:,:,-1:] #[N,3,1]

    # transform cam poses to avg_c2w's view space
    cam_t_view = torch.matmul(avg_c2w[:,:3,:3].permute(0,2,1),  cam_t-avg_c2w[:,:3,3:]) #[N,3,1]
    rads = torch.quantile(cam_t_view.abs(), q=0.9, dim=0).reshape(-1).to(all_c2w.device) #[3]
    rads[-1] = 0.1 # move cameras closer to scene

    all_c2w = []
    for theta in torch.linspace(0., 2.*np.pi, n_steps+1)[:-1]:
        c_view = torch.tensor([-torch.sin(theta),torch.cos(theta),1]).to(rads.device) * rads #[3]
        cam_pos = torch.matmul(avg_c2w[:,:3,:3], c_view.reshape(1,3,1)) + avg_c2w[:,:3,3:] #[1,3,1]
        cam_pos = cam_pos.reshape(-1)  #[3]
        l = F.normalize(center - cam_pos, p=2, dim=0) #[3]
        s = F.normalize(l.cross(up), p=2, dim=0) #[3]
        u = F.normalize(s.cross(l), p=2, dim=0) #[3]
        c2w = torch.cat([torch.stack([-s, u, l], dim=1), cam_pos[:,None]], axis=1) #[3,4]

        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0) #[N,3,4]


    # import matplotlib.pyplot as plt
    # circ_t = all_c2w[:,:,-1:] #[N,3,1]
    # cam_t = cam_t.detach().cpu().numpy()
    # circ_t = circ_t.detach().cpu().numpy()
    # center_t = center.detach().cpu().numpy()
    # center2 = center2.detach().cpu().numpy()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter(cam_t[:,0], cam_t[:,1], cam_t[:,2])
    # ax.scatter(circ_t[:,0], circ_t[:,1], circ_t[:,2])
    # ax.scatter(center_t[0], center_t[1], center_t[2])
    # ax.scatter(center2[0], center2[1], center2[2])

    # # ax.set_zlim(0,-1)
    # plt.show()
    # assert False

    return all_c2w


class LLFFDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}


    def read_intrinsics(self):
        self.downsample = 1./self.cfg.downsample

        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.cfg.dataroot, 'sparse/0/cameras.bin'))

        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh_base = (w, h)

        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0,  0,  1]])
        
        wz = int(w*self.cfg.zoom)
        hz = int(h*self.cfg.zoom)
        self.img_wh = (wz, hz) # img_wh after zoom
        self.directions = get_ray_directions(h, w, self.K).reshape(h,w,3)
        self.directions = F.interpolate(self.directions.unsqueeze(0).permute(0,3,1,2), (hz,wz), mode="bilinear", align_corners=False)
        self.directions = self.directions.permute(0,2,3,1)[0] #[h,w,3]

        # supersampling ray directions
        wz2, hz2 = int(wz*2), int(hz*2)
        self.directions_x4 = F.interpolate(self.directions.unsqueeze(0).permute(0,3,1,2), (hz2,wz2), mode="bilinear", align_corners=False)
        self.directions_x4 = torch.nn.Unfold(2,1,0,2)(self.directions_x4).reshape(3,4,hz,wz).permute(2,3,1,0) #[HW43]

        # reshape directions
        self.directions = self.directions.reshape(hz*wz,3).to(self.rank)
        self.directions_x4 = self.directions_x4.reshape(hz*wz,-1).to(self.rank)

        # convert intrinsics from openGL to pytorch3d NDC
        self.K = torch.FloatTensor([[fx, 0, cx],
                                    [0, -fy, cy],
                                    [0,  0,  1]])

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.cfg.dataroot, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        perm = np.argsort(img_names)
        if '360_v2' in self.cfg.dataroot and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.cfg.dataroot, folder, name)
                     for name in sorted(img_names)]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[perm, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.cfg.dataroot, 'sparse/0/points3D.bin'))
        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)

        self.poses, self.pts3d = center_poses(poses, pts3d)

        scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= scale
        self.pts3d /= scale

        if split == 'test': # use precomputed test poses
            scene = self.cfg.dataroot.split('/')[-1]
            wz,hz = self.img_wh
            # self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            self.poses = torch.FloatTensor(self.poses).to(self.rank)
            self.poses = create_circular_poses(self.poses, n_steps=self.cfg.n_test_traj_steps).to(self.rank)
            self.rays = torch.ones(size=(len(self.poses),hz*wz,3)).cpu()
            self.orig_rays = self.rays.clone().cpu()
            return
        
        # for train and val sets
        wz,hz = self.img_wh

        self.all_images = []
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            img = Image.open(img_path)
            img = img.resize(self.img_wh_base, Image.BICUBIC)
            # img = read_image(img_path, self.img_wh_base, blend_a=False)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
            self.all_images.append(img[...,:3])

        self.rays = torch.stack(self.all_images, dim=0).float().to('cpu').reshape(-1,hz*wz,3)
        self.poses = torch.FloatTensor(self.poses).to(self.rank) # (N_images, 3, 4)
        self.orig_rays = self.rays.clone().cpu()


    def setup(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.rank = get_rank()
        self.apply_mask = False

        self.read_intrinsics()
        self.read_meta(split)


        K = torch.FloatTensor(self.K).to(self.rank)
        N = len(self.poses)
        bw,bh = self.img_wh_base
        near, far = 0.1,1000.0
        proj = get_proj_matrix(K,N ,near,far,bh,bw).to(self.rank)
        self.mvp_mtx = get_mvp_matrix(self.poses, proj)
        self.fg_masks = None
        self.latents = None
        self.lr_masks = None



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
                    'fg_mask': None,
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
                'fg_mask': None,
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
            'fg_mask': None,
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
        

class LLFFDataset(Dataset, LLFFDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class LLFFIterableDataset(IterableDataset, LLFFDatasetBase):
    def __init__(self, cfg, split):
        self.setup(cfg, split)

    def __iter__(self):
        while True:
            yield {}

@dataclass
class LLFFDataModuleConfig:
    dataroot: str
    downsample: int # downsample training images
    zoom: int # upscale training images
    n_test_traj_steps: int = 120

@register('llff-camera-datamodule')
class LLFFDataModule(pl.LightningDataModule):
    cfg: LLFFDataModuleConfig


    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(LLFFDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, 'fit']:
            self.train_dataset = LLFFIterableDataset(self.cfg, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = LLFFDataset(self.cfg, 'val')
        if stage in [None, 'test']:
            self.test_dataset = LLFFDataset(self.cfg, 'test')
        if stage in [None, 'predict']:
            self.predict_dataset = LLFFDataset(self.cfg, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)       