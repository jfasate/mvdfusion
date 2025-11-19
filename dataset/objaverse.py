import os
import glob
import torch
import imageio
import json
import numpy as np
from PIL import Image
from einops import rearrange
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform


AZIMUTHS_16 = [
    0.0, 0.7853981852531433, 1.5707963705062866, 2.356194496154785,
    3.1415927410125732, 3.9269907474517822, 4.71238899230957, 5.497786998748779,
    0.39269909262657166, 1.1780972480773926, 1.9634954929351807, 2.7488934993743896,
    3.5342917442321777, 4.319689750671387, 5.105088233947754, 5.890486240386963
]

ELEVATIONS_16 = [
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622,
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622,
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622,
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622
]

def _unscale_depth( depths):
    shift = 0.5
    scale = 2.0
    depths = depths * scale + shift
    return depths

class Objaverse(torch.utils.data.Dataset):

    def __init__(self,
                 root='',
                 subset='400k',
                 camera_type='fixed_set',
                 stage='train',
                 image_size=256,
                 sample_batch_size = None,
                 fix_elevation = True,
                 load_depth = False,
                 load_mask = False,
                 up_vec = 'y',
                 ):
        super().__init__()

        self.root = root
        self.subset  = subset
        self.camera_type = camera_type
        self.stage = stage
        self.image_size = image_size
        self.sample_batch_size = sample_batch_size
        self.fix_elevation = fix_elevation
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.up_vec = up_vec

        subset_list_addr = f'{root}/subset_list/{subset}_{stage}.json'
        assert os.path.exists(subset_list_addr), 'subset not found'

        with open(subset_list_addr) as fp:
            self.subset_list = json.load(fp)

        print(f'loaded {len(self.subset_list)} entries from {subset}|{stage}')

        self.azimuths_16 = torch.tensor(AZIMUTHS_16)
        self.elevations_16 = torch.tensor(ELEVATIONS_16)
        self.__init_fixed_set_cameras()

    def __len__(self):
        return len(self.subset_list)
    
    def __getitem__(self, index):

        #@ LOAD SCENE FOLDER
        scene_dir = f'{self.root}/{self.subset}/{self.subset_list[index]}/views/'
        scene_list = glob.glob(scene_dir + '*_rgb.jpg')

        if self.camera_type == 'fixed_set':
            # Support both 16-view and 64-view datasets
            num_views = len(scene_list)
            if num_views not in [16, 64]:
                raise ValueError(f"Expected 16 or 64 views but found {num_views} in {scene_dir}")

        #@ GET BATCH IDX
        if self.fix_elevation:
            num_views = len(scene_list)
            if num_views == 16:
                # For 16-view dataset at single elevation, use all views
                if self.stage == 'train':
                    batch_idx = torch.arange(0, 16, step=1)
                else:
                    batch_idx = torch.arange(0, 16, step=1)
            elif num_views == 64:
                # For 64-view dataset, use views at 30° elevation (indices 40-56)
                if self.stage == 'train':
                    batch_idx = torch.arange(8+16+8+8, 8+16+8+8+16, step=1)
                else:
                    batch_idx = torch.arange(8+16+8+8, 8+16+8+8+16, step=1)
            else:
                raise ValueError(f"Unsupported number of views: {num_views}")
        else:
            if self.stage == 'test' or self.sample_batch_size == None:
                batch_idx = torch.arange(len(scene_list))
            elif self.sample_batch_size is not None:
                batch_idx = torch.randperm(len(scene_list))[:self.sample_batch_size]
            else:
                raise NotImplementedError

        #@ LOAD DATA
        images, masks, depths = self._load_images(scene_dir, batch_idx)

        R, T, f, c, azimuth, elevation = self._load_fixed_set_cameras(batch_idx)

        #@ RETURN DICT
        frame_dict = {
            'index': index,
            'idx': self.subset_list[index],
            'images':images,
            'R': R,
            'T': T,
            'f': f,
            'c': c,
            'azimuth': azimuth,
            'elevation': elevation,
        }

        if self.load_depth:
            frame_dict['depths'] = depths

        if self.load_mask:
            frame_dict['masks'] = masks

        return frame_dict

    def _load_images(self, scene_dir, batch_idx):

        images = []
        masks = []
        depths = []

        for idx in batch_idx:
            
            #@ LOAD RGB
            img_addr = scene_dir + f'{idx:03d}_rgb.jpg'
            img = Image.open(img_addr).convert('RGB')
            img = torch.tensor(np.array(img).astype(np.float32), dtype=torch.float32) / 255.0
            images.append(img)

            #@ LOAD DEPTH
            if self.load_depth:
                depth_addr = scene_dir + f'{idx:03d}_depth.png'
                depth_img = imageio.v3.imread(depth_addr)
                
                # Handle both 16-bit and 8-bit depth images
                if depth_img.dtype == np.uint16:
                    # 16-bit depth: normalize to [0, 1]
                    depth = torch.tensor(depth_img.astype(np.float32), dtype=torch.float32) / 65535.0
                else:
                    # 8-bit depth: normalize to [0, 1]
                    depth = torch.tensor(depth_img, dtype=torch.float32) / 255.0
                
                depths.append(depth)

            #@ LOAD MASK
            if self.load_mask:
                mask_addr = scene_dir + f'{idx:03d}_mask.jpg'
                mask = Image.open(mask_addr).convert('L')
                mask = torch.tensor(np.array(mask).astype(np.float32), dtype=torch.float32) / 255.0
                masks.append(mask)

        images = torch.stack(images).permute(0,3,1,2) # (B, 3, H, W)
        
        if self.load_depth:
            depths = torch.stack(depths).unsqueeze(1) # (B, 1, H, W)
        
        if self.load_mask:
            masks = torch.stack(masks).unsqueeze(1) # (B, 1, H, W)

        return images, masks, depths

    def _load_fixed_set_cameras(self, batch_idx):

        azimuth = self.azimuths_16[batch_idx]
        elevation = self.elevations_16[batch_idx]
        
        R = self.cameras_b64.R[batch_idx]
        T = self.cameras_b64.T[batch_idx]
        f = self.cameras_b64.focal_length[batch_idx]
        c = self.cameras_b64.principal_point[batch_idx]

        return R, T, f, c, azimuth, elevation
    
    def _normalize_depths(self, depths):
        
        shift = 0.5
        scale = 2.0
        depths = depths * scale + shift
        return depths
    
    def __init_fixed_set_cameras(self):

        #@ INIT INTRINSICS
        camera_lens = 35
        sensor_width = 32
        distances = 1.5
        focal_x = camera_lens * 2 / sensor_width
        focal_y = camera_lens * 2 / sensor_width
        principal_point = ((0,0),)

        #@ INIT EXTRINSICS
        x = torch.cos(self.azimuths_16)*torch.cos(self.elevations_16)
        y = torch.sin(self.azimuths_16)*torch.cos(self.elevations_16)
        z = torch.sin(self.elevations_16)
        cam_pts = torch.stack([x,y,z],-1) * distances
        
        if self.up_vec == 'z':
            R, T = look_at_view_transform(eye=cam_pts, up=((0,0,1),))
        elif self.up_vec == '-z':
            R, T = look_at_view_transform(eye=cam_pts, up=((0,0,-1),))
        elif self.up_vec == 'y':
            R, T = look_at_view_transform(
                dist=distances,
                azim=self.azimuths_16 * 180 / torch.pi + 90,
                elev=self.elevations_16 * 180 / torch.pi,
                up=((0, 1, 0),),
            )
        else: raise NotImplementedError

        self.cameras_b64 = PerspectiveCameras(
                                R=R, 
                                T=T, 
                                focal_length = ((focal_x, focal_y),),
                                principal_point=principal_point,
                            )   
