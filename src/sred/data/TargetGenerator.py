import numpy as np
from typing import Union
import pathlib
import sred.utils as utils
import os

class TargetGenerator:
    """ Class for generating inpainted targets for SelfReDepth """

    def __init__(self, config: dict = None) -> None:
        """
        Parameters
        ----------
        config : dict
        
            strat : str = 'FMM' | 'GFMM' | 'interGFMM'[default]
                Generation strategy to use
            cleaner_min : uint = 5
                Minimum size allowed by the artefact cleaner
            rad_d : uint = 5
                Radius for depth image inpainting
            
            [For GFMM and interGFMM]:

            reg_interpolation: bool = True
                Whether to use dedicated hole interpolation step
            boxblur_k : uint = 9
                Kernel size of the post-interpolation box blur
            cam_params : dict
                Camera extrinsics and instrinsics
            rad_rgb : uint = 5
                Radius for RGB image inpainting
        
        """
        
        self.reset_config()
        self.reconfig(config)
    

    def reset_config(self) -> None:
        """Resets the configs to the default values
        """
        self.strat = 'interGFMM'
        self.cleaner_min = 5
        self.rad_d = 5
        self.reg_interpolation = True
        self.boxblur_k = 9
        self.cam_params = {
            # intrinsics
            'd_fx': 1,
            'd_fy': 1,
            'd_cx': 0,
            'd_cy': 0,
            'rgb_fx': 1,
            'rgb_fy': 1,
            'rgb_cx': 0,
            'rgb_cy': 0,
            # extrinsics
            'angle': 0,
            't_x': 0,
            't_y': 0
        }
        self.rad_rgb = 5


    def reconfig(self, config: dict) -> None:
        """Change the value of any/all generator parameter
        
        Parameters
        ----------
        config : dict
        
            strat : str = 'FMM' | 'GFMM' | 'interGFMM'[default]
                Generation strategy to use
            cleaner_min : uint = 5
                Minimum size allowed by the artefact cleaner
            rad_d : uint = 5
                Radius for depth image inpainting
            
            [For GFMM and interGFMM]:

            reg_interpolation: bool = True
                Whether to use dedicated hole interpolation step
            boxblur_k : uint = 9
                Kernel size of the post-interpolation box blur
            cam_params : dict
                Camera extrinsics and instrinsics
            rad_rgb : uint = 5
                Radius for RGB image inpainting
        
        """

        if config is None or not isinstance(config, dict):
            return

        if 'strat' in config:
            self.strat = config['strat']
        if 'cleaner_min' in config:
            self.cleaner_min = config['cleaner_min']
        if 'rad_d' in config:
            self.rad_d = config['rad_d']
        if 'reg_interpolation' in config:
            self.reg_interpolation = config['reg_interpolation']
        if 'boxblur_k' in config:
            self.boxblur_k = config['boxblur_k']
        if 'cam_params' in config:
            self.cam_params = {}
            self.cam_params['d_fx'] = config['cam_params']['d_fx']
            self.cam_params['d_fy'] = config['cam_params']['d_fy']
            self.cam_params['d_cx'] = config['cam_params']['d_cx']
            self.cam_params['d_cy'] = config['cam_params']['d_cy']
            self.cam_params['rgb_fx'] = config['cam_params']['rgb_fx']
            self.cam_params['rgb_fy'] = config['cam_params']['rgb_fy']
            self.cam_params['rgb_cx'] = config['cam_params']['rgb_cx']
            self.cam_params['rgb_cy'] = config['cam_params']['rgb_cy']
            self.cam_params['angle'] = config['cam_params']['angle']
            self.cam_params['t_x'] = config['cam_params']['t_x']
            self.cam_params['t_y'] = config['cam_params']['t_y']
        if 'rad_rgb' in config:
            self.rad_rgb = config['rad_rgb']


    def generate_one(self, d_img: np.ndarray, rgb_img: np.ndarray = None) -> np.ndarray:
        """Generate a single target depth image (solely with arrays)

        Parameters
        ----------
        d_img : ndarray
            Unaltered depth image
        
        rgb_img : ndarray
            Unaltered RGB image
        
        Returns
        -------
        ndarray
            New generated target depth image
        """

        out = np.copy(d_img)

        if self.cleaner_min > 0:
            out = utils.cleanup_depth(d_img, self.cleaner_min)
        
        if self.strat == 'FMM':
            out = utils.inpaint(out, out == 0, radius=self.rad_d)
            return out
        
        if self.reg_interpolation:
            filled = utils.hole_interpolation(out)
            blured = utils.smoothen_filled_holes(out, filled)
            reg_rgb = utils.register_rgb(rgb_img, blured, self.cam_params)
        else:
            reg_rgb = utils.register_rgb(rgb_img, out, self.cam_params)

        if self.strat == 'GFMM':
            inpainted_rgb = utils.inpaint(reg_rgb, np.all(reg_rgb == 0, axis=2), radius=self.rad_rgb)
        elif self.strat == 'interGFMM':
            inpainted_rgb = utils.inpaint(reg_rgb, np.all(reg_rgb == 0, axis=2), guide=out, radius=self.rad_rgb)
        
        out = utils.inpaint(out, out == 0, guide=inpainted_rgb, radius=self.rad_d)

        return out

    def generate(self, d_dirs: list, rgb_dirs: list, dst_dirs: list, pattern: str = '*.png', exist_ok=False) -> list:
        """Generates multiple target depth images (from disk to disk)

        Parameters
        ----------
        d_dirs : list
            Directories with depth sequences
        
        rgb_dirs : list
            Directories with matching RGB sequences
        
        dst_dirs: list
            Destination directories where the generated targets are stored.
        
        pattern: str
            Matching pattern to retrieve images from inside the directories

        exist_ok: bool
            Control if a destination folder already existing counts as error
            (if exists it is ignored/not computed)
        
        Returns
        -------
        list
            Number of generated targets for each given folder
        """

        if len(d_dirs) == 0:
            raise ValueError('No input directories')
        if len(d_dirs) != len(rgb_dirs):
            raise ValueError('Number of depth directories and RGB directories not matching')
        if len(dst_dirs) != len(d_dirs):
            raise ValueError('Number of depth directories and destination directories not matching')
        
        for dir in d_dirs:
            if not os.path.isdir(dir):
                raise ValueError(f"Directory not found: {dir}")
        for dir in rgb_dirs:
            if not os.path.isdir(dir):
                raise ValueError(f"Directory not found: {dir}")
        if not exist_ok:
            for dir in dst_dirs:
                if os.path.isdir(dir):
                    raise ValueError(f"Destination directory already exists: {dir}")
        
        totals = []
        for d_f, rgb_f, dst_f in zip(d_dirs, rgb_dirs, dst_dirs):
            print(f'Generating to directory {str(dst_f)}')
            if exist_ok and os.path.isdir(dst_f):
                print('[Directory Already Exists. Skipping]\n')
                totals += [len(list(pathlib.Path(dst_f).glob('*.png')))]
                continue
            os.makedirs(dst_f)

            all_d_paths = sorted(list(pathlib.Path(d_f).glob(pattern)))
            all_rgb_paths = sorted(list(pathlib.Path(rgb_f).glob(pattern)))
            if len(all_d_paths) != len(all_rgb_paths):
                raise RuntimeError('Number of depth frames and RGB frames not matching')
            
            totals += [len(all_d_paths)]
            for i, (d_img_path, rgb_img_path) in enumerate(zip(all_d_paths, all_rgb_paths)):
                d_img = utils.read_img(d_img_path)
                rgb_img = utils.read_img(rgb_img_path)
                
                target_img = self.generate_one(d_img, rgb_img)
                utils.write_img(target_img, dst_f / d_img_path.name)

                print(f'{i+1}/{totals[-1]} - {(i+1)/totals[-1]*100:.2f}%   ', end='\r')
            print(f'{totals[-1]}/{totals[-1]} - 100.00%\t\t\t\n')
        print('[All Done]')

        return totals
