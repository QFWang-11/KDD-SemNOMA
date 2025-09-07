import os, math, random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from collections import OrderedDict

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from kd_semnoma.models.SemNOMA_with_KD import SemNOMA_KD_AWGN, SemNOMA_KD_Rayleigh
from kd_semnoma.datasets.ffhq256_group_wrapper import test_loader
#from kd_semnoma.datasets.cifar10_group_wrapper import test_loader

class BaseSampler:
    def __init__(self, configs, im_size=512, use_fp16=False):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/sample/
        '''
        self.configs = configs
        self.configs.im_size = im_size
        self.configs.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if hasattr(self.configs.model.params, 'use_fp16'):
            self.configs.model.params.use_fp16 = use_fp16

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()    # setup seed

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.configs.seed if seed is None else seed
        seed += (self.rank) * 10000
        if self.rank == 0:
            print(f'Setting random seed {seed}', flush=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    def setup_dist(self):
        """Initialize distributed training settings"""
        self.num_gpus = torch.cuda.device_count()
        
        # Default single GPU setup
        self.rank = 0
        torch.cuda.set_device(0)
        
        # Multi-GPU distributed training initialization
        if self.num_gpus > 1:
            # Check if running in distributed environment
            if 'LOCAL_RANK' in os.environ:
                self.rank = int(os.environ['LOCAL_RANK'])
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method('spawn')
                torch.cuda.set_device(self.rank)
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=self.num_gpus,
                    rank=self.rank
                )
            else:
                print('Warning: Multiple GPUs detected but not running in distributed mode. '
                     'To enable distributed training, use torch.distributed.launch or torchrun.')
    
    def build_model(self):
        obj = util_common.get_obj_from_str(self.configs.model.target)
        model = obj(**self.configs.model.params).cuda()
        if not self.configs.model.ckpt_path is None:
            self.load_model(model, self.configs.model.ckpt_path)
        if self.configs.use_fp16:
            model.convert_to_fp16()
        self.model = model
        self.freeze_model(self.model)
        self.model.eval()

    def load_model(self, model, ckpt_path=None):
        if self.rank == 0:
            print(f'Loading from {ckpt_path}...', flush=True)
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
        if self.rank == 0:
            print('Loaded Done', flush=True)

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

class DifFaceSampler(BaseSampler):
    def restore_func(self, y0, csi, model_kwargs_ir):
        # y0: (batchsize, num_devices, c, h, w)
        # csi: (batchsize, 1)
        if model_kwargs_ir is None:
            inputs = y0
        elif 'mask' in model_kwargs_ir:
            inputs = torch.cat([y0, model_kwargs_ir['mask']], dim=1)
        else:
            raise ValueError("Not compativle type for model_kwargs!")

        original_dtype = y0.dtype
        with torch.no_grad():
            # Ensure input type matches model
            # Get model dtype
            out = self.model_ir(y0.type(self.dtype), csi.type(self.dtype)).type(original_dtype)  # Stage 1 model

        return out

    def build_model(self):
        super().build_model()   # Override parent class

        obj = util_common.get_obj_from_str(self.configs.diffusion.target)
        self.diffusion = obj(**self.configs.diffusion.params)    # Dynamically load model class and instantiate model
        print("Diffusion model initialized:", self.diffusion.__class__.__name__)
        
        # Replace diffused estimator with our own model for visualization
        model_ir = SemNOMA_KD_AWGN(num_devices=2, M=32, teacher_checkpoint_path='/mnt/d/wqf/DifFace-master/pretrain_stage1_models/UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders.pth', cross_layer_idx=4).cuda()          
        #model_ir = SemNOMA_KD_Rayleigh(num_devices=2, M=32, teacher_checkpoint_path='/mnt/d/wqf/DifFace-master/pretrain_stage1_models/UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders_Rayleigh_FFHQ256_54.pth', cross_layer_idx=4).cuda()          
        # Load pre-trained weights for Stage 1 network
        if self.configs.model_ir.ckpt_path:
            self.load_model(model_ir, self.configs.model_ir.ckpt_path)
        if self.configs.use_fp16:
            model_ir = model_ir.half()
        self.model_ir = model_ir
        self.model_ir.eval()

        # for restoration
        if 'net_hq' in self.configs:
            ckpt_path = self.configs.net_hq.ckpt_path
            params = self.configs.net_hq.get('params', dict)
            net_hq = util_common.get_obj_from_str(self.configs.net_hq.target)(**params)
            self.net_hq = net_hq.cuda()
            self.load_model(self.net_hq, ckpt_path)
            self.net_hq.eval()
        if 'net_lq' in self.configs:
            ckpt_path = self.configs.net_lq.ckpt_path
            params = self.configs.net_lq.get('params', dict)
            net_lq = util_common.get_obj_from_str(self.configs.net_lq.target)(**params)
            self.net_lq = net_lq.cuda()
            self.load_model(self.net_lq, ckpt_path)
            self.net_lq.eval()
            self.freeze_model(self.net_lq)

    def sample_func_ir_aligned(
            self,
            y0,
            csi,
            start_timesteps=None,
            need_restoration=True,
            model_kwargs_ir=None,
            gamma=0.,
            eta=0.,
            num_update=1,
            regularizer=None,
            cond_kwargs=None,
            save_stage1_dir=None,  # Directory to save Stage 1 outputs
            ):
        '''
        Input low-quality image y0, initialize related parameters.
        Resize image to required model size if needed.
        If need_restoration is True, call restore_func to repair image.
        Use diffusion model to perform noise sampling on the repaired image.
        Use diffusion model's DDIM sampling method ddim_sample_loop to generate high-quality image samples.
        Restore generated samples to original image size and return.
        '''
        '''
        Input:
            y0: b x c x h x w torch tensor, low-quality image, [0, 1], RGB, float32
            csi: SNR for stage1, b x 1, float32
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 'ddim250'), range [0, 249]
            model_kwargs_ir: additional parameters for restoration model
            gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
            num_update: number of update for x_start based on Eq. (1)
            regularizer: the constraint for R in Eq. (1)
            cond_kwargs: extra params for the regrlarizer
            eta: hyper-parameter eta for ddim
        Output:
            sample: n x c x h x w, torch tensor, [0,1], RGB
        '''
        if start_timesteps is None:
            start_timesteps = self.diffusion.num_timesteps

        # basic image restoration
        # y0: (batchsize, num_devices, c, h, w)
        device = next(self.model.parameters()).device
        y0 = y0.to(device=device)
        csi = csi.to(device=device)

        #h_old, w_old = y0.shape[2:4]
        b, n, c, h, w = y0.shape
        h_old, w_old = h, w
        if not (h_old == self.configs.im_size and w_old == self.configs.im_size):
            y0 = F.interpolate(y0, size=(self.configs.im_size,)*2, mode='bicubic', antialias=True)
            if (cond_kwargs is not None) and ('mask' in cond_kwargs):
                cond_kwargs['mask'] = detect_mask(y0, thres=0)

        if need_restoration:
            im_hq = self.restore_func(y0, csi, model_kwargs_ir)  # (batchsize * num_devices, c, h, w)
        else:
            im_hq = y0.view(b*n, c, h, w)
        im_hq.clamp_(0.0, 1.0)

        # diffuse for im_hq
        yt = self.diffusion.q_sample(
                x_start=util_image.normalize_th(im_hq, mean=0.5, std=0.5, reverse=False),  # Stage 1 restored image
                t=torch.tensor([start_timesteps,]*im_hq.shape[0], device=device),
                )

        assert yt.shape[-1] == self.configs.im_size and yt.shape[-2] == self.configs.im_size
        #yt = im_hq   #without error contraction
        sample = self.diffusion.ddim_sample_loop(
                self.model,
                y0=util_image.normalize_th(y0.view(b*n,c,h,w), mean=0.5, std=0.5, reverse=False),
                shape=yt.shape,
                noise=yt,
                start_timesteps=start_timesteps,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs=None,
                device=None,
                progress=False,
                eta=eta,
                gamma=0.,
                num_update=num_update,
                regularizer=regularizer,
                cond_kwargs=cond_kwargs,
                )
        sample = util_image.normalize_th(sample, reverse=True).clamp(0.0, 1.0)

        if not (h_old == self.configs.im_size and w_old == self.configs.im_size):
            sample = F.interpolate(sample, size=(h_old, w_old), mode='bicubic', antialias=True).clamp(0.0, 1.0)

        return sample, im_hq

    def sample_func_ir_unaligned(
            self,
            y0,
            micro_bs=16,
            start_timesteps=None,
            need_restoration=True,
            eta=0.0,
            draw_box=False,
            ):
        '''
        Input:
            y0: h x w x c numpy array, uint8, BGR, or image path
            micro_bs: batch size for face restoration
            upscale: upsampling factor for the restorated image
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 'ddim250'), range [0, 249]
            draw_box: draw a box for each face
            eta: hyper-parameter eat for ddim
        Output:
            restored_img: h x w x c, numpy array, uint8, BGR
            restored_faces: list, h x w x c, numpy array, uint8, BGR
        '''
        def _process_batch(cropped_faces_list):
            length = len(cropped_faces_list)
            cropped_face_t = torch.cat(
                    util_image.img2tensor(cropped_faces_list, bgr2rgb=True, out_type=self.dtype),
                    axis=0).cuda() / 255.
            restored_faces = self.sample_func_ir_aligned(
                    cropped_face_t,
                    start_timesteps=start_timesteps,
                    need_restoration=need_restoration,
                    gamma=0,
                    eta=eta,
                    model_kwargs_ir=None,
                    )[0]      # [0, 1], b x c x h x w
            return restored_faces

        assert not self.configs.aligned

        self.face_helper.clean_all()
        self.face_helper.read_image(y0)
        num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=False,
                resize=640,
                eye_dist_threshold=5,
                )
        if self.rank == 0:
            print(f'\tdetect {num_det_faces} faces', flush=True)
        # align and warp each face
        self.face_helper.align_warp_face()

        num_cropped_face = len(self.face_helper.cropped_faces)
        if num_cropped_face > micro_bs:
            restored_faces = []
            for idx_start in range(0, num_cropped_face, micro_bs):
                idx_end = idx_start + micro_bs if idx_start + micro_bs < num_cropped_face else num_cropped_face
                current_cropped_faces = self.face_helper.cropped_faces[idx_start:idx_end]
                current_restored_faces = _process_batch(current_cropped_faces)
                current_restored_faces = util_image.tensor2img(
                        list(current_restored_faces.split(1, dim=0)),
                        rgb2bgr=True,
                        min_max=(0, 1),
                        out_type=np.uint8,
                        )
                restored_faces.extend(current_restored_faces)
        else:
            restored_faces = _process_batch(self.face_helper.cropped_faces)
            restored_faces = util_image.tensor2img(
                    list(restored_faces.split(1, dim=0)),
                    rgb2bgr=True,
                    min_max=(0, 1),
                    out_type=np.uint8,
                    )
        for xx in restored_faces:
            self.face_helper.add_restored_face(xx)

        # paste_back
        bg_img = self.bg_model.enhance(
                self.face_helper.input_img,
                outscale=self.configs.detection.upscale,
                )[0]
        self.face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=draw_box,
                )

        return restored_img, restored_faces

    def inference(
            self,
            in_path,
            out_path,
            bs=1,
            start_timesteps=None,
            need_restoration=True,
            gamma=0.,
            num_update=1,
            task='restoration',
            draw_box=False,
            suffix=None,
            eta=0.0,
            mask_back=False,
            gt_save_dir=None,  # Directory to save ground truth images
            ):
        '''
        Input:
            in_path: testing image path or folder
            out_path: folder to save the retorated results
            bs: batch size, totally on all the GPUs
            start_timesteps: integer, range [0, num_timesteps-1],
                for accelerated sampling (e.g., 250), range [0, 249]
            need_restoration: degradation removal with diffused estimator
            gamma: hyper-parameter to regularize the predicted x_start
                    0.5 * || x - x_0 ||_2 + gamma * R(x_0, y_0)      (1)
            num_update: number of update for x_start based on Eq. (1)
            task: 'restoration' or 'inpainting'
                  For inpainting, we assumed that the masked area is initially filled with 0.
            cond_kwargs: extra params for the regrlarizer
            draw_box: draw a box for each face
            eta: eta for ddim
            mask_back: only for inparinting, lq * (1-mask) + res * mask
        '''
        def _process_batch_aligned(y0, csi, cond_kwargs, model_kwargs_ir):
            '''
            y0: (batch, num_devices, c, w h)
            csi: (batch, 1)
            '''
            sample, stage1_out = self.sample_func_ir_aligned(
                    y0,
                    csi,
                    start_timesteps,
                    need_restoration=need_restoration,
                    gamma=gamma,
                    num_update=num_update,
                    regularizer= masking_regularizer if task=='inpainting' else None,
                    cond_kwargs=cond_kwargs,
                    eta=eta,
                    model_kwargs_ir=model_kwargs_ir,
                    save_stage1_dir=out_path / 'stage1_outputs',  # Save Stage 1 outputs
                    )
            return sample, stage1_out

        def _process_batch_unaligned(y0):
            '''
            y0: image path or h x w x c numpy array, uint8, BGR
            '''
            restored_img, restored_faces = self.sample_func_ir_unaligned(
                    y0,
                    micro_bs=16,
                    start_timesteps=start_timesteps,
                    need_restoration=need_restoration,
                    draw_box=draw_box,
                    eta=eta,
                    )  # h x w x c, uint8, BGR
            return restored_img, restored_faces

        assert task in ['restoration', 'inpainting']
        if not self.configs.aligned:
            assert task == 'restoration', "Only support image restoration for unalinged image!"

        # prepare result path, convert to Path object
        in_path = in_path if isinstance(in_path, Path) else Path(in_path)
        out_path = out_path if isinstance(out_path, Path) else Path(out_path)
        restored_face_dir = out_path / 'restored_faces'  # Path to save restored images
        stage1_dir = out_path / 'stage1_outputs'  # Path to save Stage 1 outputs
        gt_dir = Path(gt_save_dir) if gt_save_dir else out_path / 'ground_truth'  # Default path to save ground truth images
        if self.rank == 0:  # Explicitly create directories
            util_common.mkdir(out_path, parents=True)
            util_common.mkdir(restored_face_dir, parents=True)
            util_common.mkdir(stage1_dir, parents=True)  
            util_common.mkdir(gt_dir, parents=True)     
        if not self.configs.aligned:
            restored_image_dir = out_path / 'restored_image'
            if self.rank == 0:
                util_common.mkdir(restored_image_dir, parents=True)

        if in_path.is_dir():
            if self.configs.aligned:
                '''
                dataset = BaseDataFolder(
                        dir_path=in_path,
                        transform_type='default',
                        transform_kwargs={'mean':0, 'std':1.0},
                        need_path=True,
                        im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                        )
                dataloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=bs,
                        shuffle=False,
                        )
                '''
                dataloader = test_loader[0]    # Test data at SNR=0dB
                if self.rank == 0:
                    print(f'Number of testing images: {len(dataloader)}', flush=True)
                for batch_idx, data in enumerate(tqdm(dataloader)):
                    y0, csi = data
                    batchsize, num_devices, channels, height, width = y0.shape
                    micro_batchsize = math.ceil(bs / self.num_gpus)
                    ind_start = self.rank * micro_batchsize
                    ind_end = ind_start + micro_batchsize
                    if ind_start < y0.shape[0]:
                        current_y0 = y0[ind_start:ind_end]
                        current_csi = csi[ind_start:ind_end]
                        #current_path = data['path'][ind_start:ind_end]
                        
                        # Save ground truth images
                        y0_flat = current_y0.reshape(-1, channels, height, width)  # (batchsize * num_devices, c, h, w)
                        for jj in range(y0_flat.shape[0]):
                            gt_img = y0_flat[jj].permute(1, 2, 0).cpu().numpy()  # h x w x c, [0,1], RGB
                            save_path = gt_dir / f'batch_{batch_idx}_{jj}_gt.png'
                            util_image.imwrite(gt_img, save_path, chn='rgb', dtype_in='float32')
                            
                        if task == 'inpainting':
                            mask = detect_mask(current_y0, thres=0)
                            cond_kwargs = model_kwargs_ir = {'mask': mask.cuda()}
                        else:
                            cond_kwargs = model_kwargs_ir = None

                        # Perform image restoration and sampling
                        sample, stage1_out = _process_batch_aligned(current_y0, current_csi, cond_kwargs, model_kwargs_ir)
                        # sample: (b*num_devices, c, h, w), [0,1], RGB
                        if (not cond_kwargs is None) and 'mask' in cond_kwargs and mask_back:
                            sample = sample * cond_kwargs['mask'] + current_y0.cuda() * (1-cond_kwargs['mask'])

                        # Save Stage 1 outputs
                        for jj in range(stage1_out.shape[0]):
                            stage1_img = stage1_out[jj].permute(1, 2, 0).cpu().numpy()
                            save_path = stage1_dir / f'batch_{batch_idx}_{jj}_stage1.png'
                            util_image.imwrite(stage1_img, save_path, chn='rgb', dtype_in='float32')

                        # Save diffusion restored images
                        for jj in range(sample.shape[0]):
                            restored_face = sample[jj].permute(1, 2, 0).cpu().numpy()  # h x w x c, [0,1], RGB
                            save_path = restored_face_dir / f'batch_{batch_idx}_{jj}_restored.png'
                            util_image.imwrite(restored_face, save_path, chn='rgb', dtype_in='float32')

            else:
                assert self.num_gpus == 1
                im_path_list = [x for x in in_path.glob('*.[jJpP][pPnN]*[gG]')]
                print(f'Number of testing images: {len(im_path_list)}', flush=True)

                for im_path_current in im_path_list:
                    restored_img, restored_faces = _process_batch_unaligned(str(im_path_current))  # h x w x c, uint8, BGR

                    if suffix == 'gamma':
                        save_path = restored_image_dir / f'{im_path_current.stem}_g{gamma:.2f}_s{start_timesteps}.png'
                    else:
                        save_path = restored_image_dir / f'{im_path_current.stem}.png'
                    util_image.imwrite(restored_img, save_path, chn='bgr', dtype_in='uint8')

                    assert isinstance(restored_faces, list)
                    for ii, restored_face in enumerate(restored_faces):
                        save_path = restored_face_dir / f'{im_path_current.stem}_{ii:03d}.png'
                        util_image.imwrite(restored_face, save_path, chn='bgr', dtype_in='uint8')
        else:
            y0 = util_image.imread(in_path, chn='rgb', dtype='float32')
            y0 = util_image.img2tensor(y0, bgr2rgb=False, out_type=torch.float32) # 1 x c x h x w, [0,1]
            if task == 'inpainting':
                mask = detect_mask(y0, thres=0)
                cond_kwargs = model_kwargs_ir = {'mask': mask.cuda()}
            else:
                cond_kwargs = model_kwargs_ir = None

            if self.configs.aligned:
                sample = _process_batch_aligned(y0, cond_kwargs, model_kwargs_ir)
                if (not cond_kwargs is None) and 'mask' in cond_kwargs and mask_back:
                    sample = sample * cond_kwargs['mask'] + y0.cuda() * (1-cond_kwargs['mask'])
                restored_face = sample.squeeze(0).permute(1,2,0).cpu().numpy()  # h x w x c, [0,1], RGB
                if suffix == 'gamma':
                    if 'ddim' in self.configs.diffusion.params.timestep_respacing:
                        save_path = restored_face_dir / f'{in_path.stem}_g{gamma:.2f}_ddim{start_timesteps}_e{eta:.1f}_n{num_update}.png'
                    else:
                        save_path = restored_face_dir / f'{in_path.stem}_g{gamma:.2f}_ddpm{start_timesteps}_n{num_update}.png'
                else:
                    save_path = restored_face_dir / f'{in_path.stem}.png'
                util_image.imwrite(restored_face, save_path, chn='rgb', dtype_in='float32')
            else:
                restored_img, restored_faces = _process_batch_unaligned(str(in_path))  # h x w x c, uint8, BGR
                if suffix == 'gamma':
                    save_path = restored_image_dir / f'{in_path.stem}_g{gamma:.2f}.png'
                else:
                    save_path = restored_face_dir / f'{in_path.stem}.png'
                util_image.imwrite(restored_img, save_path, chn='bgr', dtype_in='uint8')

                assert isinstance(restored_faces, list)
                for ii, restored_face in enumerate(restored_faces):
                    save_path = restored_face_dir / f'{in_path.stem}_{ii:03d}.png'
                    util_image.imwrite(restored_face, save_path, chn='bgr', dtype_in='uint8')

        if self.num_gpus > 1:
            dist.barrier()

        if self.rank == 0:
            print(f'Please enjoy the results in {str(out_path)}...', flush=True)

        return gt_dir, stage1_dir, restored_face_dir  # Return paths for ground truth, Stage 1 outputs, and restored images

class Stage1Sampler(BaseSampler):
    def restore_func(self, y0, csi, model_kwargs_ir):
        # y0:(batchsize, num_devices, c, w, h)
        # csi:(batchsize, 1)
        original_dtype = y0.dtype
        with torch.no_grad():
            # Stage 1 model inference
            out = self.model_ir(y0.type(self.dtype), csi.type(self.dtype)).type(original_dtype)
        return out

    def build_model(self):
        super().build_model()
        
        # Build only Stage 1 model
        model_ir = SemNOMA_KD_AWGN(num_devices=2, M=32,teacher_checkpoint_path='/mnt/d/wqf/DifFace-master/pretrain_stage1_models/UL_NOMA_2_Users_Perfect_SIC_A1_M_32_mult_1_no_bind_mae_embed_1decoders.pth',cross_layer_idx=4).cuda()
        
        if self.configs.model_ir.ckpt_path:
            self.load_model(model_ir, self.configs.model_ir.ckpt_path)
        if self.configs.use_fp16:
            model_ir = model_ir.half()
        self.model_ir = model_ir
        self.model_ir.eval()

    def inference_stage1(
        self,
        stage1_save_dir,
        gt_save_dir,
    ):
        """
        Simplified inference function that only performs Stage 1 inference
        Parameters:
            in_path: input data path
            stage1_save_dir: Stage 1 output save path
            gt_save_dir: ground truth images save path
            bs: batch size
        """
        # Prepare save paths
        stage1_dir = Path(stage1_save_dir)
        gt_dir = Path(gt_save_dir)
        util_common.mkdir(stage1_dir, parents=True)
        util_common.mkdir(gt_dir, parents=True)

        # Load test data (adjust according to actual data loader)
        dataloader = test_loader[10]  # Assume using SNR=-5dB test data
        
        for batch_idx, data in enumerate(tqdm(dataloader)):
            y0, csi = data  # Assume data format is (batch, devices, c, h, w)
            batchsize, num_devices, channels, height, width = y0.shape
            
            # Save ground truth images
            y0_flat = y0.reshape(-1, channels, height, width)
            for jj in range(y0_flat.shape[0]):
                gt_img = y0_flat[jj].permute(1, 2, 0).cpu().numpy()
                save_path = gt_dir / f'batch_{batch_idx}_{jj}_gt.png'
                util_image.imwrite(gt_img, save_path, chn='rgb', dtype_in='float32')
            
            # Stage 1 inference
            with torch.no_grad():
                device = next(self.model.parameters()).device
                y0 = y0.to(device=device)
                csi = csi.to(device=device)
                stage1_out = self.restore_func(y0, csi, None)  # (batch*devices, c, h, w)
                stage1_out = stage1_out.clamp(0.0, 1.0)
            
            # Save Stage 1 outputs
            for jj in range(stage1_out.shape[0]):
                stage1_img = stage1_out[jj].permute(1, 2, 0).cpu().numpy()
                save_path = stage1_dir / f'batch_{batch_idx}_{jj}_stage1.png'
                util_image.imwrite(stage1_img, save_path, chn='rgb', dtype_in='float32')

        print(f'Stage1 results saved to: {stage1_dir}')
        print(f'Ground truth saved to: {gt_dir}')
        return stage1_dir, gt_dir

@torch.enable_grad()
def masking_regularizer(y0, x0, cond_kwargs):
    '''
    Input:
        y0: low-quality image, b x c x h x w, [-1, 1]
        x0: predicted high-quality image, b x c x h x w, [-1, 1]
        cond_kwargs: additional network parameters.
    '''
    mask = cond_kwargs['mask']
    if 'vqgan' in cond_kwargs:
        pred = cond_kwargs['vqgan'].decode(x0)
    else:
        pred = x0

    loss = (F.mse_loss(pred, y0, reduction='none') * (1 - mask)).sum()

    return loss

def detect_mask(y0, thres):
    '''
    Input:
        y0: low-quality image, b x c x h x w , [0, 1]
    '''
    ysum = torch.sum(y0, dim=1, keepdim=True)
    mask = torch.where(ysum==thres, torch.ones_like(ysum), torch.zeros_like(ysum))
    return mask


