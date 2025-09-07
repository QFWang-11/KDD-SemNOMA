import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from sampler import DifFaceSampler, Stage1Sampler
from basicsr.utils.download_util import load_file_from_url


# Define constant dictionaries for start timesteps and gamma values for different tasks
_START_TIMESTEPS = {'restoration':10 , 'inpainting': 120}  # IDDPM sampling acceleration
_GAMMA = {'restoration': 0.0, 'inpainting': 0.5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "-i",
            "--in_path",
            type=str,
            default='./testdata/cropped_faces',
            help='Folder to save the low quality image',
            )
    parser.add_argument(
            "-o",
            "--out_path",
            type=str,
            default='./results',
            help='Folder to save the restored results',
            )
    parser.add_argument(
            "--aligned",
            action='store_true',
            help='Input are alinged faces',
            )
    parser.add_argument(
            "--use_fp16",
            action='store_true',
            help='Activate float16 for inference',
            )
    parser.add_argument(
            "--task",
            type=str,
            default='restoration',
            choices=['restoration', 'inpainting'],
            help='Task',
            )
    parser.add_argument(
            "--eta",
            type=float,
            default=0.5,
            help='Hyper-parameter eta in ddim',
            )
    parser.add_argument(
            "--bs",
            type=int,
            default=16,
            help='Batch size for inference',
            )
    parser.add_argument(
            "--seed",
            type=int,
            default=12345,
            help='Random Seed',
            )
    parser.add_argument(
            "--draw_box",
            action='store_true',
            help='Draw box for face in the unaligned case',
            )
    parser.add_argument(
        "--model_ir_ckpt",
        type=str,
        default='path/to/your/model_ir_checkpoint.pth',  # replace with the stage1 model path
        help='Path to your custom model_ir checkpoint',
    )
    parser.add_argument("--gt_save_dir", type=str, default=None, help='Folder to save ground truth images (optional)')
    args = parser.parse_args()

    # configurations
    if args.task == 'restoration':
        #cfg_path = 'configs/sample/difface_inpainting_lama256.yaml'
        cfg_path = 'configs/sample/iddpm_cifar10.yaml'
    elif args.task == 'inpainting':
        cfg_path = 'configs/sample/difface_inpainting_lama256.yaml'
    else:
        raise ValueError("Only accept task types of 'restoration' and 'inpainting'!")

    # setting configurations
    configs = OmegaConf.load(cfg_path)
    configs.seed = args.seed
    configs.diffusion.params.timestep_respacing = 'ddim250'  # 4x acceleration
    configs.model_ir.ckpt_path = Path(args.model_ir_ckpt)    # normalize path

    # prepare the checkpoint
    if args.task == 'restoration':
        configs.aligned = args.aligned
    elif args.task == 'inpainting':
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq256_ema750000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
                )
        configs.aligned = True
    else:
        raise ValueError("Only accept task types of 'restoration' and 'inpainting'!")

    if not configs.aligned and args.bs != 1:
        args.bs = 1
        print("Resetting batchsize to be 1 for unaligned case.")
    
    # Build DiffusionFaceSampler instance and perform inference
    # build the sampler for diffusion
    '''
    sampler = Stage1Sampler(configs)
    stage1_dir, gt_dir = sampler.inference_stage1(
        stage1_save_dir='/mnt/d/wqf/DifFace-master/output_dir_10_dB_DDIM_20_MU_AWGN/stage1_outputs',
        gt_save_dir='/mnt/d/wqf/DifFace-master/output_dir_10_dB_DDIM_20_MU_AWGN/ground_truth'
    )
    '''
    sampler_dist = DifFaceSampler(
            configs,
            im_size=configs.model.params.image_size,
            use_fp16=args.use_fp16,
            )
    gt_dir, stage1_dir, restored_face_dir = sampler_dist.inference(
            in_path=args.in_path,
            out_path=args.out_path,
            bs=args.bs,
            start_timesteps=_START_TIMESTEPS[args.task],  # for restoration task, start_timestep=100
            task=args.task,
            need_restoration=True,
            gamma=_GAMMA[args.task],
            num_update=1,
            draw_box=args.draw_box,
            suffix=None,
            eta=args.eta if args.task =='restoration' else 1.0,  # 1.0 for random sampling same as DDPM, increases generation randomness
            mask_back=True,
            gt_save_dir=args.gt_save_dir,
            )

if __name__ == '__main__':
    main()
