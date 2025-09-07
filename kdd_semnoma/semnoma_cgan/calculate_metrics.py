import pyiqa
import torch
import tempfile
import shutil
from pathlib import Path
from utils_eval import util_image
from torch.utils.data import DataLoader, Dataset
#from lpips import LPIPS

class ImageDataset(Dataset):
    def __init__(self, gt_paths, gen_paths, stage1_paths):
        self.gt_paths = gt_paths
        self.gen_paths = gen_paths
        self.stage1_paths = stage1_paths

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_img = util_image.imread(str(self.gt_paths[idx]), chn='rgb').transpose(2, 0, 1)
        gen_img = util_image.imread(str(self.gen_paths[idx]), chn='rgb').transpose(2, 0, 1)
        stage1_img = util_image.imread(str(self.stage1_paths[idx]), chn='rgb').transpose(2, 0, 1)
        return (torch.from_numpy(gt_img).float(),
                torch.from_numpy(gen_img).float(),
                torch.from_numpy(stage1_img).float())


def evaluate_image_quality(gt_dir, gen_dir, stage1_dir, ref_dir=None, batch_size=32, device='cuda'):
    # 获取所有图像路径
    gt_files = {p.name: p for p in Path(gt_dir).glob('*.png')}
    gen_files = {p.name: p for p in Path(gen_dir).glob('*.png')}
    stage1_files = {p.name: p for p in Path(stage1_dir).glob('*.png')}

    # 提取基名并匹配路径
    gt_base_names = {p.stem.replace('_gt', '') for p in gt_files.values() if p.name.endswith('_gt.png')}
    matched_paths = []
    for base_name in gt_base_names:
        gt_name = f'{base_name}_gt.png'
        gen_name = f'{base_name}_restored.png'
        stage1_name = f'{base_name}_stage1.png'
        if (gt_name in gt_files and 
            gen_name in gen_files and 
            stage1_name in stage1_files):
            matched_paths.append((
                gt_files[gt_name],
                gen_files[gen_name],
                stage1_files[stage1_name]
            ))
    
    # 限制样本量
    num_samples = min(len(matched_paths), 100 * batch_size)
    matched_paths = matched_paths[:num_samples]
    gt_paths, gen_paths, stage1_paths = zip(*matched_paths)
    gt_paths, gen_paths, stage1_paths = list(gt_paths), list(gen_paths), list(stage1_paths)

    if not gt_paths:
        raise ValueError("没有找到匹配的图像对，请检查目录中的文件命名和数量。")

    # 创建数据集和DataLoader
    dataset = ImageDataset(gt_paths, gen_paths, stage1_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化指标
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    #lpips_metric = LPIPS(net='squeeze').to(device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    # 累积指标
    metrics_sum = {
        'PSNR': 0.0, 'SSIM': 0.0, 'LPIPS': 0.0, 'NIQE': 0.0,
        'Stage1_PSNR': 0.0, 'Stage1_SSIM': 0.0, 'Stage1_LPIPS': 0.0, 'Stage1_NIQE': 0.0
    }
    num_batches = 0

    for gt_batch, gen_batch, stage1_batch in dataloader:
        gt_batch, gen_batch, stage1_batch = gt_batch.to(device), gen_batch.to(device), stage1_batch.to(device)
        #print("gen_batch 形状:", gen_batch.shape)
        # 计算指标
        metrics_sum['PSNR'] += psnr_metric(gen_batch, gt_batch).mean().item()
        metrics_sum['SSIM'] += ssim_metric(gen_batch, gt_batch).mean().item()
        metrics_sum['LPIPS'] += lpips_metric(gen_batch, gt_batch).mean().item()
        #metrics_sum['NIQE'] += niqe_metric(gen_batch).mean().item()
        metrics_sum['Stage1_PSNR'] += psnr_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_SSIM'] += ssim_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_LPIPS'] += lpips_metric(stage1_batch, gt_batch).mean().item()
        #metrics_sum['Stage1_NIQE'] += niqe_metric(stage1_batch).mean().item()

        num_batches += 1
        torch.cuda.empty_cache()

    # 平均指标
    metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    # FID计算
    with tempfile.TemporaryDirectory() as tmp_gen_dir, \
         tempfile.TemporaryDirectory() as tmp_stage1_dir, \
         tempfile.TemporaryDirectory() as tmp_ref_dir:
        for i, p in enumerate(gen_paths):
            shutil.copy(p, Path(tmp_gen_dir)/f"gen_{i:04d}.png")
        for i, p in enumerate(stage1_paths):
            shutil.copy(p, Path(tmp_stage1_dir)/f"stage1_{i:04d}.png")
        ref_paths = sorted(Path(ref_dir).glob('*.png'))[:len(gt_paths)] if ref_dir else gt_paths
        for i, p in enumerate(ref_paths):
            shutil.copy(p, Path(tmp_ref_dir)/f"ref_{i:04d}.png")
        fid_metric = pyiqa.create_metric('fid', device=device)
        metrics['FID'] = fid_metric(tmp_gen_dir, tmp_ref_dir).item()
        metrics['Stage1_FID'] = fid_metric(tmp_stage1_dir, tmp_ref_dir).item()

    # 打印结果
    print(f"\n最终生成结果指标（样本量：{len(gt_paths)}）：")
    print(f"PSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f}")
    print(f"LPIPS: {metrics['LPIPS']:.4f} | NIQE: {metrics['NIQE']:.2f}")
    print(f"FID: {metrics['FID']:.2f}")
    print(f"\n一阶段结果指标（样本量：{len(gt_paths)}）：")
    print(f"PSNR: {metrics['Stage1_PSNR']:.2f} dB | SSIM: {metrics['Stage1_SSIM']:.4f}")
    print(f"LPIPS: {metrics['Stage1_LPIPS']:.4f} | NIQE: {metrics['Stage1_NIQE']:.2f}")
    print(f"FID: {metrics['Stage1_FID']:.2f}")

    return metrics


if __name__ == "__main__":
    gt_dir = r"D:\wqf\Deep_GAN_decomposition\output_15dB_rayleigh\ground_truth"
    gen_dir = r"D:\wqf\Deep_GAN_decomposition\output_15dB_rayleigh\restored_faces"
    stage1_dir = r"D:\wqf\Deep_GAN_decomposition\output_15dB_rayleigh\stage1_outputs"
    ref_dir = r"D:\wqf\FFHQ\FFHQ_256\val"  
    
    metrics = evaluate_image_quality(
        gt_dir=gt_dir,
        gen_dir=gen_dir,
        stage1_dir=stage1_dir,
        ref_dir=ref_dir,
        batch_size=32,
        device='cuda:1'
    )