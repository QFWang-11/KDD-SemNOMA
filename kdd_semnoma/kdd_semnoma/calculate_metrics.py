import pyiqa
import torch
import tempfile
import shutil
from pathlib import Path
from utils import util_image
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

'''
class ImageDataset(Dataset):
    def __init__(self, gt_paths, stage1_paths):
        self.gt_paths = gt_paths
        self.stage1_paths = stage1_paths

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, idx):
        gt_img = util_image.imread(str(self.gt_paths[idx]), chn='rgb').transpose(2, 0, 1)
        stage1_img = util_image.imread(str(self.stage1_paths[idx]), chn='rgb').transpose(2, 0, 1)
        return (torch.from_numpy(gt_img).float(),
                torch.from_numpy(stage1_img).float())
        
def evaluate_image_quality(gt_dir, stage1_dir, ref_dir=None, batch_size=32, device='cuda'):
    # 获取所有图像路径
    gt_files = {p.name: p for p in Path(gt_dir).glob('*.png')}
    stage1_files = {p.name: p for p in Path(stage1_dir).glob('*.png')}

    # 提取基名并匹配路径
    gt_base_names = {p.stem.replace('_gt', '') for p in gt_files.values() if p.name.endswith('_gt.png')}
    matched_paths = []
    for base_name in gt_base_names:
        gt_name = f'{base_name}_gt.png'
        stage1_name = f'{base_name}_stage1.png'
        if (gt_name in gt_files and  
            stage1_name in stage1_files):
            matched_paths.append((
                gt_files[gt_name],
                stage1_files[stage1_name]
            ))
    
    # 限制样本量
    num_samples = min(len(matched_paths), 100 * batch_size)
    matched_paths = matched_paths[:num_samples]
    gt_paths, stage1_paths = zip(*matched_paths)
    gt_paths, stage1_paths = list(gt_paths), list(stage1_paths)

    if not gt_paths:
        raise ValueError("没有找到匹配的图像对，请检查目录中的文件命名和数量。")

    # 创建数据集和DataLoader
    dataset = ImageDataset(gt_paths, stage1_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化指标
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    # 累积指标
    metrics_sum = {
        'PSNR': 0.0, 'SSIM': 0.0, 'LPIPS': 0.0, 'NIQE': 0.0,
        'Stage1_PSNR': 0.0, 'Stage1_SSIM': 0.0, 'Stage1_LPIPS': 0.0, 'Stage1_NIQE': 0.0
    }
    num_batches = 0

    for gt_batch, stage1_batch in dataloader:
        gt_batch, stage1_batch = gt_batch.to(device), stage1_batch.to(device)

        # 计算指标
        metrics_sum['Stage1_PSNR'] += psnr_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_SSIM'] += ssim_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_LPIPS'] += lpips_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_NIQE'] += niqe_metric(stage1_batch).mean().item()

        num_batches += 1
        torch.cuda.empty_cache()

    # 平均指标
    metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    # FID计算
    with tempfile.TemporaryDirectory() as tmp_stage1_dir, \
         tempfile.TemporaryDirectory() as tmp_ref_dir:
        for i, p in enumerate(stage1_paths):
            shutil.copy(p, Path(tmp_stage1_dir)/f"stage1_{i:04d}.png")
        ref_paths = sorted(Path(ref_dir).glob('*.png'))[:len(gt_paths)] if ref_dir else gt_paths
        for i, p in enumerate(ref_paths):
            shutil.copy(p, Path(tmp_ref_dir)/f"ref_{i:04d}.png")
        fid_metric = pyiqa.create_metric('fid', device=device)
        metrics['Stage1_FID'] = fid_metric(tmp_stage1_dir, tmp_ref_dir).item()

    # 打印结果
    print(f"\n一阶段结果指标（样本量：{len(gt_paths)}）：")
    print(f"PSNR: {metrics['Stage1_PSNR']:.2f} dB | SSIM: {metrics['Stage1_SSIM']:.4f}")
    print(f"LPIPS: {metrics['Stage1_LPIPS']:.4f} | NIQE: {metrics['Stage1_NIQE']:.2f}")
    print(f"FID: {metrics['Stage1_FID']:.2f}")

    return metrics
'''
#生成结果与真实图像不匹配，可能由于文件名不一致，图像数量不足或者排序问题，修改后不再依赖简单的排序和截取，通过文件名中的共同基名显示匹配图像路径
'''
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
'''
'''
def evaluate_image_quality(gt_dir, gen_dir, stage1_dir, ref_dir=None, batch_size=32, device='cuda'):
    # 获取文件路径
    gt_paths = sorted(Path(gt_dir).glob('*.png'))[:80 * batch_size]
    gen_paths = sorted(Path(gen_dir).glob('*.png'))[:80 * batch_size]
    stage1_paths = sorted(Path(stage1_dir).glob('*.png'))[:80 * batch_size]

    # 创建数据集和 DataLoader
    dataset = ImageDataset(gt_paths, gen_paths, stage1_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 初始化指标
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    # 累积指标
    metrics_sum = {
        'PSNR': 0.0, 'SSIM': 0.0, 'LPIPS': 0.0, 'NIQE': 0.0,
        'Stage1_PSNR': 0.0, 'Stage1_SSIM': 0.0, 'Stage1_LPIPS': 0.0, 'Stage1_NIQE': 0.0
    }
    num_batches = 0

    for gt_batch, gen_batch, stage1_batch in dataloader:
        gt_batch, gen_batch, stage1_batch = gt_batch.to(device), gen_batch.to(device), stage1_batch.to(device)

        # 计算指标
        metrics_sum['PSNR'] += psnr_metric(gen_batch, gt_batch).mean().item()
        metrics_sum['SSIM'] += ssim_metric(gen_batch, gt_batch).mean().item()
        metrics_sum['LPIPS'] += lpips_metric(gen_batch, gt_batch).mean().item()
        metrics_sum['NIQE'] += niqe_metric(gen_batch).mean().item()
        metrics_sum['Stage1_PSNR'] += psnr_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_SSIM'] += ssim_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_LPIPS'] += lpips_metric(stage1_batch, gt_batch).mean().item()
        metrics_sum['Stage1_NIQE'] += niqe_metric(stage1_batch).mean().item()

        num_batches += 1
        # 清理显存
        torch.cuda.empty_cache()

    # 平均指标
    metrics = {k: v / num_batches for k, v in metrics_sum.items()}

    # FID 计算（保持原逻辑）
    with tempfile.TemporaryDirectory() as tmp_gen_dir, \
         tempfile.TemporaryDirectory() as tmp_stage1_dir, \
         tempfile.TemporaryDirectory() as tmp_ref_dir:
        for i, p in enumerate(gen_paths):
            shutil.copy(p, Path(tmp_gen_dir)/f"gen_{i:04d}.png")
        for i, p in enumerate(stage1_paths):
            shutil.copy(p, Path(tmp_stage1_dir)/f"stage1_{i:04d}.png")
        ref_paths = sorted(Path(ref_dir).glob('*.png'))[:80 * batch_size] if ref_dir else gt_paths
        for i, p in enumerate(ref_paths):
            shutil.copy(p, Path(tmp_ref_dir)/f"ref_{i:04d}.png")
        fid_metric = pyiqa.create_metric('fid', device=device)
        metrics['FID'] = fid_metric(tmp_gen_dir, tmp_ref_dir).item()
        metrics['Stage1_FID'] = fid_metric(tmp_stage1_dir, tmp_ref_dir).item()

    # 打印结果（保持原逻辑）
    print(f"\n最终生成结果指标（样本量：{len(gt_paths)}）：")
    print(f"PSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f}")
    print(f"LPIPS: {metrics['LPIPS']:.4f} | NIQE: {metrics['NIQE']:.2f}")
    print(f"FID: {metrics['FID']:.2f}")
    print(f"\n一阶段结果指标（样本量：{len(gt_paths)}）：")
    print(f"PSNR: {metrics['Stage1_PSNR']:.2f} dB | SSIM: {metrics['Stage1_SSIM']:.4f}")
    print(f"LPIPS: {metrics['Stage1_LPIPS']:.4f} | NIQE: {metrics['Stage1_NIQE']:.2f}")
    print(f"FID: {metrics['Stage1_FID']:.2f}")

    return metrics
'''
'''
def evaluate_image_quality(gt_dir, gen_dir, stage1_dir, ref_dir=None, batch_size=32, device='cuda'):
    """计算前10个batch的图像质量指标"""
    # 初始化指标计算器
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    # 获取并验证文件路径
    gt_paths = sorted(Path(gt_dir).glob('*.png'))
    gen_paths = sorted(Path(gen_dir).glob('*.png'))
    stage1_paths = sorted(Path(stage1_dir).glob('*.png'))
    #assert len(gt_paths) == len(gen_paths), "GT和生成图片数量不一致"
    assert gt_dir != gen_dir, "GT和生成目录不能相同"

    # 计算要处理的样本总数（10个batch）
    total_samples = 30 * batch_size
    gt_paths = gt_paths[:total_samples]
    gen_paths = gen_paths[:total_samples]
    stage1_paths = stage1_paths[:total_samples]
    
    # 读取图像数据到Tensor
    gt_tensors, gen_tensors, stage1_tensors = [], [], []
    for gt_p, gen_p, stage1_p in zip(gt_paths, gen_paths, stage1_paths):
        gt_img = util_image.imread(str(gt_p), chn='rgb').transpose(2, 0, 1)
        gen_img = util_image.imread(str(gen_p), chn='rgb').transpose(2, 0, 1)
        stage1_img = util_image.imread(str(stage1_p), chn='rgb').transpose(2, 0, 1)
        
        gt_tensors.append(torch.from_numpy(gt_img).unsqueeze(0))
        gen_tensors.append(torch.from_numpy(gen_img).unsqueeze(0))
        stage1_tensors.append(torch.from_numpy(stage1_img).unsqueeze(0))

    gt_tensor = torch.cat(gt_tensors).float().to(device)
    gen_tensor = torch.cat(gen_tensors).float().to(device)
    stage1_tensor = torch.cat(stage1_tensors).float().to(device)

    print(f"[数据范围验证] GT tensor - Min:{gt_tensor.min().item():.3f} Max:{gt_tensor.max().item():.3f}")
    print(f"[数据范围验证] Gen tensor - Min:{gen_tensor.min().item():.3f} Max:{gen_tensor.max().item():.3f}")
    print(f"[数据范围] Stage1: [{stage1_tensor.min().item():.3f}, {stage1_tensor.max().item():.3f}]")
    
    # 计算基础指标
    metrics = {
        'PSNR': psnr_metric(gen_tensor, gt_tensor).mean().item(),
        'SSIM': ssim_metric(gen_tensor, gt_tensor).mean().item(),
        'LPIPS': lpips_metric(gen_tensor, gt_tensor).mean().item(),
        'NIQE': niqe_metric(gen_tensor).mean().item(),
        
        # 一阶段结果指标
        'Stage1_PSNR': psnr_metric(stage1_tensor, gt_tensor).mean().item(),
        'Stage1_SSIM': ssim_metric(stage1_tensor, gt_tensor).mean().item(),
        'Stage1_LPIPS': lpips_metric(stage1_tensor, gt_tensor).mean().item(),
        'Stage1_NIQE': niqe_metric(stage1_tensor).mean().item(),
    }

    # 计算FID（需要临时目录）
    with tempfile.TemporaryDirectory() as tmp_gen_dir, \
         tempfile.TemporaryDirectory() as tmp_stage1_dir, \
         tempfile.TemporaryDirectory() as tmp_ref_dir:

        # 保存生成图片
        for i, p in enumerate(gen_paths):
            shutil.copy(p, Path(tmp_gen_dir)/f"gen_{i:04d}.png")
        # 保存一阶段图片
        for i, p in enumerate(stage1_paths):
            shutil.copy(p, Path(tmp_stage1_dir)/f"stage1_{i:04d}.png")

        # 准备参考集（优先使用外部参考集）
        ref_paths = sorted(Path(ref_dir).glob('*.png')) if ref_dir else gt_paths
        ref_paths = ref_paths[:total_samples]
        for i, p in enumerate(ref_paths):
            shutil.copy(p, Path(tmp_ref_dir)/f"ref_{i:04d}.png")

        # 计算FID
        metrics['FID'] = fid_metric(tmp_gen_dir, tmp_ref_dir).item()
        metrics['Stage1_FID'] = fid_metric(tmp_stage1_dir, tmp_ref_dir).item()

    # 打印结果
    print(f"\n最终生成结果指标（样本量：{total_samples}）：")
    print(f"PSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f}")
    print(f"LPIPS: {metrics['LPIPS']:.4f} | NIQE: {metrics['NIQE']:.2f}")
    print(f"FID: {metrics['FID']:.2f}")

    print(f"\n一阶段结果指标（样本量：{total_samples}）：")
    print(f"PSNR: {metrics['Stage1_PSNR']:.2f} dB | SSIM: {metrics['Stage1_SSIM']:.4f}")
    print(f"LPIPS: {metrics['Stage1_LPIPS']:.4f} | NIQE: {metrics['Stage1_NIQE']:.2f}")
    print(f"FID: {metrics['Stage1_FID']:.2f}")

    return metrics
'''
# 使用示例
if __name__ == "__main__":
    gt_dir = "/mnt/d/wqf/DifFace-master/output_cifar10_ddim_0dB_awgn/ground_truth"
    gen_dir = "/mnt/d/wqf/DifFace-master/output_cifar10_ddim_0dB_awgn/restored_faces"
    stage1_dir = "/mnt/d/wqf/DifFace-master/output_cifar10_ddim_0dB_awgn/stage1_outputs"
    ref_dir = "/mnt/d/wqf/DifFace-master/torch_dataset/cifar/train"  # 可选
    #ref_dir = "/mnt/d/wqf/FFHQ/FFHQ_256/val/"
    
    metrics = evaluate_image_quality(
        gt_dir=gt_dir,
        gen_dir=gen_dir,
        stage1_dir=stage1_dir,
        ref_dir=ref_dir,
        batch_size=32,
        device='cuda:1'
    )

# 主脚本中的调用保持不变，只需替换 evaluate_image_quality
# 示例调用
#gt_dir = "/root/DifFace-master/output_dir/ground_truth"
#gen_dir = "/root/DifFace-master/output_dir/restored_faces"
#ref_dir = None
#batch_size = 32
#device = 'cuda'

#batch_0_metrics = evaluate_image_quality_1(gt_dir, gen_dir, ref_dir, batch_size=batch_size, device=device)
#print(f"Batch 0 Metrics: PSNR={batch_0_metrics['PSNR']:.2f},  "
#      f"NIQE={batch_0_metrics['NIQE']:.2f}")