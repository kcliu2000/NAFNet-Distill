import sys
from os import path as osp
# 直接將根目錄加入系統路徑，徹底解決 ModuleNotFoundError
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

import logging
import torch
import numpy as np
from tqdm import tqdm

# 匯入指標計算套件
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


def main():
    opt = parse_options(is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'test' in phase:
            dataset_opt['phase'] = 'test'
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    model = create_model(opt)

    print("===> Loading LPIPS model...")
    loss_fn_lpips = lpips.LPIPS(net='alex').cuda()

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        
        # 自動抓取 results_root，如果沒有設定就預設放在 results/實驗名稱 下
        result_dir = opt['path'].get('results_root', f"./results/{opt['name']}")
        
        psnr_list, ssim_list, lpips_list = [], [], []

        model.net_g.eval()
        with torch.no_grad():
            for idx, val_data in enumerate(tqdm(test_loader)):
                model.feed_data(val_data)
                model.test()
                
                visuals = model.get_current_visuals()
                pred = visuals['result'].cuda()
                gt = visuals['gt'].cuda()
                
                pred_clip = torch.clamp(pred, 0, 1)

                img_path = val_data.get('lq_path', ['unknown'])[0]
                name = osp.splitext(osp.basename(img_path))[0]

                # 計算 LPIPS (-1 到 1)
                img_gt_lpips = gt * 2.0 - 1.0
                img_rest_lpips = pred_clip * 2.0 - 1.0
                lpips_score = loss_fn_lpips(img_rest_lpips, img_gt_lpips).item()

                # 計算 PSNR & SSIM (0 到 1, HWC, Numpy)
                pred_np = pred_clip.squeeze(0).cpu().numpy().transpose((1, 2, 0))
                gt_np = gt.squeeze(0).cpu().numpy().transpose((1, 2, 0))

                psnr_score = psnr_loss(pred_np, gt_np)
                ssim_score = ssim_loss(pred_np, gt_np, channel_axis=-1, data_range=1.0)

                psnr_list.append(psnr_score)
                ssim_list.append(ssim_score)
                lpips_list.append(lpips_score)

                with open(osp.join(result_dir, 'psnr_ssim_lpips.txt'), 'a') as f:
                    f.write(f"{name} ----> PSNR: {psnr_score:.4f}, SSIM: {ssim_score:.4f}, LPIPS: {lpips_score:.4f}\n")

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)

        print("\n" + "="*50)
        print(f"🏆 Final Results for NAFNet {opt['name']} ({len(test_loader)} images):")
        print(f"   => PSNR  (越大越好) : {avg_psnr:.4f}")
        print(f"   => SSIM  (越大越好) : {avg_ssim:.4f}")
        print(f"   => LPIPS (越小越好) : {avg_lpips:.4f}")
        print("="*50 + "\n")

        with open(osp.join(result_dir, 'psnr_ssim_lpips.txt'), 'a') as f:
            f.write(f"\n[FINAL AVERAGE] PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}\n")

if __name__ == '__main__':
    main()