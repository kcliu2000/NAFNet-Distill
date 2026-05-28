import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_1d_power_spectrum(image_path):
    """讀取圖片並計算 1D 徑向平均功率頻譜"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ 找不到圖片: {image_path}")
        return None
    
    # 2D FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    power_spectrum = np.abs(fshift)**2
    
    # 徑向平均 (Radial Average)
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr
    
    return radial_profile

def main():
    # 這是你指定的 6 張測試圖片 (假設附檔名為 .png)
    image_names = [
        'GOPR0854_11_00_000078.png', 
        'GOPR0869_11_00_000083.png', 
        'GOPR0871_11_00_000054.png', 
        'GOPR0881_11_01_000203.png', 
        'GOPR0868_11_00_000096.png', 
        'GOPR0862_11_00_000059.png'
    ]
    
    # 👇 請確認這裡的資料夾路徑是正確的！
    # 👇 把這三行改成這樣：
    gt_dir = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/test/sharp/'
    expB_dir = './results/NAFNet_Test_ExpB_OutputKD/visualization/gopro-test/'
    expC_dir = './results/NAFNet_Test_ExpC_WaveletKD/visualization/gopro-test/'
    
    ps_gt_list, ps_expB_list, ps_expC_list = [], [], []
    
    print(f"開始處理 {len(image_names)} 張圖片的頻譜分析...")
    
    for name in image_names:
        print(f"  - 處理中: {name}")
        gt_ps = get_1d_power_spectrum(os.path.join(gt_dir, name))
        b_ps = get_1d_power_spectrum(os.path.join(expB_dir, name))
        c_ps = get_1d_power_spectrum(os.path.join(expC_dir, name))
        
        if gt_ps is not None and b_ps is not None and c_ps is not None:
            ps_gt_list.append(gt_ps)
            ps_expB_list.append(b_ps)
            ps_expC_list.append(c_ps)
            
    if not ps_gt_list:
        print("❌ 錯誤：沒有成功讀取到任何圖片，請檢查資料夾路徑！")
        return

    # 將所有圖片的頻譜截斷到相同長度並計算平均值
    min_len = min([len(ps) for ps in ps_gt_list])
    
    avg_ps_gt = np.mean([ps[:min_len] for ps in ps_gt_list], axis=0)
    avg_ps_expB = np.mean([ps[:min_len] for ps in ps_expB_list], axis=0)
    avg_ps_expC = np.mean([ps[:min_len] for ps in ps_expC_list], axis=0)
    
    frequencies = np.arange(min_len)
    
    # 開始畫圖
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(frequencies, np.log10(avg_ps_gt), label='Ground Truth', color='black', linestyle='--', linewidth=2)
    plt.plot(frequencies, np.log10(avg_ps_expB), label='ExpB (Output KD)', color='royalblue', linewidth=1.5)
    plt.plot(frequencies, np.log10(avg_ps_expC), label='ExpC (Wavelet KD)', color='crimson', linewidth=1.5)
    
    plt.title('Average 1D Power Spectrum (6 GoPro Test Images)', fontsize=16, fontweight='bold')
    plt.xlabel('Spatial Frequency (Low \u2192 High)', fontsize=14)
    plt.ylabel('Log Power', fontsize=14)
    plt.legend(fontsize=12)
    
    # 標示高頻區域 (最後 40% 的頻帶)
    plt.axvspan(min_len * 0.6, min_len, color='gray', alpha=0.1, label='High-Frequency Region')
    
    plt.tight_layout()
    output_fig = 'batch_spectrum_comparison.png'
    plt.savefig(output_fig, dpi=300)
    print(f"\n✅ 批量平均頻譜比較圖已生成並儲存為: {output_fig}")

if __name__ == '__main__':
    main()