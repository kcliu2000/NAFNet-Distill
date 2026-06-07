import os
import shutil

# 🚨 這次要「追加」的 7 張測試圖片
image_names = [
    'GOPR0410_11_00_000199.png', 
    'GOPR0410_11_00_000212.png', 
    'GOPR0384_11_00_000011.png', 
    'GOPR0384_11_00_000006.png', 
    'GOPR0384_11_00_000004.png', 
    'GOPR0385_11_01_003095.png',
    'GOPR0385_11_01_003098.png'
]

# 原始大資料夾路徑
src_gt = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO/test/sharp/'
src_lq = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO/test/blur/'

# 你的迷你資料夾路徑 (維持不變)
dst_base = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/'
dst_gt = os.path.join(dst_base, 'test/sharp/')
dst_lq = os.path.join(dst_base, 'test/blur/')

# ⚠️ 這次拿掉了「清空資料夾」的程式碼，確保舊圖片被保留
os.makedirs(dst_gt, exist_ok=True)
os.makedirs(dst_lq, exist_ok=True)

print("開始將新圖片加入迷你測試集...")
for name in image_names:
    # 複製 Ground Truth (sharp)
    if os.path.exists(os.path.join(src_gt, name)):
        shutil.copy(os.path.join(src_gt, name), os.path.join(dst_gt, name))
    else:
        print(f"⚠️ 找不到 GT 圖片: {name}")
    
    # 複製 輸入模糊圖 (blur)
    if os.path.exists(os.path.join(src_lq, name)):
        shutil.copy(os.path.join(src_lq, name), os.path.join(dst_lq, name))
    else:
        print(f"⚠️ 找不到 Blur 圖片: {name}")
        
    print(f"✅ 已加入: {name}")

# 算一下現在總共有幾張圖
total_imgs = len(os.listdir(dst_gt))
print(f"\n🎉 準備完成！現在迷你資料夾內總共有 {total_imgs} 張圖片。")