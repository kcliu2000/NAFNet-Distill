import os
import shutil

# 你挑選的 6 張菁英圖片
image_names = [
    'GOPR0854_11_00_000078.png', 
    'GOPR0869_11_00_000083.png', 
    'GOPR0871_11_00_000054.png', 
    'GOPR0881_11_01_000203.png', 
    'GOPR0868_11_00_000096.png', 
    'GOPR0862_11_00_000059.png'
]

# 原始大資料夾路徑
src_gt = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO/test/sharp/'
src_lq = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO/test/blur/'

# 新的迷你資料夾路徑
dst_gt = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/test/sharp/'
dst_lq = '/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/test/blur/'

# 建立新資料夾
os.makedirs(dst_gt, exist_ok=True)
os.makedirs(dst_lq, exist_ok=True)

print("開始複製圖片...")
for name in image_names:
    # 複製 Ground Truth (sharp)
    if os.path.exists(os.path.join(src_gt, name)):
        shutil.copy(os.path.join(src_gt, name), os.path.join(dst_gt, name))
    
    # 複製 輸入模糊圖 (blur)
    if os.path.exists(os.path.join(src_lq, name)):
        shutil.copy(os.path.join(src_lq, name), os.path.join(dst_lq, name))
        
    print(f"✅ 已複製: {name}")

print(f"\n🎉 迷你資料集建立完成！存放在: /home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/")