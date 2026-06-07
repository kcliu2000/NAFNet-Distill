import cv2
import os
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定圖片名稱與裁切位置 (請依實際情況修改)
# ==========================================
img_name = 'GOPR0410_11_00_000199.png'  # 換成你觀察到文字輪廓最明顯的那張圖

# 設定裁切框的位置 (單位: 像素)
# 建議: 可以先把原圖下載到電腦，用本機的圖片檢視器(如小畫家、Mac預覽程式)
# 游標指到你要的文字左上角和右下角，看一下 X 和 Y 座標
y_start, y_end = 350, 500  # 高度 (上下範圍)
x_start, x_end = 790, 1000  # 寬度 (左右範圍)

# ==========================================
# 2. 設定資料夾路徑 (已經幫你寫好伺服器路徑)
# ==========================================
path_gt = f'/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/test/sharp/{img_name}'
path_blur = f'/home/m11302124/MIMO-UNet-Wavelet/dataset/GOPRO_6imgs/test/blur/{img_name}'
path_expB = f'/home/m11302124/NAFNet/results/NAFNet_Test_ExpB_OutputKD_archived_20260601_203416/visualization/gopro-test/{img_name}'
path_expC = f'/home/m11302124/NAFNet/results/NAFNet_Test_ExpC_WaveletKD_archived_20260601_203432/visualization/gopro-test/{img_name}'

# 建立儲存結果的資料夾
save_dir = f'./patch_comparisons/{img_name.split(".")[0]}'
os.makedirs(save_dir, exist_ok=True)

# ==========================================
# 3. 讀取圖片與防呆檢查
# ==========================================
def load_and_crop(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到圖片: {path}")
    
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 🚨 印出原圖大小與你的裁切座標，幫你抓蟲！
    print(f"[{path.split('/')[-1]}] 原圖大小(Y, X, C): {img_rgb.shape} | 預計裁切 Y:{y_start}~{y_end}, X:{x_start}~{x_end}")
    
    patch = img_rgb[y_start:y_end, x_start:x_end]
    
    # 防呆檢查：如果切出來是空的，提早報錯並給予提示
    if patch.size == 0:
        raise ValueError("⚠️ 裁切失敗！切出來的小圖是空的。請檢查是不是 X 和 Y 填反了，或是 start 大於 end！")
        
    return patch

print(f"正在處理圖片: {img_name}")
patch_gt = load_and_crop(path_gt)
patch_blur = load_and_crop(path_blur)
patch_expB = load_and_crop(path_expB)
patch_expC = load_and_crop(path_expC)

# 將四張小截圖獨立存下來 (存回 BGR 格式給 cv2 寫入)
cv2.imwrite(f"{save_dir}/1_GT.png", cv2.cvtColor(patch_gt, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{save_dir}/2_Blur.png", cv2.cvtColor(patch_blur, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{save_dir}/3_ExpB.png", cv2.cvtColor(patch_expB, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{save_dir}/4_ExpC.png", cv2.cvtColor(patch_expC, cv2.COLOR_RGB2BGR))

# ==========================================
# 4. 畫出並排對比圖 (Matplotlib)
# ==========================================
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(f"Patch Comparison: {img_name}", fontsize=16, fontweight='bold')

titles = ['(a) Ground Truth', '(b) Blurred Input', '(c) ExpB (Output KD)', '(d) ExpC (Wavelet KD)']
patches = [patch_gt, patch_blur, patch_expB, patch_expC]

for ax, title, patch in zip(axes, titles, patches):
    ax.imshow(patch)
    ax.set_title(title, fontsize=14)
    ax.axis('off')  # 關閉座標軸刻度

plt.tight_layout()
combine_save_path = f"{save_dir}/combined_comparison.png"
plt.savefig(combine_save_path, dpi=300, bbox_inches='tight')
print(f"✅ 對比圖已成功儲存至: {combine_save_path}")
print(f"✅ 四張獨立的小截圖也存放在同一個資料夾中！")