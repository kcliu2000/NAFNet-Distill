import cv2
import numpy as np

# 1. 讀取你上傳的三張圖片
# 請確保檔名與實際存放的路徑一致
img1 = cv2.imread('combined_comparison-1.png')
img2 = cv2.imread('combined_comparison-2.png')
img3 = cv2.imread('combined_comparison-3.png')

# 防呆檢查：確保圖片都有成功讀取
if img1 is None or img2 is None or img3 is None:
    raise ValueError("⚠️ 找不到圖片，請檢查檔名或路徑是否正確！")

# 2. 強制統一寬度 (以第一張圖為基準)
# 因為 matplotlib 存圖時，邊界可能會有一兩個像素的微小落差
# 我們把第二、第三張圖的寬度對齊第一張，高度按比例縮放，避免 vstack 報錯
h1, w1 = img1.shape[:2]

def resize_to_match_width(img, target_width):
    h, w = img.shape[:2]
    if w == target_width:
        return img
    # 計算新的高度以維持長寬比
    new_h = int(h * (target_width / w))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)

img2_resized = resize_to_match_width(img2, w1)
img3_resized = resize_to_match_width(img3, w1)

# 3. 垂直拼接 (Vertical Stack)
# 將三張圖由上到下接在一起
final_img = np.vstack((img1, img2_resized, img3_resized))

# 4. 存檔輸出
save_path = 'final_paper_figure.png'
cv2.imwrite(save_path, final_img)
print(f"✅ 成功拼貼！大圖已儲存為: {save_path}")