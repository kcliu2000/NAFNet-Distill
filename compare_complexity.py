import torch
from thop import profile
from thop import clever_format

# 引入你的 NAFNet 架構 (根據 BasicSR 的路徑)
# 如果路徑不同，請確認 basicsr/models/archs/ 裡面的 NAFNet 檔名與 Class 名稱
try:
    from basicsr.models.archs.NAFNet_arch import NAFNet
except ImportError:
    from basicsr.models.archs.local_arch import NAFNetLocal as NAFNet

def main():
    # ---------------------------------------------------------
    # 1. 定義 老師 (Teacher) 的架構
    # 對應 YAML: enc_blk_nums: [1, 1, 1, 28]
    # ---------------------------------------------------------
    teacher = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1]
    )

    # ---------------------------------------------------------
    # 2. 定義 學生 (Student) 的架構
    # 對應 YAML: enc_blk_nums: [1, 1, 1, 14]
    # ---------------------------------------------------------
    student = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 14],
        dec_blk_nums=[1, 1, 1, 1]
    )

    # ---------------------------------------------------------
    # 3. 建立測試用的假圖片輸入 (Dummy Input)
    # 解析度設定為 256x256 (這會影響 FLOPs 的大小)
    # ---------------------------------------------------------
    h, w = 256, 256
    dummy_input = torch.randn(1, 3, h, w)

    print(f"正在計算模型複雜度... 測試圖片大小: {h}x{w}\n")

    # ---------------------------------------------------------
    # 4. 執行計算 (macs = Multiply-Accumulate Operations)
    # ---------------------------------------------------------
    macs_t, params_t = profile(teacher, inputs=(dummy_input, ), verbose=False)
    macs_s, params_s = profile(student, inputs=(dummy_input, ), verbose=False)

    # 將數字轉為好讀的格式 (例如 1.2M, 15.3G)
    macs_t_str, params_t_str = clever_format([macs_t, params_t], "%.2f")
    macs_s_str, params_s_str = clever_format([macs_s, params_s], "%.2f")

    # ---------------------------------------------------------
    # 5. 印出漂亮的比對報表
    # ---------------------------------------------------------
    print("=" * 45)
    print(f"{'NAFNet 模型複雜度對比 (Teacher vs Student)':^40}")
    print("=" * 45)
    
    print(f"[Teacher] 滿血版 (28 blocks in encoder)")
    print(f"  ▶ 參數量 (Params) : {params_t_str}")
    print(f"  ▶ 運算量 (MACs)   : {macs_t_str}\n")
    
    print(f"[Student] 輕量版 (14 blocks in encoder)")
    print(f"  ▶ 參數量 (Params) : {params_s_str}")
    print(f"  ▶ 運算量 (MACs)   : {macs_s_str}\n")
    
    print("-" * 45)
    # 計算減少的比例
    param_reduction = 100 * (1 - params_s / params_t)
    macs_reduction = 100 * (1 - macs_s / macs_t)
    
    print(f"🏆 輕量化成果：")
    print(f"  ▶ 參數量減少了 : {param_reduction:.2f} %")
    print(f"  ▶ 運算量減少了 : {macs_reduction:.2f} %")
    print("=" * 45)

if __name__ == '__main__':
    main()