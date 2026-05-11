import torch
from basicsr.models.archs.NAFNet_arch import NAFNet
from ptflops import get_model_complexity_info

teacher = NAFNet(width=64, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
student = NAFNet(width=64, enc_blk_nums=[1, 1, 2, 4], middle_blk_num=6, dec_blk_nums=[1, 1, 1, 1])

def print_complexity(model, name):
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f"[{name}]")
    print(f"  ➜ 參數數量 (Params): {params}")
    print(f"  ➜ 運算量 (MACs):    {macs}\n")

print("========================================")
print_complexity(teacher, "NAFNet 老師模型 [2,2,4,8] & mid 12")
print_complexity(student, "NAFNet 學生模型 [1,1,2,4] & mid 6")
print("========================================")