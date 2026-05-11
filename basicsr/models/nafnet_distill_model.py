import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.models.archs import define_network

# 📌 武器：小波引導知識蒸餾 Loss (Wavelet KD Loss)
class HaarWaveletKDLoss(torch.nn.Module):
    def __init__(self, high_freq_weight=2.0):
        super(HaarWaveletKDLoss, self).__init__()
        self.hf_weight = high_freq_weight
        h0 = torch.tensor([[1/2, 1/2], [1/2, 1/2]]).view(1, 1, 2, 2)
        h1 = torch.tensor([[1/2, 1/2], [-1/2, -1/2]]).view(1, 1, 2, 2)
        h2 = torch.tensor([[1/2, -1/2], [1/2, -1/2]]).view(1, 1, 2, 2)
        h3 = torch.tensor([[1/2, -1/2], [-1/2, 1/2]]).view(1, 1, 2, 2)
        filters = torch.cat([h0, h1, h2, h3], dim=0).repeat(3, 1, 1, 1)
        self.register_buffer('filters', filters)

    def forward(self, student_img, teacher_img):
        stu_wav = F.conv2d(student_img, self.filters, stride=2, groups=3)
        tea_wav = F.conv2d(teacher_img, self.filters, stride=2, groups=3)
        stu_LL, stu_H = stu_wav[:, 0:3, :, :], stu_wav[:, 3:, :, :]
        tea_LL, tea_H = tea_wav[:, 0:3, :, :], tea_wav[:, 3:, :, :]
        loss_LL = F.l1_loss(stu_LL, tea_LL)
        loss_H = F.l1_loss(stu_H, tea_H)
        return loss_LL + self.hf_weight * loss_H

# 📌 NAFNet 通用型蒸餾框架
class NAFNetDistillationModel(ImageRestorationModel):
    def __init__(self, opt):
        super(NAFNetDistillationModel, self).__init__(opt)
        
        # 1. 建立並載入 Teacher 模型
        self.net_teacher = define_network(deepcopy(opt['network_teacher']))
        self.net_teacher = self.model_to_device(self.net_teacher)
        self.net_teacher.eval()
        
        load_path = self.opt['path'].get('pretrain_network_teacher', None)
        
        # 🛡️ 防禦機制：如果 BasicSR 試圖在 resume 時載入不存在的 teacher checkpoint
        if load_path and 'net_teacher_' in load_path:
            # 強制導回官方下載的滿血版權重
            load_path = 'experiments/pretrained_models/NAFNet-GoPro-width64.pth'
            print(f"⚠️ 偵測到 Resume 操作。強制將 Teacher 載入路徑設為: {load_path}")
            
        if load_path is not None:
            # 預設先找 'params'，找不到就用 None
            try:
                self.load_network(self.net_teacher, load_path, True, param_key='params')
            except KeyError:
                self.load_network(self.net_teacher, load_path, True, param_key=None)
            
        # 2. 讀取蒸餾設定
        self.kd_weight = self.opt['train'].get('kd_weight', 0.5)
        self.use_wavelet_kd = self.opt['train'].get('use_wavelet_kd', False)
        if self.use_wavelet_kd:
            self.wav_kd_weight = self.opt['train'].get('wavelet_kd_weight', 0.5)
            self.wavelet_kd_loss = HaarWaveletKDLoss().to(self.device)

    def optimize_parameters(self, current_iter, tb_logger=None):
        self.optimizer_g.zero_grad()

        # NAFNet 直接回傳單一 tensor，不像 FFTformer 是 list
        preds_student = self.net_g(self.lq)
        self.output = preds_student

        with torch.no_grad():
            preds_teacher = self.net_teacher(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # 基礎像素 Loss (通常是 PSNR/L1 Loss)
        if self.cri_pix:
            l_pix = self.cri_pix(preds_student, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
        # 傳統 Output KD Loss
        l_kd = F.l1_loss(preds_student, preds_teacher) * self.kd_weight
        l_total += l_kd
        loss_dict['l_kd'] = l_kd

        # 小波知識蒸餾 Loss
        if self.use_wavelet_kd:
            l_wav_kd = self.wavelet_kd_loss(preds_student, preds_teacher) * self.wav_kd_weight
            l_total += l_wav_kd
            loss_dict['l_wav_kd'] = l_wav_kd

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)