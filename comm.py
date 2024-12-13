import torch

from param import Hyper_Param

DEVICE = Hyper_Param['DEVICE']
class Channel:
    def __init__(self, _iscomplex):
        self._iscomplex = _iscomplex



    def ideal_channel(self, x):
        return x

    def awgn(self, x, seed,snr):
        torch.manual_seed(seed)
        x = x.clone()
        std = (10 ** (-snr / 10.) / 2) ** 0.5 if self._iscomplex else (10 ** (
                    -snr / 10.)) ** 0.5  # for complex xs.
        # noise = torch.randn_like(x).to(DEVICE) * std
        # y = x + noise
        # print(noise)
        y = x + torch.randn_like(x).to(DEVICE) * std
        return y

    def fading(self, x, seed, snr):
        torch.manual_seed(seed)
        if self._iscomplex:
            x_shape = x.shape
            _dim = x_shape[1] // 2
            _std = (10 ** (-snr / 10.) / 2) ** 0.5
            ch = (torch.randn(x_shape[0], 2) / (2 ** 0.5)).to(DEVICE)
            # _mul = torch.abs(ch).to(DEVICE)# should divide 2**0.5 here.

            x_ = x.clone()
            x_[:, :_dim] = x_[:, :_dim] * ch[:, 0].view(-1, 1) - x_[:, _dim:] * ch[:, 1].view(-1, 1)
            x_[:, _dim:] = x_[:, _dim:] * ch[:, 0].view(-1, 1) - x_[:, :_dim] * ch[:, 1].view(-1, 1)
            x = x_
        else:
            _std = (10 ** (-snr / 10.)) ** 0.5
            x = x * torch.abs(torch.randn(x.shape[0], 1)).to(DEVICE)
        y = x + torch.randn_like(x) * _std

        # print(y)
        # equal_y = y/ _mul.view(-1,1)
        # print(equal_y)
        return y


class Digitalize:
    def __init__(self, quant_max, qam_order, device='cuda'):
        self.num_bit = int(torch.log2(torch.tensor(qam_order)).item())  # num_bit을 정수로 변환
        self.qam_order = qam_order
        self.min_data = 0
        self.max_data = quant_max
        self.device = device

        # QAM 심볼 생성 및 고정된 정규화 인수 계산
        num_side = torch.sqrt(torch.tensor(self.qam_order)).item()
        # M = int (num_side ** 0.5)
        # print(M)
        # M = int(self.qam_order ** 0.5)
        qam_points = torch.arange(-(num_side - 1), num_side, 2).float()  # [-3, -1, 1, 3] for 16-QAM
        norm_factor = torch.sqrt(torch.mean(torch.abs(qam_points)**2 * 2))  # 평균 파워 정규화
        self.qam_points = (qam_points / norm_factor).to(self.device)  # (M,) 형태로 유지


    def Quantization(self, data):
        q_level = 2 ** self.num_bit
        data_clipped = torch.clamp(data, min=self.min_data, max=self.max_data)
        scaled_data = (data_clipped - self.min_data) / (self.max_data - self.min_data) * (q_level - 1)
        quantized_data = torch.round(scaled_data).int()
        return quantized_data

    def Modulation(self, data):
        M = int(self.qam_order ** 0.5)
        # I와 Q 성분 계산
        I = self.qam_points[(data // M) % M]  # (1, N)
        Q = self.qam_points[data % M]  # (1, N)

        # I와 Q를 (1, 2N) 형식으로 이어붙이기
        qam_symbols = torch.cat((I, Q), dim=1)  # (1, 2N)
        return qam_symbols

    def Demodulation(self, symbols):
        M = int(self.qam_order ** 0.5)

        # I와 Q 성분 분리 (1, N) 형태로 사용
        half_len = symbols.size(1) // 2
        I = symbols[:, :half_len]  # (batch_size, N)
        Q = symbols[:, half_len:]  # (batch_size, N)

        # I/Q를 가장 가까운 QAM 포인트로 매핑
        I_index = torch.argmin(torch.abs(I.unsqueeze(-1) - self.qam_points), dim=-1)  # (batch_size, N)
        Q_index = torch.argmin(torch.abs(Q.unsqueeze(-1) - self.qam_points), dim=-1)  # (batch_size, N)

        # 원래의 양자화된 값 복원
        data = I_index * M + Q_index
        return data

    def Dequantization(self, quantized_data):
        q_level = 2 ** self.num_bit
        data = (quantized_data.float() / (q_level - 1)) * (self.max_data - self.min_data) + self.min_data
        return data

    def Txapply(self, data):
        quantized_data = self.Quantization(data.to(self.device))
        modulated_symbols = self.Modulation(quantized_data)
        return modulated_symbols

    def Rxapply(self, symbols):
        demodulated_data = self.Demodulation(symbols.to(self.device))
        recovered_data = self.Dequantization(demodulated_data)
        return recovered_data





