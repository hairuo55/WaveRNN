import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class f_Loss(torch.nn.Module):
    def __init__(self):
        super(f_Loss, self).__init__()
        self.hann_window_512 = torch.hann_window(441,requires_grad=True).cuda() # 20ms 110set5ms 512
        self.hann_window_128 = torch.hann_window(110,requires_grad=True).cuda() # 5ms 55step2.5ms 128
        self.hann_window_2048 = torch.hann_window(882,requires_grad=True).cuda() # 40ms 551setp10ms 2048

    def forward(self, model_output):
        audio_gen,audio_real = model_output
        #for window_choice in [512,128,2048]:
        nf_gen = torch.stft(audio_gen,512,hop_length=110,win_length=441,window=self.hann_window_512,onesided=False) #20ms
        nf_real = torch.stft(audio_real,512,hop_length=110,win_length=441,window=self.hann_window_512,onesided=False) #
        nf_gen_power = torch.pow(nf_gen,2)
        nf_real_power = torch.pow(nf_real,2)

        nf_gen_power_plus = nf_gen_power[:,:,:,0] + nf_gen_power[:,:,:,1]
        nf_real_power_plus = nf_real_power[:,:,:,0] + nf_real_power[:,:,:,1]
        loss = torch.sum(torch.pow(torch.log(nf_real_power_plus+1e-10)-torch.log(nf_gen_power_plus+1e-10),2)) / (audio_gen.size(0)*nf_gen.size(2)*nf_gen.size(1))

        nf_gen_128 = torch.stft(audio_gen,128,hop_length=55,win_length=110,window=self.hann_window_128,onesided=False)
        nf_real_128 = torch.stft(audio_real,128,hop_length=55,win_length=110,window=self.hann_window_128,onesided=False)
        nf_gen_power_128 = torch.pow(nf_gen_128,2)
        nf_real_power_128 = torch.pow(nf_real_128,2)
        nf_gen_power_plus_128 = nf_gen_power_128[:,:,:,0] + nf_gen_power_128[:,:,:,1]
        nf_real_power_plus_128 = nf_real_power_128[:,:,:,0] + nf_real_power_128[:,:,:,1]
        loss_128 = torch.sum(torch.pow(torch.log(nf_real_power_plus_128+1e-10) - torch.log(nf_gen_power_plus_128+1e-10),2)) / (audio_gen.size(0)*nf_gen_128.size(2)*nf_gen_128.size(1))

        nf_gen_2048 = torch.stft(audio_gen,1024,hop_length = 220,win_length=882,window=self.hann_window_2048,onesided=False)
        nf_real_2048 = torch.stft(audio_real,1024,hop_length = 220,win_length=882,window=self.hann_window_2048,onesided=False)
        nf_gen_power_2048 = torch.pow(nf_gen_2048,2)
        nf_real_power_2048 = torch.pow(nf_real_2048,2)
        nf_gen_power_plus_2048 = nf_gen_power_2048[:,:,:,0] + nf_gen_power_2048[:,:,:,1]
        nf_real_power_plus_2048 = nf_real_power_2048[:,:,:,0] + nf_real_power_2048[:,:,:,1]
        loss_2048 = torch.sum(torch.pow(torch.log(nf_real_power_plus_2048+1e-10) - torch.log(nf_gen_power_plus_2048+1e-10),2)) / (audio_gen.size(0)*nf_gen_2048.size(2)*nf_gen_2048.size(1))

        return  (loss + loss_128 +loss_2048)/3.0
