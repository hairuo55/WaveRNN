import sys
sys.path.append("/data/wangtao/wavernn/WaveRNN")
import generate
from generate import gen_from_mel , gen_with_x_and_mel
import torch
import numpy as np

mel_path = "/data/wangtao/wavernn/WaveRNN/data_vctk/tk_22050_delsil/mel_fmax_8000_22050/p276_320.npy"

wave_path = "/data/wangtao/wavernn/WaveRNN/data_vctk/tk_22050_delsil/quant_fmax_8000_22050/p276_320.npy"

path = "/data/wangtao/wavernn/WaveRNN/checkpoints/f_loss/checkpoint_400k_steps.pyt"
save_path = "1.wav"


x = torch.from_numpy(np.load(wave_path))



melspec = torch.from_numpy(np.load(mel_path))
melspec = torch.unsqueeze(melspec,0)

gen_from_mel(melspec,x,path,save_path)

#gen_with_x_and_mel(melspec, x , path, save_path)
