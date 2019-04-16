import glob
from utils_rnn.display import *
from utils_rnn.dsp import *
import hparams as hp
from multiprocessing import Pool, cpu_count
from utils_rnn.paths import Paths
import pickle
import argparse
import numpy as np
import pdb
import torch
import generate
# We're using the audio processing from TacoTron2 to make sure it matches
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT

MAX_WAV_VALUE = 32768.0 #2^15
from scipy.io.wavfile import read

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN')
parser.add_argument('--path', '-p', default=hp.wav_path, help='directly point to dataset path (overrides hparams.wav_path')
parser.add_argument('--extension', '-e', default='.wav', help='file extension to search for in dataset folder')
args = parser.parse_args()

extension = args.extension
path = args.path
import ulaw

def get_files(path, extension='.wav') :
    filenames = []
    for filename in glob.iglob(f'{path}/**/*{extension}', recursive=True):
        filenames += [filename]
    return filenames


stft = TacotronSTFT(filter_length=hp.filter_length,
                         hop_length=hp.hop_length,
                         win_length=hp.win_length,
                         sampling_rate=hp.sampling_rate,
                         mel_fmin=hp.mel_fmin, mel_fmax=hp.mel_fmax)

def get_mel(audio,stft):
    audio_norm = audio / MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    melspec = stft.mel_spectrogram(audio_norm)
    melspec = torch.squeeze(melspec, 0)
    return melspec


def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate

def convert_file(path) :
    y1,sr = load_wav_to_torch(path)
    mel = get_mel(y1,stft).numpy()
    quant_u = ulaw.lin2ulaw(y1.numpy())
    return mel.astype(np.float32), quant_u.astype(np.int16)


def process_wav(path) :
    id = path.split('/')[-1][:-4]
    m, x = convert_file(path)
    #np.save(f'{paths.mel}{id}.npy', m)
    #np.save(f'{paths.quant}{id}.npy', x)
    return id
pdb.set_trace()
wavs = "/data/wangtao/data/VCTK/VCTK-Corpus/wav48/p238/p238_262.wav"
mel,quant = convert_file(wavs)
save_path = "mel_data_1.wav"
restore_path = "/data/wangtao/wavernn/WaveRNN/checkpoints/8bit_mulaw/checkpoint_100k_steps.pyt"
#mel = np.load("mel_data.npy")
mel_torch = torch.from_numpy(mel)
mel_unsqueeze = torch.unsqueeze(mel_torch,0)
data = generate.gen_from_mel(mel_unsqueeze,restore_path,save_path)
print(wavs.split('/')[-1][:-4])


if False:
    wav_files = get_files(path, extension)
    paths = Paths(hp.data_path, hp.model_id)

    print(f'\n{len(wav_files)} {extension[1:]} files found in "{path}"\n')

    if len(wav_files) == 0 :

        print('Please point wav_path in hparams.py to your dataset,')
        print('or use the --path option.\n')

    else :

        simple_table([('Sample Rate', hp.sample_rate),
                      ('Bit Depth', hp.bits),
                      ('Mu Law', hp.mu_law),
                      ('Hop Length', hp.hop_length),
                      ('CPU Count', cpu_count())])

        pool = Pool(processes=cpu_count())
        dataset_ids = []

        for i, id in enumerate(pool.imap_unordered(process_wav, wav_files), 1):
            dataset_ids += [id]
            bar = progbar(i, len(wav_files))
            message = f'{bar} {i}/{len(wav_files)} '
            stream(message)

        with open(f'{paths.data}dataset_ids.pkl', 'wb') as f:
            pickle.dump(dataset_ids, f)

        print('\n\nCompleted. Ready to run "python train.py". \n')
