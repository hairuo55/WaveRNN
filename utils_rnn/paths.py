import os


class Paths :
    def __init__(self, data_path, model_id) :
        self.data = data_path
        self.quant = f'{data_path}/quant_fmax_8000_22050/'
        self.mel = f'{data_path}/mel_fmax_8000_22050/'
        self.checkpoints = f'checkpoints/{model_id}/'
        self.latest_weights = f'{self.checkpoints}latest_weights.pyt'
        #self.latest_weights = ""
        #self.latest_weights = f'/data/wangtao/wavernn/WaveRNN/checkpoints/8bit_mulaw_fmax_8000_22050/checkpoint_300k_steps.pyt'
        self.output = f'model_outputs/{model_id}/'
        self.step = f'{self.checkpoints}/step.npy'
        self.log = f'{self.checkpoints}log.txt'
        self.create_paths()

    def create_paths(self) :
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.output, exist_ok=True)
