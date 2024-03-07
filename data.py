import os
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore.dataset as ds
import cv2
from mindspore.dataset.vision import Inter as Inter
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from PIL import Image
import librosa
import scipy.io as scio
from mindspore import Tensor

def audio_extract(wav_file, sr=16000):
    wav=librosa.load(wav_file,sr=sr)[0]
    # Takes a waveform(length 160000,sampling rate 16000) and extracts filterbank features(size 400*64)
    spec=librosa.core.stft(wav,n_fft=4096,hop_length=200,win_length=1024,window="hann",center=True,pad_mode="constant")
    mel=librosa.feature.mfcc(S=np.abs(spec), sr=sr,n_mfcc=64)#melspectrogram(S=np.abs(spec),sr=sr,n_mels=64,fmax=8000)
    #print(mel.shape)
    logmel=librosa.core.power_to_db(mel[:,:300])#300
    if logmel.shape[1]!=300:
       logmel=np.column_stack([logmel,[[0]*(300-int(logmel.shape[1]))]*64])
    return logmel.T.astype("float32")

import os
import numpy as np
import scipy.io as scio
from PIL import Image
import numpy.random as random

class PrecompDataset:
    def __init__(self, data_split, opt, transform=None):
        self.data_path = opt['dataset']['data_path']
        self.dic_path = opt['dataset']['dic_path']
        self.transform = transform
        #         self.loc = './data/'
        #         image_dir="./data/rsicd_images/"
        #         audio_dir="./data/rsicd_audios/"
        #   dic=open("./data/rsicd.txt","r+").readlines()
        #   file_name = "./data/rsicd_precomp/" = self.data_path
        self.img_path = opt['dataset']['image_path']
        self.aud_path = opt['dataset']['audio_path']
        file_name = self.data_path + '%s.txt' % data_split
        dic = open(self.dic_path, "r+").readlines()
        dataset = open(file_name, "r+")
        datatype = "rsicd"
        d={}

        self.audios = []
        self.images = []
        self.labels = []

        for i in dic:
            if datatype=="rsicd":
               k,v,l=i.strip().split()
               if k in d:
                  continue
               d[k]=[v.strip(),l.strip()]
            else:
               k,v=i.strip().split()
               d[k]=[v.strip()]
        dataset=set(dataset)

        for iname in dataset:
            iname = iname.strip()
            if iname.endswith(".jpg") or iname.endswith(".tif"):
                try:
                    if datatype == "rsicd":
                        aname = d[iname][0]
                    else:
                        aname = iname.split(".")[0] + "S1.wav"

                    image_path = os.path.join(self.img_path, iname)
                    audio_path = os.path.join(self.aud_path, aname)

                    img = cv2.resize(cv2.imread(image_path).astype(np.float32), (224, 224))
                    aud = audio_extract(audio_path)
                    # aud = extract_feature(audio_path)
                    self.images.append(img)
                    self.audios.append(aud)
                    if datatype == "rsicd":
                        self.labels.append(int(d[iname][1]))
                    else:
                        self.labels.append(int(d[iname][0]))


                except Exception as e:
                    print(e)
        self.length = len(self.audios)
        print(len(self.images), len(self.audios), len(self.labels))


    def train_transform(self, image):

        # Resize
        image = image.resize((278, 278), Image.BILINEAR)
        # Random rotation
        angle = random.randint(-90, 90)
        image = image.rotate(angle)
        # Random crop
        crop_size = (224, 224)
        x = random.randint(0, image.width - crop_size[0])
        y = random.randint(0, image.height - crop_size[1])
        image = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
        # ToTensor
        image = np.array(image, dtype=np.float32) / np.float32(255.0)
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
           
        return image

    def test_transform(self, image):
        # Resize
        image = image.resize((224, 224), Image.BILINEAR)
        # ToTensor
        image = np.array(image, dtype=np.float32) / np.float32(255.0)
        # Normalize
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, index):
        img = self.images[index]
        img = img.transpose((2, 0, 1))
        if self.transform is not None:
            img = self.transform(img)
        aud = np.expand_dims(self.audios[index], 0)
        label = np.array(self.labels[index], dtype=np.uint)
        label = np.squeeze(EncodingOnehot(label, 30)) # 21,7,30

        return img / 255, aud, label


    def __len__(self):
        return self.length

def get_precomp_loader(data_split, batch_size=100, shuffle=True, opt={}):
    dset = PrecompDataset(data_split, opt)

    data_loader = ds.GeneratorDataset(dset, ["image", "audio", "label"], shuffle=shuffle, num_parallel_workers=opt['dataset']['workers'])
    data_loader = data_loader.batch(batch_size)
    return data_loader


def get_loaders(opt={}):
    train_loader = get_precomp_loader( 'train',
                                      opt['dataset']['batch_size'], True, opt)
    val_loader = get_precomp_loader( 'test',
                                    opt['dataset']['batch_size_val'], False, opt)
    return train_loader, val_loader


def get_test_loader(opt):
    test_loader = get_precomp_loader( 'test',
                                      opt['dataset']['batch_size_val'], False, opt)
    return test_loader

def EncodingOnehot(target,nclasses):

    return np.eye(nclasses)[target]