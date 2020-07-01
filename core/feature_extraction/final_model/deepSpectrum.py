"""
语音分段->导出mel-spectrum.jpg->输入vgg or alexnet->输出特征矩阵->放入GRU训练
参考文献:Sentiment Analysis Using Image-based Deep Spectrum Features
"""
from global_values import *
import config
import os 
from common.log_handler import get_logger
from common.slide_windows import slideWindowsDeep
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from matplotlib.image import imread

logger = get_logger()

#example genImag('300_')
def genImg(file):
    if os.path.exists(config.data_dir+file+'P/img'):
        pass
    else:    
        os.mkdir(config.data_dir+file+'P/img')

    path = config.data_dir+file+'P/'+file+SUFFIX['wav']
    y, sr = librosa.load(path, sr=None)
    slideWindowsDeep(slide_size = 5,hop_size=4,fps=sr,data=y,file=file)
    logger.info(f"{file} mel-spectrum images have bees extracted!")

#example genDeepFea('300_)
def genDeepFea(model,file):        
    transform1 = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    images = os.listdir(config.data_dir+file+'P/img')
    images.sort(key=lambda img: eval(img[:img.find('-')]))#保证图片按时间顺序读取
    images = [imread(config.data_dir+file+'P/img/'+image) for image in images]

    image = images[0]
    image= transform1(image).unsqueeze(0)
    features=model.forword(image)

    for image in images[1:]:
        image= transform1(image).unsqueeze(0)
        result=model.forword(image)
        features = np.vstack([features,result])
    #logger.info(f"{file} deep features have bees extracted!")
    return features


class VggFc7(nn.Module):
    def __init__(self,vgg19):
        super(VggFc7,self).__init__()
        self.features = vgg19.features
        self.adaAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.fc = vgg19.classifier[:4]

    def forword(self,x):
        output = self.features(x)
        output = self.adaAvgPool2d(output)
        output = output.view(-1,512*7*7)
        output = self.fc(output)

        return output.data.numpy()

class alexnetFc7(nn.Module):
    def __init__(self,alexnet):
        super(alexnetFc7,self).__init__()
        self.features = alexnet.features
        self.adaAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(6,6))
        self.fc = alexnet.classifier[:5]

    def forword(self,x):
        output = self.features(x)
        output = self.adaAvgPool2d(output)
        output = output.view(-1,256*6*6)
        output = self.fc(output)

        return output.data.numpy()



