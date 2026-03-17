import os
import sys
import glob
import time
import cv2
import clip
import numpy as np
import torch
from tqdm import tqdm
from torch import einsum
from Nets.Network import Network
from Utilities import Consistency
import Utilities.DataLoaderFM as DLr
from torch.utils.data import DataLoader
from Utilities.CUDA_Check import GPUorCPU
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
DEVICE = GPUorCPU.DEVICE

class ZeroOneNormalize(object):
    def __call__(self, img):
        return img.float().div(255)

class Fusion:
    def __init__(self,
                 modelpath='RunTimeData/2025-04-21 10.08.42/model4.ckpt',
                 dataroot='D:\\ouyangbaicai\\MSI-DTrans\\Datasets\\Eval',
                 dataset_name='HBU-CVMDSP',
                 threshold=0.005,
                 window_size=5,
                 ):
        self.DEVICE = GPUorCPU().DEVICE
        self.MODELPATH = modelpath
        self.DATAROOT = dataroot
        self.DATASET_NAME = dataset_name
        self.THRESHOLD = threshold
        self.window_size = window_size
        self.window = torch.ones([1, 1, self.window_size, self.window_size], dtype=torch.float).to(self.DEVICE)

    def __call__(self, *args, **kwargs):
        if self.DATASET_NAME != None:
            self.SAVEPATH = '/' + self.DATASET_NAME
            self.DATAPATH = self.DATAROOT + '/' + self.DATASET_NAME
            MODEL = self.LoadWeights(self.MODELPATH)
            EVAL_LIST_A, EVAL_LIST_B = self.PrepareData(self.DATAPATH)
            self.FusionProcess(MODEL, EVAL_LIST_A, EVAL_LIST_B, self.SAVEPATH, self.THRESHOLD)
        else:
            print("Test Dataset required!")
            pass

    def LoadWeights(self, modelpath):
        clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        resnet_18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet_50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_101 = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        model = Network(clip_model, resnet_18, resnet_50, resnet_101).to(DEVICE)
        for module in [model.clip_model, model.resnet_model_18, model.resnet_model_50, model.resnet_model_101]:
            for param in module.parameters():
                param.requires_grad = False
        model.load_state_dict(torch.load(modelpath))
        model.eval()
        return model

    def PrepareData(self, datapath):
        eval_list_A = sorted(glob.glob(os.path.join(datapath, 'sourceA', '*.*')))
        eval_list_B = sorted(glob.glob(os.path.join(datapath, 'sourceB', '*.*')))
        return eval_list_A, eval_list_B

    def ConsisVerif(self, img_tensor, threshold):
        # Verified_img_tensor = Consistency.Binarization(img_tensor)
        # if threshold != 0:
        Verified_img_tensor = Consistency.RemoveSmallArea(img_tensor=img_tensor, threshold=threshold)
        return Verified_img_tensor

    def FusionProcess(self, model, eval_list_A, eval_list_B, savepath, threshold):
        if not os.path.exists('./Results/all_MFFW/' + 'MFFW36'):
            os.makedirs('./Results/all_MFFW/' + "MFFW36", exist_ok=True)
        eval_data = DLr.Dataloader_Eval(eval_list_A, eval_list_B)
        eval_loader = DataLoader(dataset=eval_data,
                                 batch_size=1,
                                 shuffle=False, )
        eval_loader_tqdm = tqdm(eval_loader, colour='blue', leave=True, file=sys.stdout)
        cnt = 1
        running_time = []
        with torch.no_grad():
            for A, B in eval_loader_tqdm:
                start_time = time.time()
                # Forward
                Pre_A, real_similarities_A, fake_similarities_A, text_A = model(A, None, None, forward=True)
                Pre_B, real_similarities_B, fake_similarities_B, text_B = model(B, None, None, forward=True)
                # Reverse
                Pre = model(Pre_A, Pre_B, text_A, forward=False)
                Fused = einsum('c w h -> w h c', Pre[0]).clone().detach().cpu().numpy()
                Fused = np.clip(Fused * 255, 0, 255).astype(np.uint8)
                Fused = cv2.cvtColor(Fused, cv2.COLOR_BGR2RGB)
                cv2.imwrite('./Results/all_MFFW/' + "MFFW36" + '/' + self.DATASET_NAME + '-' + str(cnt).zfill(2) + '-fused.jpg', Fused)
                cnt += 1
                running_time.append(time.time() - start_time)
        running_time_total = 0
        for i in range(len(running_time)):
            print("process_time: {} s".format(running_time[i]))
            if i != 0:
                running_time_total += running_time[i]
        print("\navg_process_time: {} s".format(running_time_total / (len(running_time) - 1)))
        print("\nResults are saved in: " + "./Results" + savepath)


if __name__ == '__main__':
    f = Fusion()
    f()