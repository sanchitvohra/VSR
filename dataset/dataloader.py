import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import torch

class VSRDataset(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path

    def __len__(self):
        return len(os.listdir(self.video_path))

    def __getitem__(self, index):
        path = os.listdir(self.video_path)[index]

        cap = cv2.VideoCapture(os.path.join(self.video_path, path))

        frms = []

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret == True:
                frms.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else: 
                break

            if len(frms) == 30:
                break
            
        cap.release()

        frms = np.stack(frms)
        tsr = torch.FloatTensor(np.ascontiguousarray(frms)) / 255.0
        tsr = tsr.permute(0, 3, 1, 2)
        return {'gt': tsr}