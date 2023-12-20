import torch
from model import UNET
import time
import os 
import cv2
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision 
'''Codice per fare inferenza'''

folder = 'BinarySegmentation/val_images'
model = UNET(in_channels=3, out_channels=1).to('cuda')
cudnn.benchmark = True
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['net'])
#load_checkpoint(torch.load("best_model.pth"), model)
print('\033[0;33mModel Loaded!\033[0m')
if not os.path.exists(folder):
    print(f"The folder'{folder}' doesn't exist.")
    
else: 
    files = os.listdir(folder) 
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for immagine in images:
        path_immagine = os.path.join((folder), immagine)
        img = cv2.imread(path_immagine)
        img=cv2.resize(img, (240,160))
        img=np.asarray(img)
        
        img = img.transpose((2, 0, 1))  
        img = img / 255.0

        img = torch.from_numpy(img).float().unsqueeze(0).to('cuda')
        
        with torch.no_grad():
            start=time.time()
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
            end=time.time()-start

            torchvision.utils.save_image(preds, f"infer/pred_{immagine}"
        )
        print(end)

