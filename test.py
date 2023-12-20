import torch
from model import UNET
import time
import os 
import cv2
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision 
import numpy as np
import csv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, jaccard_score
csv_filename = 'metrics.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Image', 'Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC-ROC', 'IoU', 'Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


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
        folder_mask="BinarySegmentation/val_masks"
        path_mask=os.path.join((folder_mask), immagine.replace(".jpg", ".png"))
        name=immagine.replace(".jpg", ".png")
        target = cv2.imread(path_mask,0)
        target=cv2.resize(target, (240,160))
        target=np.asarray(target)
        # target = target.transpose((2, 0, 1))  
        # target = target / 255.0
        # target = torch.from_numpy(target).uint8().unsqueeze(0).to('cuda')
        with torch.no_grad():
            start=time.time()
            preds = torch.sigmoid(model(img))
            preds = (preds > 0.5).float()
            end=time.time()-start
            print(end)
            print(preds.size(), target.shape)
            preds = preds.squeeze().cpu().numpy() * 255
            preds=preds.astype(np.uint8)
            concatenated_img = np.concatenate((target, preds), axis=1)
            cv2.imwrite(f"test/{name}", concatenated_img)

            print(preds.max(), target.max())
            target_flat = target.flatten()
            preds_flat = preds.flatten()

            target_flat=(target_flat/255).astype(np.uint8)
            preds_flat=(preds_flat/255).astype(np.uint8)

            accuracy = accuracy_score(target_flat, preds_flat)
            recall = recall_score(target_flat, preds_flat)
            precision = precision_score(target_flat, preds_flat)
            f1 = f1_score(target_flat, preds_flat)
            roc_auc = roc_auc_score(target_flat, preds_flat)
            jaccard = jaccard_score(target_flat, preds_flat)
            with open(csv_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Scrive i risultati delle metriche nel file CSV
                writer.writerow({
                    'Image': immagine,
                    'Accuracy': accuracy,
                    'Recall': recall,
                    'Precision': precision,
                    'F1-Score': f1,
                    'AUC-ROC': roc_auc,
                    'IoU': jaccard,
                    'Time': end
                })

