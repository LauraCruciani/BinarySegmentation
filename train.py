import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import time
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE =2
NUM_EPOCHS = 50
NUM_WORKERS = 1
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "BinarySegmentation/train_images/"
TRAIN_MASK_DIR = "BinarySegmentation/train_masks/"
VAL_IMG_DIR = "BinarySegmentation/val_images/"
VAL_MASK_DIR = "BinarySegmentation/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("best.pth"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()


    best = {'epoch':0,'dice_score':0.0} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter

    for epoch in range(NUM_EPOCHS):

        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, NUM_EPOCHS,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        dice_score=check_accuracy(val_loader, model, device=DEVICE)


        # Save checkpoint of latest and best model.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, 'latest_model.pth')
        trigger += 1
        if dice_score > best['dice_score']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, 'best_model.pth')
            best['epoch'] = epoch
            best['dice_score'] = dice_score
            trigger = 0
            save_predictions_as_imgs(epoch, val_loader, model, folder="saved_images/", device=DEVICE)
        print('Best performance at Epoch: {} | dice_score: {}'.format(best['epoch'],best['dice_score']))

        # save model
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer":optimizer.state_dict(),
        # }
        #save_checkpoint(checkpoint)

        # check accuracy
        

        # print some examples to a folder



if __name__ == "__main__":
    main()