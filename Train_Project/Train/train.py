from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from numpy import *
import warnings
import copy

from utils.dataset import DatasetInit
from utils.utils import (load_checkpoint, save_checkpoint, basic_transform, train_transform, get_loaders, train_fn,
                         val_fn, compute_validation_loss, create_result_dir, save_predictions_as_imgs, get_result)

warnings.filterwarnings("ignore")

NUM_FOLD = 5  # number of fold
NUM_EPOCHS = 100  # number of epochs
BATCH_SIZE = 4  # batch size
NUM_WORKERS = 0  # number of workers
PIN_MEMORY = True  # pin memory switch
DROP_LAST = True  # drop last switch
DEFAULT_LEARNING_RATE = 1e-4  # default learning rate
DYNAMIC_LR = True  # dynamic learning rate switch
IMG_HEIGHT = 224  # image height
IMG_WIDTH = 224  # image width
DATASET = "LUNG"  # name of dataset
IMG_DIR = "../Dataset/images"  # train image directory
MASK_DIR = "../Dataset/masks"  # train mask directory
SAVE_IMG = True  # save image
SAVE_MODEL = True  # save model switch
LOAD_MODEL = False  # load model switch
USE_AMP = True  # mixed accuracy training switch
DEVICE = "cuda:0"  # device
MODEL_NAME = "VET_FF_Net"  # name of model

# Import model
from models.VVET_FF_Net.VET_FF_Net_Model import VEET_FF_Net

# Define model
MODEL = VEET_FF_Net().to(DEVICE)


def main():
    kf = KFold(n_splits=NUM_FOLD, shuffle=False)
    ba_transform = basic_transform(height=IMG_HEIGHT, width=IMG_WIDTH)
    tr_transform = train_transform(height=IMG_HEIGHT, width=IMG_WIDTH)
    dataset = DatasetInit(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=ba_transform)

    result_list = []

    result_path = create_result_dir(MODEL_NAME, DATASET)

    for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
        print('\n--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_dataset = DatasetInit(image_dir=IMG_DIR, mask_dir=MASK_DIR, transform=tr_transform)

        train_subset = Subset(train_dataset, train_ids)
        val_subset = Subset(dataset, val_ids)

        train_loader = get_loaders(train_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                   drop_last=DROP_LAST)
        val_loader = get_loaders(val_subset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                                 drop_last=DROP_LAST)

        model = copy.deepcopy(MODEL)

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)

        if DYNAMIC_LR:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1,
                                                             verbose=True)
        else:
            scheduler = None

        if LOAD_MODEL:
            load_checkpoint(torch.load("checkpoint.pth.tar"), model)

        scaler = torch.cuda.amp.GradScaler()

        best_loss = float('inf')

        dice_list = []

        for epoch in range(NUM_EPOCHS):
            train_loss = train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler, epoch, amp=USE_AMP,
                                  scheduler=scheduler, dlr=DYNAMIC_LR, device=DEVICE)

            val_loss = compute_validation_loss(val_loader, model, loss_fn, DEVICE)

            if SAVE_MODEL and train_loss < best_loss:
                best_loss = train_loss
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, f"{MODEL_NAME}_{DATASET}_fold{fold}.pth.tar")

            pa, iou, dice, sensitivity, specificity, precision, auc, accuracy = val_fn(val_loader, model, device=DEVICE)
            epoch_data = [pa, iou, dice, sensitivity, specificity, precision, auc, accuracy, train_loss, val_loss]
            result_list.append(epoch_data)

            dice_list.append(dice)

        if SAVE_IMG:
            save_predictions_as_imgs(val_loader, model, folder=(result_path + "/saved_images{}").format(str(fold)),
                                     device=DEVICE)

        print(f"\n--------FOLD {fold} Finished--------")
        print(f"Mean Dice: {mean(dice_list)}")

        if '/' in str(IMG_DIR):
            data = IMG_DIR.split("/", 2)[1]
        print("\n{}".format(str(data)))
        if '(' in str(model):
            models = str(model).split("(", 1)[0]
        print("{}".format(str(models)))
        print("DEFAULT_LEARNING_RATE = ", DEFAULT_LEARNING_RATE)

    get_result(result_list, MODEL_NAME, DATASET, result_path)


if __name__ == "__main__":
    main()
