import torch
import torchvision
import torch.nn as nn
from dataset import OurDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import auc
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_transform(image_hight=224,image_width=224):
    transform = A.Compose(
        [
            A.Resize(height=image_hight, width=image_width),
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
    return transform


def get_test_loader(
        test_dir,
        test_maskdir,
        batch_size,
        val_transform,
        num_workers=4,
        pin_memory=False,
        drop_list1=True
):

    test_ds = OurDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        # drop_list = drop_list1
    )

    return test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    TP=0
    FP=0
    FN=0
    TN=0
    tp= 0
    fp = 0
    fn = 0
    tn = 0
    auc = 0
    f_score = 0
    specificity = 0
    dice_score = 0
    jaccard_score = 0
    recall = 0
    precision = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            x = model(x)
            x = torch.tensor(x)
            preds = torch.sigmoid(x)
            preds = (preds > 0.5).float()
            #auc = roc_auc_score(y[:, 0], preds[:, 0])
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            TP = (preds * y).sum()
            TN = num_correct - (preds * y).sum()
            FP = (preds - preds * y).sum()
            FN = (y - preds * y).sum()
            tp += (preds * y).sum()
            tn += num_correct - tp
            fp += (preds - preds * y).sum()
            fn += (y - preds * y).sum()
            # auc = metrics.auc(fpr, tpr)

            a=y.cpu().numpy()
            b=preds.cpu().numpy()
            aa = list(np.array(a).flatten())
            bb = list(np.array(b).flatten())
            auc = metrics.roc_auc_score(aa, bb,multi_class='ovo')

            #auc=0
            precision += TP / ((TP + FP) + 1e-8)
            recall += TP / ((TP + FN) + 1e-8)
            f_score += (2 * TP) / ((FP + 2 * TP + FN) + 1e-8)
            specificity += TN / ((TN + FP) + 1e-8)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            jaccard_score += (preds * y).sum() / ((preds + y).sum() + 1e-8 - (preds * y).sum())

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.4f}")
    print(f"Dice score: {dice_score / len(loader)}")
    print(f"Jaccard score: {jaccard_score / len(loader)}")
    print(f"Precision:{precision / len(loader)}")
    print(f"Recall:{recall / len(loader)}")
    print(f"F1-score:{f_score / len(loader)}")
    print(f"Specificity:{specificity / len(loader)}")
    print(f"AUC:{auc}")
    print(f"TP total :{tp}")
    print(f"TN total:{tn}")
    print(f"FP total:{fp}")
    print(f"FN total:{fn}")


    # print(type(num_correct/num_pixels*100))
    model.train()

    return (num_correct / num_pixels * 100).cpu().numpy(), (dice_score / len(loader)).cpu().numpy(), (
                jaccard_score / len(loader)).cpu().numpy(), (precision / len(loader)).cpu().numpy(), (
                       recall / len(loader)).cpu().numpy(), (f_score / len(loader)).cpu().numpy(), (
                       specificity / len(loader)).cpu().numpy(), auc, tp.cpu().numpy(),tn.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy()




def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/lable_{idx}.png")
        torchvision.utils.save_image(x, f"{folder}/image_{idx}.png")


    model.train()