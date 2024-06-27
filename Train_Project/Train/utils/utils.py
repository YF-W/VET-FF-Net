import os
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def basic_transform(height, width):
    data_transform = A.Compose([A.Resize(height=height, width=width),
                                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                                ToTensorV2()])

    return data_transform


def train_transform(height, width):
    data_transform = A.Compose([A.Resize(height=height, width=width),
                                A.VerticalFlip(always_apply=False, p=0.5),
                                A.HorizontalFlip(always_apply=False, p=0.5),
                                A.RandomRotate90(always_apply=False, p=0.5),
                                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
                                ToTensorV2()])

    return data_transform


def get_loaders(dataset, batch_size, num_workers, pin_memory, drop_last):
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader


def compute_validation_loss(data_loader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device=device)
            targets = targets.float().unsqueeze(1).to(device=device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item()

    return val_loss / len(data_loader)


def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler, epoch, amp, scheduler, dlr, device):
    print(f"---Epoch:{epoch}---")
    loop = tqdm(train_loader, total=len(train_loader), mininterval=0.1, miniters=1)
    train_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        # forward
        if amp:
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        train_loss += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    if dlr:
        # 在每个 epoch 结束后计算当前的验证集损失
        current_loss = compute_validation_loss(val_loader, model, loss_fn, device)

        # 调度器更新学习率
        scheduler.step(current_loss)

    return train_loss / len(train_loader)


def val_fn(val_loader, model, device):
    num_correct = 0
    num_pixels = 0
    pa = 0
    iou = 0
    dice = 0
    sensitivity = 0
    specificity = 0
    precision = 0
    auc = 0

    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            x = model(x)
            x = torch.tensor(x)
            preds = torch.sigmoid(x)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            tp = (preds * y).sum()
            tn = num_correct - (preds * y).sum()
            fp = (preds - preds * y).sum()
            fn = (y - preds * y).sum()
            pa += (tp + tn) / ((tp + tn + fp + fn) + 1e-8)
            iou += tp / ((tp + fp + fn) + 1e-8)
            dice += (2 * tp) / ((2 * tp + fp + fn) + 1e-8)
            sensitivity += tp / ((tp + fn) + 1e-8)
            specificity += tn / ((tn + fp) + 1e-8)
            precision += tp / ((tp + fp) + 1e-8)

            a = y.cpu().numpy()  # 标签tensor转为list
            b = preds.cpu().numpy()  # 预测tensor转为list
            aa = list(np.array(a).flatten())  # 高维转为1维度
            bb = list(np.array(b).flatten())  # 高维转为1维度
            auc = metrics.roc_auc_score(aa, bb, multi_class='ovo')

    pa = (pa / len(val_loader)).cpu().numpy()
    iou = (iou / len(val_loader)).cpu().numpy()
    dice = (dice / len(val_loader)).cpu().numpy()
    sensitivity = (sensitivity / len(val_loader)).cpu().numpy()
    specificity = (specificity / len(val_loader)).cpu().numpy()
    precision = (precision / len(val_loader)).cpu().numpy()
    accuracy = (num_correct / num_pixels).cpu().numpy()

    print(f"PA: {pa}")
    print(f"IoU: {iou}")
    print(f"Dice: {dice}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Precision: {precision}")
    print(f"AUC: {auc}")
    print(f"Accuracy: {accuracy}")

    model.train()

    return pa, iou, dice, sensitivity, specificity, precision, auc, accuracy


def create_result_dir(model_name, dataset):
    original_path = os.getcwd()

    primary_path = "../Record_Data"
    os.chdir(primary_path)

    folder_name = f"{model_name}_{dataset}"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    else:
        print("\033[31mRecordData文件夹内当前项目已存在，请仔细检查命名，若需要进行覆盖，请手动删除原项目。\033[0m")
        print(
            "\033[31mThe current project already exists in the RecordData folder, please double check the naming and manually delete the original project if you need to perform an overwrite.\033[0m")
        print("\033[30m有笨逼，但我不说是谁。\033[0m")

        quit()

    folder_path = f"{folder_name}"
    os.chdir(folder_path)

    for i in range(5):
        image_folder = f"saved_images{i}"
        if not os.path.exists(image_folder):
            os.mkdir(image_folder)

    os.chdir(original_path)

    result_path = f"../Record_Data/{folder_name}"

    return result_path


def save_predictions_as_imgs(val_loader, model, folder="saved_images/", device="cuda"):
    model.eval()

    for idx, (x, y) in enumerate(val_loader):
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


def get_result(result_list, model_name, dataset, path):
    column = ['PA', 'IoU', 'Dice', 'Sensitivity', 'Specificity', 'Precision', 'AUC', 'Accuracy', 'Train_loss',
              'Val_loss']
    log = pd.DataFrame(columns=column, data=result_list)
    log.to_csv(path + f"/{model_name}_{dataset}.csv")
