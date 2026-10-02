import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.models as models
from   torch.utils.data import DataLoader
import random
import math
import numpy as np

from modelA import DualHeadResNet18
from dataloader import MyDataset
from utils import *
from torchvision import datasets, transforms

Kp = 1
Kn = 1
Kc = 1.0
Kd = 1.0
EPOCHS = 70
BATCH_SIZE = 16
FOLDER_PATH = 'dataset'
PATH_TRAINED = 'trained/modelB'
MODEL = "modelA"
LR = 0.5e-4


if __name__ == "__main__":
    print("GPU :", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualHeadResNet18(num_classes=1)
    init_dual_head_resnet(model)
    pretrained_weights = models.ResNet18_Weights.DEFAULT.get_state_dict(progress=True)
    model.load_state_dict(pretrained_weights, strict=False)
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    sequence_list = get_sequences(FOLDER_PATH)
    tv_sequence_list = [i for i in sequence_list if 'LIGHTING' in i] + [i for i in sequence_list if 'FOG/LOW' in i]

    train_list = random.sample(tv_sequence_list, int(0.8 * len(tv_sequence_list)))
    val_list = list(set(tv_sequence_list) - set(train_list))
    test_list = list(set(sequence_list) - set(tv_sequence_list))

    print(len(train_list), len(val_list), len(test_list))

    train_dataset = MyDataset(FOLDER_PATH, train_list)
    val_dataset = MyDataset(FOLDER_PATH, val_list)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):

        model.train()
        loss_cls_ep, loss_reg_ep, total_loss_ep = [], [], []

        for nb, dict_data in enumerate(train_loader):
            img_batch = dict_data["image_tensor"].to(device)
            data_annot = dict_data["annotation_tensor"].to(device)

            t_cls = torch.where(data_annot[:, 4] == 0, 1.0, 0.0).unsqueeze(1)
            t_dis = (torch.where(data_annot[:, 4] == 0, data_annot[:, 5], torch.zeros_like(data_annot[:, 5]))).unsqueeze(1)

            optimizer.zero_grad()
            pred_cls, pred_dis = model(img_batch)

            pred_cls = pred_cls.reshape(img_batch.shape[0], 1)
            pred_dis = pred_dis.reshape(img_batch.shape[0], 1)

            t_cls = t_cls.reshape(img_batch.shape[0],1)
            t_dis = t_dis.reshape(img_batch.shape[0],1)


            current_batch_size = img_batch.shape[0]
            for i in range(current_batch_size):
                print(f'{t_cls[i].item():.2f}', f'{pred_cls[i].item():.2f}')
                print(f'{t_dis[i].item():.2f}', f'{pred_dis[i].item():.2f}')

            eps = 1e-7
            pred_cls_clamped = torch.clamp(pred_cls, eps, 1.0 - eps)
            loss_cls = torch.mean(- Kp * t_cls * torch.log(pred_cls_clamped) - Kn * (1 - t_cls) * torch.log(1 - pred_cls_clamped))
            loss_cls = Kc * loss_cls

            positive_mask = (t_cls == 1.0)
            if positive_mask.sum() > 0:
                loss_reg = torch.mean((t_dis[positive_mask] - pred_dis[positive_mask]) ** 2)
            else:
                loss_reg = torch.tensor(0.0, device=device)
            
            loss_reg = Kd * loss_reg

            total_loss = loss_cls + loss_reg
            total_loss.backward()
            optimizer.step()

            print(f"Classification Loss: {loss_cls.item():.4f}")
            print(f"Distance Loss:       {loss_reg.item():.4f}")
            print(f"Total Combined Loss: {total_loss.item():.4f}")

            loss_cls_ep.append(loss_cls.item())
            loss_reg_ep.append(loss_reg.item())
            total_loss_ep.append(total_loss.item())

        print(np.mean(loss_cls_ep), np.mean(loss_reg_ep), np.mean(total_loss_ep))

        # --- VALIDATION PHASE ---

        model.eval()
        loss_cls_ep, loss_reg_ep, total_loss_ep = [], [], []
        with torch.no_grad():
            for nb, dict_data in enumerate(val_loader):
                img_batch = dict_data["image_tensor"].to(device)
                data_annot = dict_data["annotation_tensor"].to(device)

                t_cls = torch.where(data_annot[:, 4] == 0, 1.0, 0.0).unsqueeze(1)
                t_dis = (torch.where(data_annot[:, 4] == 0, data_annot[:, 5], torch.zeros_like(data_annot[:, 5]))).unsqueeze(1)

                pred_cls, pred_dis = model(img_batch)

                pred_cls = pred_cls.reshape(img_batch.shape[0], 1)
                pred_dis = pred_dis.reshape(img_batch.shape[0], 1)

                pred_cls_clamped = torch.clamp(pred_cls, eps, 1.0 - eps)

                t_cls = t_cls.reshape(img_batch.shape[0],1)
                t_dis = t_dis.reshape(img_batch.shape[0],1)


                loss_cls = torch.mean(- Kp * t_cls * torch.log(pred_cls_clamped) - Kn * (1 - t_cls) * torch.log(1 - pred_cls_clamped))
                loss_cls = Kc * loss_cls

                positive_mask = (t_cls == 1.0)
                if positive_mask.sum() > 0:
                    loss_reg = torch.mean((t_dis[positive_mask] - pred_dis[positive_mask]) ** 2)
                else:
                    loss_reg = torch.tensor(0.0, device=device)

                loss_reg = Kd * loss_reg
                total_loss = loss_cls + loss_reg

                loss_cls_ep.append(loss_cls.item())
                loss_reg_ep.append(loss_reg.item())
                total_loss_ep.append(total_loss.item())

        print(np.mean(loss_cls_ep), np.mean(loss_reg_ep), np.mean(total_loss_ep))

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict()
        }, PATH_TRAINED / f"{MODEL}_{epoch}.pth")


