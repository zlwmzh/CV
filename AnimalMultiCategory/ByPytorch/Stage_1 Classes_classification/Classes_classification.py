#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2019/12/3 11:11
# @Author : Micky
# @Desc : 代码注释
# @File : Classes_classification.py
# @Software: PyCharm

from torch.utils.data import DataLoader
from Classes_Network import ResNet
import os
from PIL import Image
import numpy as np
import Config
from torchvision.transforms import transforms
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import copy


class MyDataset():
    def __init__(self, features_file, labels_file, transform=None):

        self.features_file = features_file
        self.labels_file = labels_file
        self.transform = transform

        if not os.path.isfile(self.features_file) or not os.path.isfile(self.labels_file):
            print(self.annotations_file + 'does not exist!')
        self.features_file_info = np.load(self.features_file)
        self.labels_file_info = np.load(self.labels_file)
        self.size = len(self.features_file_info)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.features_file_info[idx]
        if not os.path.isfile(image_path):
            print(image_path + '  does not exist!')
            return None

        image = Image.open(image_path).convert('RGB')
        label_class = int(self.labels_file_info[idx])

        sample = {'image': image, 'classes': label_class}
        if self.transform:
            sample['image'] = self.transform(image)
        return sample



TRAIN_FEATURES_PATH = 'Classes_train_features.npy'
TRAIN_LABELS_PATH = 'Classes_train_labels.npy'
VAL_FEATURES_PATH = 'Classes_val_features.npy'
VAL_LABELS_PATH = 'Classes_val_labels.npy'
train_transforms = transforms.Compose([transforms.Resize((Config.width, Config.height)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       ])
val_transforms = transforms.Compose([transforms.Resize((Config.width, Config.height)),
                                     transforms.ToTensor()
                                     ])

train_dataset = MyDataset(features_file=TRAIN_FEATURES_PATH,
                          labels_file=TRAIN_LABELS_PATH,
                          transform=train_transforms)

test_dataset = MyDataset(features_file=VAL_FEATURES_PATH,
                        labels_file=VAL_LABELS_PATH,
                         transform=val_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train_model(model, criterion, optimizer, num_epochs=50):
    Loss_list = {'train': [], 'val': []}
    Accuracy_list_classes = {'train': [], 'val': []}
    CHECKPOINT_DIR = './models'
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    checkpoint = './models/best_model.pt'

    # 模型恢复
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-*' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_classes = 0

            for idx, data in enumerate(data_loaders[phase]):
                # print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                labels_classes = data['classes'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)
                    # x_classes = x_classes.view(-1, 2)

                    _, preds_classes = torch.max(x_classes, 1)

                    loss = criterion(x_classes, labels_classes)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                corrects_classes += torch.sum(preds_classes == labels_classes)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('{} Loss: {:.4f}  Acc_classes: {:.2%}'.format(phase, epoch_loss, epoch_acc_classes))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))

            model.load_state_dict(best_model_wts)
            torch.save(model.state_dict(), checkpoint)

    print('Best val classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list, Accuracy_list_classes


network = ResNet(2, 0.5).to(device)
# optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.5)
optimizer = optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.99))
# optimizer = optim.Adagrad(network.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Decay LR by a factor of 0.1 every 1 epochs
model, Loss_list, Accuracy_list_classes = train_model(network, criterion, optimizer, num_epochs=100)
