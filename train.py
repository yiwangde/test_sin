# -*- coding: utf-8 -*-
from PIL import Image
import pandas as pd
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import dataset
from PIL import ImageFile
from dataset import MyDataset


def train():

    # path of data
    img_path = '/home/data/images/'
    train_csv = pd.read_csv('/home/data/labels/training.csv')
    test_csv = pd.read_csv('/home/data/labels/test.csv')

    train_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.29, 0.40, 0.42], std=[0.25, 0.23, 0.26]),
    ])
    test_aug = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.30, 0.41, 0.42], std=[0.25, 0.23, 0.25]),
    ])

    train_data = MyDataset(root=img_path, input_data=train_csv, aug=train_aug)
    train_data = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=8)
    test_data = MyDataset(root=img_path, input_data=test_csv, aug=test_aug)
    test_data = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=8)

    print('load data over.')

    model = models.resnet18(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    model.load_state_dict(torch.load('./train_models/mnasnet0.5_top1_67.823-3ffadce67e.pth'), map_location=device)
    model.fc = nn.Linear(in_features=512, out_features=20, bias=True)
    
    
    model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model = nn.DataParallel(model)

    print('device:', device)

    epoch = 200
    for epo in range(1, epoch + 1):
        logger = 'epoch:{}'.format(epo)
        start_time = time.time()

        train_loss = 0
        model.train()
        for _, [img, fileid, label] in enumerate(train_data):
            inputs, labels = img.to(device).float(), label.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
               
            optimizer.step()
            train_loss += float(loss.cpu().item())
        logger += ', train loss:{:.6f}'.format(train_loss)

        logger = 'Time:{}, '.format(int(time.time() - start_time)) + logger
        print(logger)

        if epo == epoch:
            ans_file = []
            ans_pred = []
            with torch.no_grad():
                model.eval()
                for _, [img, fileid] in enumerate(test_data):
                    inputs = img.to(device).float()
                    outputs = model(inputs)
                    ans_file.extend(fileid)
                    ans_pred.extend(outputs.max(1)[1].detach().cpu().numpy())

            ans = [[ans_file[i], ans_pred[i]] for i in range(len(ans_file))]
            ans = pd.DataFrame(ans, columns=['FileID', 'SpeciesID'])
            ans.to_csv('output_{}.csv'.format(epo), index=None)
            print('save over')
    print('Finish.')

if not os.path.exists("./models")
    os.mkdir("./models")

torch.save(model, "./models/" +  time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + ".pth")

if __name__ == "__main__":

    train()






