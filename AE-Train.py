from PyTorchNN import AutoEncoder, DenoisingAutoEncoder
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torchvision import transforms
from torchvision import utils as vutils
import json
import os

data_path = "./Data/imgs/"
workers = 4
lr = 0.001
NUM_GPU = torch.cuda.device_count()
device = torch.device("cuda:0" if (torch.cuda.is_available() and NUM_GPU > 0) else "cpu")
min_losses = [100.0]


def save_imgs(epoch_number, batch_num, batch, img_save_path):
    name = img_save_path + epoch_number + "_" + str(batch_num) + ".png"
    img_grid = vutils.make_grid(batch.to(device), padding=2, normalize=True).cpu()
    vutils.save_image(img_grid, name)
    return

def train(ae, loss, optimizer, dataloader, start_epoch, total_epochs, model_save_path, img_save_path):
    losses = []
    end_epoch = start_epoch + total_epochs 
    total_steps = len(dataloader)
    min_loss = min_losses[-1]
    for epoch in range(start_epoch, end_epoch):
        print("="*70)
        print("EPOCH: [%d/%d]" % (epoch, end_epoch-1))
        print("="*70)
        for i, data in enumerate((dataloader), 0):
            ae.zero_grad()
            x = data[0].to(device) 
            y_pred = ae(x)
            error = loss(y_pred,x)
            error.backward()
            optimizer.step()
            losses.append(error.item())
            print("\tSTEP: [%d/%d]\t\t===========>>>\t\tLoss: [%0.5f], MinLoss: [%0.5f]" % (i,total_steps,losses[-1],min_loss), end = "\r")
            
            if (min_loss > losses[-1]):
                min_loss = losses[-1]
                torch.save(ae, model_save_path + "_.pt")
                save_imgs("MinLoss", "", y_pred, img_save_path)
            if (i%100 == 0):
                torch.save(ae, model_save_path + "_.pt")
                save_imgs("_Epoch_" + str(epoch), i, y_pred, img_save_path)
                
        min_losses.append(min_loss)
        print("\tLoss:" , [round(i,5) for i in losses[-10:]])
        print("\tMinLoss:", round(min_losses[-1],5), end="\n\n")
        if(min_losses[-1] == min_losses[-2]):
            print("Loss became constant for previous two epochs")
            while(True):
                flag = input("continue?(y/n)")
                if(flag == "n"):
                    torch.save(ae, model_save_path + "_.pt")
                    exit()
                elif(flag == "y"):
                    break
                else:
                    print("Wrong Input!!")
                

    torch.save(ae, model_save_path + "_.pt")
    return ae, losses

if __name__ == "__main__":
    print("Using Device:", device)
    version = str(input("Version of Model Architecture to use?: "))
    while(True):
        retrain = input("Retrain?(y/n)")
        if (retrain == "y"):
            model_path = "./Model/ae_version" + str(version) + "_.pt"
            ae = torch.load(model_path).to(device)
            break
        elif(retrain == "n"):
            with open("./Model Architecture/version" + version + ".json") as f:
                data = json.load(f)
            arch = data["architecture"]
            BATCH_SIZE = data["batch_size"]
            IMAGE_SIZE = data["image_size"]
            NUM_CHANNELS = data["channels"]
            input_shape = (BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            # ae = AutoEncoder(arch, input_shape).to(device)
            # ae.getSummary()
            ae = DenoisingAutoEncoder(arch, input_shape, noise_factor=data["noise_factor"]).to(device)
            break
        else:
            print("Wrong Input!")
    
    
    ae.getSummary()
    loss = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr = lr)
    start_epoch = 0
    dataset = dset.ImageFolder(root = data_path,
                               transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                                               transforms.CenterCrop(IMAGE_SIZE),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                                               ]))
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = workers)
    img_save_path = "./Training_reconstruction/Version " + version + "/"
    if (not os.path.exists(img_save_path)):
        os.mkdir(img_save_path)
    model_save_path = "./Model/ae_version" + version
    epochsToTrain = int(input("\n\nEnter Epochs to Train: "))
    if(not epochsToTrain):
        exit()
    start_epoch = 0
    all_losses = []
    while(True):
        ae, losses = train(ae, loss, optimizer, dataloader, start_epoch, epochsToTrain, model_save_path, img_save_path)
        start_epoch = epochsToTrain
        all_losses.append(losses)

        if(input("Train More? (y/n):") == "y"):
            epochsToTrain = int(input("Enter Epochs to Train: "))
        else:
            break

    exit()
