import cv2
import os
from tqdm import tqdm
import shutil
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
import numpy as np
import pandas as pd
import time

def updateInitFile():
    _init_data["InitConstants"]["StartEpoch"] = startEpoch
    _init_data["Flags"]["initialCopyCompleted"] = flag_initialCopyCompleted
    _init_data["Flags"]["trainingCompleted"] = flag_trainingCompleted
    _init_data["Flags"]["filteringCompleted"] = flag_filteringCompleted
    _init_data["Flags"]["filterMoveCompleted"] = flag_filterMoveCompleted

    with open(paths_initFile, "w") as f:
        json.dump(_init_data, f, indent=4)
    print("Updated Init File")
    return

def getFilteredImgsPaths(from_path):
    paths = []
    for folderName in os.listdir(from_path):
        path = os.path.join(from_path, folderName)
        for imgPath in os.listdir(path):
            paths.append(os.path.join(path, imgPath))
    return paths

def copyAll(pathList, to_path):
    print("Copying Files for Training to:", to_path)
    if(len(pathList) > 0):
        for i, file_name in enumerate(tqdm(pathList)):
            try:
                new_path = shutil.copy(file_name, to_path)
                os.rename(new_path, to_path + str(i) + ".jpg")
            except:
                print("Error in moving: ", file_name)
                exit()
    return

def deleteFiles(folder_path):
    
    file_list = os.listdir(folder_path)
    if(len(file_list) > 0):
        print("\tDeleting Files from", folder_path, "...")
        for file_name in tqdm(file_list):
            file_path = os.path.join(folder_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    print(f"Skipped: {file_name} (Not a file)")
            except Exception as e:
                print(f"Error while deleting {file_name}: {e}")
    return

def save_imgs(img_save_path, epoch_id, batch_id, batch):
    # id format == "xxx_"
    name = img_save_path + epoch_id + batch_id + ".png"
    img_grid = vutils.make_grid(batch.to(device), padding=2, normalize=True).cpu()
    vutils.save_image(img_grid, name)
    return

def makeFolder(path):
    if (not os.path.exists(path)):
        os.mkdir(path)
    else:
        deleteFiles(folder_path=path)
    return

def trainAE(ae, loss, optimizer, dataloader, start_epoch, total_epochs, model_save_path, img_save_path):
    ae.train()
    losses = []
    end_epoch = start_epoch + total_epochs 
    total_steps = len(dataloader)
    min_loss = min_losses[-1]
    for epoch in range(start_epoch, end_epoch):
        epoch_id = "EPOCH_" + str(epoch) + "_"
        print("\t" + "="*70)
        print("\t" + "EPOCH: [%d/%d]" % (epoch, end_epoch-1))
        print("\t" +"="*70)
        for i, data in enumerate((dataloader), 0):
            ae.zero_grad()
            x = data[0].to(device) 
            y_pred = ae(x)
            error = loss(y_pred,x)
            error.backward()
            optimizer.step()
            losses.append(error.item())
            print("\t" +"\tSTEP: [%d/%d]\t\t===========>>>\t\tLoss: [%0.5f], MinLoss: [%0.5f]" % (i,total_steps,losses[-1],min_loss), end = "\r")
            
            if (min_loss > losses[-1]):
                min_loss = losses[-1]
            if (i%100 == 0):
                torch.save(ae, model_save_path)
                batch_id = str(i)
                save_imgs(img_save_path, epoch_id, batch_id, y_pred)
                
        min_losses.append(min_loss)
        print("\t" +"\tLoss:" , [round(i,5) for i in losses[-10:]])
        print("\t" +"\tMinLoss:", round(min_losses[-1],5), end="\n\n")
        if(min_losses[-1] == min_losses[-2]):
            print("\t" +"\t**Loss became constant for previous two epochs\n")
            break
         
    torch.save(ae, model_save_path)
    return ae 

def save_encodings(encodings, img_paths, save_path):
    print("\t" + "Saving Encodings...")
    df = pd.DataFrame(encodings)
    df["path"] = img_paths
    df.to_csv(save_path, index = False)
    return df

def getEncodings(model, dataloader):
    print("\t" + "Getting Encodings...")
    img_encoding = []
    for data in tqdm(dataloader):
        img_encoding.append((model(data[0].to(device))).cpu().detach().numpy())
    
    return np.concatenate((img_encoding), axis = 0)

def getPaths(dataloader):
    img_path = []
    for paths in dataloader.sampler.data_source.imgs:
        img_path.append(paths[0])
    return img_path

def copyFiltered(filtered_paths, dst_path):
    # !!! DO THIS
    exit("ERROR")
    print("\tMoving Filtered Images...")
    rand_names = random.sample(range(10000000, 20000000), len(filtered_paths))
    for i, path in enumerate(tqdm(filtered_paths)):
        path = path.replace("\\","/")
        try:
            new_path = os.path.join(folderPath, str(rand_names[i]) + ".jpg")
            new_path = shutil.copy(path, dst_path)
            os.rename(new_path, dst_path + "_new_" + str(i) + ".jpg")
            if(len((dst_path + str(i) + ".jpg").split(".")) > 2):
                print("WRONG NAME:",dst_path + str(i) + ".jpg")
                exit()
        except:
            print("\t\tAlready Exists or invalid path:", path.split("/")[-1] )
    return

def getFilteredIndices(ae, dataloader):
    print("\tGetting Filtered Frames...")
    ae.eval()
    model = ae.encoder_model.to(device)
    enc= getEncodings(model, dataloader)
    enc[enc<0.5] = 0
    enc[enc>=0.5] = 1
    unq, index= np.unique(enc, axis=0, return_index=True)
    print("\t>>>>Total Unique Videos:", unq.shape[0])
    del model
    return enc, index



def setup(version):
    global device, min_losses, _init_data, EPOCHS,  startEpoch, BATCH_SIZE, IMAGE_SIZE, NUM_CHANNELS, input_shape, lr, workers, ae, transformations
    global paths_initFilteredImgs, paths_saveReconstFolder, paths_saveFilterFolder, paths_saveEncodingFolder, paths_trainSaveFolder, paths_trainDataloader, paths_modelArch, paths_modelSave 
    global flag_initialCopyCompleted, flag_trainingCompleted, flag_filteringCompleted, flag_filterMoveCompleted
    global paths_initFile
    
    paths_initFile = "./Model/ExtraFiltering/init.json"
    print("Using Init File:", paths_initFile)
    if(not os.path.exists(paths_initFile)):
        print("Init File Not Found!")
        exit()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    min_losses = [100.0]

    with open(paths_initFile, "r") as f:
        _init_data = json.load(f)
        
        EPOCHS = _init_data["InitConstants"]["EpochsToTrain"]
        startEpoch = _init_data["InitConstants"]["StartEpoch"]
        
        flag_initialCopyCompleted = _init_data["Flags"]["initialCopyCompleted"] 
        flag_trainingCompleted = _init_data["Flags"]["trainingCompleted"]
        flag_filteringCompleted = _init_data["Flags"]["filteringCompleted"]
        flag_filterMoveCompleted = _init_data["Flags"]["filterMoveCompleted"]

        paths_initFilteredImgs = getFilteredImgsPaths(_init_data["Paths"]["FilteredImgFolders"]["path"])
        paths_saveReconstFolder = _init_data["Paths"]["Save"]["reconstFolder"] + "Version " + version + "/"
        paths_saveFilterFolder = _init_data["Paths"]["Save"]["filterFolder"] + "Version " + version + "/"
        paths_saveEncodingFolder = _init_data["Paths"]["Save"]["encodingFolder"]
        paths_trainSaveFolder = _init_data["Paths"]["Train"]["saveFolder"]
        paths_trainDataloader = _init_data["Paths"]["Train"]["dataloader"]
        paths_modelArch = _init_data["Paths"]["Model"]["modelFolder"] + "Version " + version + "/" + _init_data["Paths"]["Model"]["arch"]
        
        with open(paths_modelArch, "r") as f:
            _init_arch = json.load(f)
            BATCH_SIZE, IMAGE_SIZE, NUM_CHANNELS  = _init_arch["init_constants"]["batch_size"], _init_arch["init_constants"]["image_size"], _init_arch["init_constants"]["channels"]
            input_shape = (BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            lr = _init_arch["init_constants"]["lr"]
            workers = _init_arch["init_constants"]["workers"]
            
            paths_modelSave = _init_data["Paths"]["Model"]["modelFolder"] + "Version " + version + "/" + _init_data["Paths"]["Model"]["model"]
            if (os.path.exists(paths_modelSave)):
                print("Retraining:", paths_modelSave.split("/")[-1])
                ae = torch.load(paths_modelSave).to(device)
            else:
                print("No model found.")
                print("Using Architecture: ", paths_modelArch.split("/")[-1])
                ae = DenoisingAutoEncoder(arch=_init_arch["architecture"], input_shape=input_shape, noise_factor=_init_arch["init_constants"]["noise_factor"]).to(device)
                startEpoch = 0
                flag_initialCopyCompleted = False
                flag_trainingCompleted = False
                flag_filteringCompleted = False
                flag_filterMoveCompleted = False
                updateInitFile()

        del _init_arch

    transformations = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    
    ae.getSummary(decoder=False)
    
    return


if __name__ == "__main__":
    
    version = input("Version to use: ")
    setup(version)
    
    # if(not flag_initialCopyCompleted):
    #     ef.makeFolder(path=paths_trainSaveFolder)
    #     copyAll(pathList=paths_initFilteredImgs, to_path=paths_trainSaveFolder)
    #     flag_initialCopyCompleted = True
    #     updateInitFile()
    
    if(not flag_trainingCompleted or not flag_filteringCompleted):
        # print(, "\t\t\t@time:", str(round((time.time() - start_time)/60, 2)) + "mins", "(" + str(round((time.time() - init_time)/60, 2)) + ")")
        dataset = dset.ImageFolder(root = paths_trainDataloader, transform = transformations)
        dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = workers, pin_memory=True, timeout=120)
        if(not flag_trainingCompleted):
            loss = nn.MSELoss()
            optimizer = optim.Adam(ae.parameters(), lr = lr)
            print("\n\n", "#"*80, sep="")
            print("\t\t\tTraining DAE")
            print("#"*80, "\n")
            makeFolder(path=paths_saveReconstFolder)
            ae = trainAE(ae, loss, optimizer, dataloader, startEpoch, EPOCHS, paths_modelSave, paths_saveReconstFolder)
            startEpoch += EPOCHS
            flag_trainingCompleted = True
            updateInitFile()

        if(not flag_filteringCompleted):
            enc, idx_list = getFilteredIndices(ae=ae, dataloader=dataloader)
            img_paths = getPaths(dataloader=dataloader)
            makeFolder(paths_saveEncodingFolder)
            save_encodings(enc, img_paths, save_path = paths_saveEncodingFolder + "final " + version + ".csv")
            filtered_paths = [img_paths[i] for i in idx_list]
            makeFolder(path=paths_saveFilterFolder)
            copyFiltered(filtered_paths=filtered_paths, dst_path=paths_saveFilterFolder)
            flag_filterMoveCompleted = True
            flag_filteringCompleted = True
            updateInitFile()
        
    if(not flag_filterMoveCompleted):
        df = pd.read_csv(paths_saveEncodingFolder + "final.csv")
        img_paths = df["path"]
        enc = df.drop(columns=["path"], inplace=False).to_numpy()
        unq, idx_list= np.unique(enc, axis=0, return_index=True)
        print("\t>>>>Total Unique Frames:", unq.shape[0])
        filtered_paths = [img_paths[i] for i in idx_list]
        # makeFolder(path=paths_saveFilterFolder)
        copyFiltered(filtered_paths=filtered_paths, dst_path=paths_saveFilterFolder)
        
        flag_filterMoveCompleted = True
        updateInitFile()

    # flag_trainingCompleted = False
    # flag_filteringCompleted = False
    # updateInitFile()
    del ae
    exit()