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

def getSortedVidPaths(_from_path):
    vid_paths = []
    for vid_name in os.listdir(_from_path):
        vid_paths.append(os.path.join(_from_path, vid_name))
    new_vid_paths = []
    for path in vid_paths:
        id = int(path.split("/")[-1].split(".")[0])
        new_vid_paths.append((id, path))
    new_vid_paths.sort(key= lambda x: x[0])
    new_vid_paths = [x[1] for x in new_vid_paths]
    return new_vid_paths


def setup():
    global device, min_losses, _init_data, EPOCHS,  startFrom, vid_paths, img_reconstFolder_path, img_filterFolder_path, img_encodingFolder_path, extraction_path, train_path, train_save_path, arch_path, BATCH_SIZE, IMAGE_SIZE, NUM_CHANNELS, input_shape, lr, workers, model_save_path, ae, transformations
    global flag_extractionCompleted, flag_trainingCompleted, flag_filteringCompleted, flag_moveFilteredCompleted
    global init_filePath
    init_filePath = "./Model/Final/init.json"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    min_losses = [100.0]

    with open(init_filePath, "r") as f:
        _init_data = json.load(f)
        
        EPOCHS = _init_data["init_constants"]["EpochsToTrain"]
        startFrom = _init_data["init_constants"]["StartFrom"]
        flag_extractionCompleted = bool(_init_data["currentVideoFlags"]["extractionCompleted"]) 
        flag_trainingCompleted = bool(_init_data["currentVideoFlags"]["trainingCompleted"])
        flag_filteringCompleted = bool(_init_data["currentVideoFlags"]["filteringCompleted"])

        _from_path = _init_data["Paths"]["Video"]["path"]
        vid_paths = getSortedVidPaths(_from_path)

        img_reconstFolder_path = _init_data["Paths"]["Save"]["img_reconst"]
        img_filterFolder_path = _init_data["Paths"]["Save"]["filtered"]
        img_encodingFolder_path = _init_data["Paths"]["Save"]["encoding"]

        extraction_path = _init_data["Paths"]["Extraction"]["path"]
        train_path = _init_data["Paths"]["Train"]["dataloader_path"]
        train_save_path = _init_data["Paths"]["Train"]["save_to_path"]
        arch_path = _init_data["Paths"]["Model"]["arch_path"]
        with open(arch_path, "r") as f:
            _init_arch = json.load(f)
            BATCH_SIZE, IMAGE_SIZE, NUM_CHANNELS  = _init_arch["init_constants"]["batch_size"], _init_arch["init_constants"]["image_size"], _init_arch["init_constants"]["channels"]
            input_shape = (BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
            lr = _init_arch["init_constants"]["lr"]
            workers = _init_arch["init_constants"]["workers"]
            
            model_save_path = _init_data["Paths"]["Model"]["model_path"]
            if (os.path.exists(model_save_path)):
                print("Retraining:", model_save_path.split("/")[-1])
                ae = torch.load(model_save_path).to(device)
            else:
                print("No model found.")
                print("Using Architecture: ", arch_path.split("/")[-1])
                ae = DenoisingAutoEncoder(arch=_init_arch["architecture"], input_shape=input_shape, noise_factor=_init_arch["init_constants"]["noise_factor"]).to(device)

        del _from_path, _init_arch

    transformations = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    return

def updateInitFile():
    _init_data["init_constants"]["StartFrom"] = startFrom
    _init_data["currentVideoFlags"]["extractionCompleted"] = flag_extractionCompleted
    _init_data["currentVideoFlags"]["trainingCompleted"] = flag_trainingCompleted
    _init_data["currentVideoFlags"]["filteringCompleted"] = flag_filteringCompleted

    with open(init_filePath, "w") as f:
        json.dump(_init_data, f, indent=4)
    print("\tUpdated Init File")
    return
    
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img.shape = (1,256, 256, 1)
    img = img.astype("float32")
    img = (img-127.5) / 127.5
    return img

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

def extractFrames(vid_path, to_path):
    print("\t" + "Extracting Frames...")
    max_frames = 40_000
    cam = cv2.VideoCapture(vid_path)
    current_frame = 0
    while(True):
        start_time = time.time()
        print("\t" + str(current_frame), end = "\r")
        ret, frame = cam.read()
        if (ret and current_frame < max_frames): 
            name = to_path + str(current_frame) + '.jpg'
            cv2.imwrite(name,frame)
            current_frame += 1
        else:
            break
        
        if time.time() - start_time > 10:
            print("Time limit exceeded. Exiting the loop.")
            break

    cam.release()
    cv2.destroyAllWindows() 
    print("\t" + "Total Extracted:", current_frame)
    return

def moveAll(from_path, to_path):
    
    file_list = os.listdir(from_path)
    print("Moving Files: >>>FROM - ", from_path, ">>>TO - ", to_path)
    if(len(file_list) > 0):
        for file_name in tqdm(file_list):
            file_name = os.path.join(from_path, file_name)
            try:
                _ = shutil.move(file_name, to_path)
            except:
                print("Alread Exists:", file_name.split("/")[-1] )
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

def moveFiltered(filtered_paths, dst_path):
    print("\tMoving Filtered Images...")
    for path in tqdm(filtered_paths):
        path = path.replace("\\","/")
        try:
            _ = shutil.move(path, dst_path)
        except:
            print("\t\tAlread Exists:", path.split("/")[-1] )
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



if __name__ == "__main__":

    setup()
    print("Using Device:", device)
    ae.getSummary(decoder=False)
    loss = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr = lr)
    start_epoch = 0
    init_time = start_time = time.time()
    
    for vid_id, vid_path in enumerate(vid_paths[startFrom:], start=startFrom):
        print("\n\n", "#"*80, sep="")
        print("\t\t\tVIDEO:", vid_id , "\t\t\t@time:", str(round((time.time() - start_time)/60, 2)) + "mins", "(" + str(round((time.time() - init_time)/60, 2)) + ")")
        print("#"*80, "\n")
        start_time = time.time()

        img_save_path = img_reconstFolder_path + "Video " + str(vid_id) + "/"
        filter_save_path = img_filterFolder_path + "Video " + str(vid_id) + "/"
        encoding_save_path = img_encodingFolder_path + "Video " + str(vid_id) + ".csv"

        if(not flag_extractionCompleted):
            makeFolder(path=train_save_path)
            extractFrames(vid_path=vid_path, to_path=train_save_path)
            flag_extractionCompleted = True
            updateInitFile()
        # moveAll(from_path=extraction_path, to_path=train_save_path)
        
        if(not flag_trainingCompleted or not flag_filteringCompleted):
            dataset = dset.ImageFolder(root = train_path, transform = transformations)
            dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = workers, pin_memory=True, timeout=120)
    
            if(not flag_trainingCompleted):
                makeFolder(path=img_save_path)
                ae = trainAE(ae, loss, optimizer, dataloader, start_epoch, EPOCHS, model_save_path, img_save_path)
                start_epoch += EPOCHS
                flag_trainingCompleted = True
                updateInitFile()

            if(not flag_filteringCompleted):
                enc, idx_list = getFilteredIndices(ae=ae, dataloader=dataloader)
                img_paths = getPaths(dataloader=dataloader)
                save_encodings(enc, img_paths, save_path = encoding_save_path)
                filtered_paths = [img_paths[i] for i in idx_list]
                makeFolder(path=filter_save_path)
                moveFiltered(filtered_paths=filtered_paths, dst_path=filter_save_path)
                
                flag_filteringCompleted = True
                updateInitFile()

        startFrom += 1
        flag_extractionCompleted = False
        flag_trainingCompleted = False
        flag_filteringCompleted = False
        updateInitFile()
    
    del ae
    exit()