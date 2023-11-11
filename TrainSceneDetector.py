from PyTorchNN import SequentialNN
from torchvision import transforms
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as dset
import json
import os
import shutil

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

def updateInitConstants():
    init_data["init_constants"]["startEpoch"] = startEpoch
    init_data["init_constants"]["epochsToTrain"] = EPOCHS
    init_data["init_constants"]['Flags']['train'] = flag_training
    init_data["init_constants"]['useForPredict'] = useForPredict
    with open(path_initFile, "w") as f:
        json.dump(init_data, f, indent=4)
    print("\t>>>>Updated Init Constants")
    return

def setup(version, init_path = "./Model/SceneDetector/init.json"):

    global EPOCHS, IMG_SIZE, BATCH_SIZE, CHANNELS, startEpoch, input_shape, lr, workers, persist
    global device, detector, transformations, useForPredict, labels, tvt_split
    global path_trainDataloader, path_savePredictFolder, path_initFile, path_modelSave, path_predictFromFolder
    global flag_training, flag_predict
    global init_data

    path_initFile = init_path
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    with open(init_path, "r") as f:
        init_data = json.load(f)

    path_trainDataloader = init_data['Paths']['TrainFolder']
    path_savePredictFolder = init_data['Paths']['SavePredictFolder']
    path_modelArch = init_data['Paths']['ModelArchFolder'] + "version " + str(version) + ".json"
    path_predictFromFolder = init_data['Paths']['PredictFromFolder'] #List of Paths
    path_modelSave = init_data['Paths']['ModelSaveFolder'] + "version " + str(version) + ".pt"

    labels = init_data['init_constants']['Labels']
    tvt_split = init_data['init_constants']['TrainValTest_Split']
    IMG_SIZE = init_data['init_constants']['image_size']
    BATCH_SIZE = init_data['init_constants']['batch_size']
    CHANNELS = init_data['init_constants']['channels']
    input_shape = (BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE)
    EPOCHS = init_data['init_constants']['epochsToTrain']
    startEpoch = init_data['init_constants']['startEpoch']
    lr = init_data['init_constants']['lr']
    workers = init_data['init_constants']['workers']
    flag_training = init_data["init_constants"]['Flags']['train']
    flag_predict = init_data["init_constants"]['Flags']['predict']
    useForPredict = init_data["init_constants"]['useForPredict']
    persist = init_data["init_constants"]['persist']
    
    transformations = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    if(os.path.exists(path_modelSave)):
        print("Retraining:", path_modelSave.split("/")[-1])
        detector = torch.load(path_modelSave).to(device)
    else:
        print("Training Fresh Model")
        print("Using Architecture: ", path_modelArch.split("/")[-1])
        with open(path_modelArch, "r") as f:
            model_data = json.load(f)
    
        detector = SequentialNN(arch=model_data['architecture'], 
                                input_shape=input_shape, 
                                transformations=transformations,
                                device=device).to(device)
        init_data['init_constants']['startEpoch'] = startEpoch = 0
        init_data["init_constants"]['Flags']['train'] = flag_training = True
        init_data["init_constants"]['Flags']['predict'] = flag_predict = True

        updateInitConstants()

    print("Setup Complete!")
    return

def getPaths(dataloader):
    img_path = []
    for paths in dataloader.sampler.data_source.imgs:
        img_path.append(paths[0])
    return img_path

def predict(pred_loader, save_labelMap):
    
    print("\t" + "Predicting...")
    y = []
    for data in tqdm(pred_loader):
        y += torch.argmax(detector(data[0].to(device)), dim = 1).tolist()
    
    print("\t" + "Copying to Folders...")
    img_paths = getPaths(dataloader=pred_loader)
    j = [len(os.listdir(path)) for path in save_labelMap.values()]
    for i in tqdm(range(len(img_paths))):
        path = img_paths[i]
        dst_path = save_labelMap[y[i]]
        new_path = shutil.copy(path, dst_path)
        os.rename(new_path, dst_path + str(j[y[i]]+i) + ".jpg")

    global useForPredict
    useForPredict = useForPredict + 1
    updateInitConstants()
    return

if __name__ == "__main__":
    
    gen = torch.Generator().manual_seed(26)

    version = input("version to use: ")
    setup(version)
    detector.getSummary()

    while(True):
        dataset = dset.ImageFolder(root = path_trainDataloader, transform = transformations)
        if(flag_training):
            if(len(tvt_split) == 3):
                train_dataset, val_dataset, test_dataset = random_split(dataset, tvt_split, generator=gen)
            elif(len(tvt_split) == 2):
                train_dataset, val_dataset = random_split(dataset, tvt_split, generator=gen)
                test_dataset = None
            else:
                train_dataset = dataset
                val_dataset = test_dataset = None
            
            if(train_dataset):
                train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = workers, pin_memory=True, timeout=120)
            else:
                train_loader = None
            if(val_dataset):
                val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = workers, pin_memory=True, timeout=120)
            else:
                val_loader = None
            if(test_dataset):
                test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = workers, pin_memory=True, timeout=120)
            else:
                test_loader = None
                
            loss = nn.CrossEntropyLoss()
            optimizer = optim.Adam(detector.parameters(), lr = lr)
            
            detector.trainSequential(loss=loss,
                                    optimizer=optimizer,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    test_loader=test_loader,
                                    start_epoch=startEpoch,
                                    total_epoch=EPOCHS,
                                    path_modelSave=path_modelSave,
                                    labels=labels,
                                    persistence=persist)
            
            startEpoch = detector.startEpoch
            updateInitConstants()

        while(True):
            trainMore = input("Train More?(y/n): ")
            if(trainMore == "n" or trainMore == "y"):
                break
            else:
                print("Wrong input!")
        
        if(trainMore == "y"):
            EPOCHS += int(input("Epochs to Train?(int): "))
            flag_training = True
            updateInitConstants()
            continue

        elif(trainMore == "n"):
            flag_training = False
            updateInitConstants()
            
        if(len(path_predictFromFolder) > useForPredict):
            while(True and os.path.exists(path_predictFromFolder[useForPredict])):
                pred = input("Predict?(y/n): ")
                if(pred == 'y'):
                    while(True):
                        flag_delete = input(f"Delete Files From Predict Folder?({path_savePredictFolder}) (y/n): ")
                        if(flag_delete == "y"):
                            for path in os.listdir(path_savePredictFolder):
                                deleteFiles(folder_path=os.path.join(path_savePredictFolder,path))
                            break
                        elif(flag_delete == "n"):
                            break
                        else:
                            print("Invalid Input")
                    path_savePredictLabelMap = {value: path_savePredictFolder + key + "/" for key,value in dataset.class_to_idx.items()}
                    pred_dataset = dset.ImageFolder(root = path_predictFromFolder[useForPredict], transform = transformations)
                    pred_loader = DataLoader(pred_dataset, batch_size = BATCH_SIZE, num_workers = workers, pin_memory=True, timeout=120)
                    predict(pred_loader=pred_loader, save_labelMap=path_savePredictLabelMap)
                elif(pred == 'n'):
                    flag_predict = False
                    updateInitConstants()
                    break
                else:
                    print("Wrong Input!")
            
        if(not flag_training and (not flag_predict or not path_predictFromFolder)):
            break
    
    exit("Finished")