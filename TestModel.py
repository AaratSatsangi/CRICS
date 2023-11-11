import PyTorchNN
import torch
from torchvision import transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    version = input("version to use: ")
    path_modelSave = "Model/SceneDetector/version " + version + ".pt"
    model = torch.load(path_modelSave)
    test_path = "./Data/SceneDetector/test/"
    print("Using test path: ", test_path)
    model.to(model.device)
    BATCH_SIZE = model.input_shape[0]
    IMG_SIZE  = model.input_shape[2]
    transformations = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])

    
    dataset = dset.ImageFolder(root = test_path, transform = transformations)
    test_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory=True, timeout=120)
    PyTorchNN.SequentialNN.testSequential(model, test_loader=test_loader, path_modelSave=path_modelSave, loss=torch.nn.CrossEntropyLoss(), labels=["Bowling","Field","Misc"])

    exit("Finished!")

