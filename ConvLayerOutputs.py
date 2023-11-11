import PyTorchNN
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')


def normalize_to_range(arr, new_min=0, new_max=255):
    arr = np.array(arr)
    old_min = np.min(arr)
    old_max = np.max(arr)
    normalized_arr = (arr - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return normalized_arr

def create_video(images, output_path, frame_rate=1):
    # Get dimensions of the first image
    image = cv2.imread(images[0])
    height, width, layers = image.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    for image_path in tqdm(images):
        frame = cv2.imread(image_path)
        for i in range(4):
            out.write(frame)

    # Release the VideoWriter
    out.release()

if __name__ == "__main__":
    version = input("version to use: ")
    path_modelSave = "Model/SceneDetector/version " + version + ".pt"
    data_path = "./Data/SceneDetector/test/"
    print("Using data path: ", data_path, "\n\n")

    model = torch.load(path_modelSave)
    model.eval()

    model.to(model.device)
    BATCH_SIZE = model.input_shape[0]
    IMG_SIZE  = model.input_shape[2]
    transformations = model.transformations

    num = 100    

    dataset = dset.ImageFolder(root = data_path, transform = transformations)
    arr = random.sample(range(0, len(dataset.imgs)), num)

    dataset = torch.utils.data.Subset(dataset, arr)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, timeout=120)

    # List of Layers from which the output has to be taken.
    listOfLayers = [
        '_1_Conv2d_',
        '_2_Conv2d_',
        '_3_Conv2d_',
        '_4_Conv2d_',
        '_5_Conv2d_'
    ]

    layer_outputs = PyTorchNN.LayerOutputs(model, listOfLayers)
    layer_outputs.getOutputs(dataloader)

    labels = layer_outputs.labels
    features = layer_outputs.makeFeatures()

    print("Total Encodings in Each Layer: ", len(list(layer_outputs.outputs.values())[0]))
    print("Total Labels: ", len(labels))

    labels = np.array(layer_outputs.labels)
    labels_idx = {
        'Bowling':np.where(labels == 0)[0],
        'Field':np.where(labels == 1)[0],
        'Misc':np.where(labels == 2)[0]
    }

    main_path = './Conv Layer Outputs/'
    paths = {}
    for cls in labels_idx.keys():
        paths[cls] = os.path.join(main_path,cls)
        if(not os.path.exists(paths[cls])):
            os.mkdir(paths[cls])
        

    num_filters = (layer_outputs.outputs[listOfLayers[-1]][0]).shape[0]
    cols = 4
    rows = int(num_filters/cols)

    print("-"*70)
    print("\t\tGeneration Images")
    print("-"*70)
    for cls in labels_idx.keys():
        print("\tGenerating for", cls)
        image_to_use = np.random.choice(labels_idx[cls])

        for k, layer_name in enumerate(tqdm(listOfLayers)):
            
            original_array = layer_outputs.outputs[layer_name][image_to_use]
            normalized_array = normalize_to_range(original_array)
            
            random_filter_idx = random.sample(range(0, normalized_array.shape[0]), num_filters)    
            fig, axes = plt.subplots(rows, cols, sharex = True, figsize=(16*cols, 16*rows))
            plt.rcParams.update({'font.size': 100})
            plt.suptitle("Output of " + layer_name + "\nTotal Filters: " + str(normalized_array.shape[0]), fontsize=100)

            for i in range(rows):
                for j in range(cols):
                    filter_num = random_filter_idx[i * cols + j]
                    image = normalized_array[filter_num, :, :]
                    axes[i, j].imshow(image, cmap='gray', aspect = 'auto')
                    axes[i, j].axis('off')
                    axes[i, j].set_title("FNum: " + str(filter_num))
                    


            plt.savefig(paths[cls] + "/" + str(k) + ". " + layer_name + '.png')
            plt.close()

    print("-"*70)
    print("\t\tMaking Videos")
    print("-"*70)
    for cls in labels_idx.keys():
        print("\tMaking for", cls)
        path = paths[cls]
        unsorted = []
        for name in os.listdir(path):
            if('.png' in name or '.jpg' in name):
                unsorted.append(name)

        sorted = [""]*len(unsorted)
        for name in unsorted:
            idx = int(name.split('.')[0])
            sorted[idx] = os.path.join(path, name)

    # List of image file paths
        output_video_path = path + "/" + cls + "_outputs.avi"
        create_video(images=sorted, output_path=output_video_path)