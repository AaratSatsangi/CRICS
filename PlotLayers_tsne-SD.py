from sklearn.manifold import TSNE
import PyTorchNN
import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    version = input("version to use: ")
    path_modelSave = "Model/SceneDetector/version " + version + ".pt"
    data_path = "./Data/SceneDetector/train/"
    print("Using data path: ", data_path, "\n\n")

    model = torch.load(path_modelSave)
    model.eval()

    model.to(model.device)
    BATCH_SIZE = model.input_shape[0]
    IMG_SIZE  = model.input_shape[2]
    transformations = model.transformations


    dataset = dset.ImageFolder(root = data_path, transform = transformations)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory=True, timeout=120)
    print("Layer Names For Reference: ")
    for name, layer in model.model.named_children():
        print("\t" + name)

    # List of Layers from which the output has to be taken.
    listOfLayers = [
        '_2_Activation_LeakyReLU_',
        '_4_Activation_LeakyReLU_', 
        '_5_Flatten_', 
        '_6_Activation_LeakyReLU_', 
        '_7_Activation_LeakyReLU_', 
        '_8_Activation_LeakyReLU_',
        '_9_Linear_'
    ]

    layer_outputs = PyTorchNN.LayerOutputs(model, listOfLayers)
    layer_outputs.getOutputs(dataloader)

    labels = layer_outputs.labels
    features = layer_outputs.makeFeatures()

    print("Total Encodings in Each Layer: ", len(list(layer_outputs.outputs.values())[0]))
    print("Total Labels: ", len(labels))

    perp = int(input("Perplexity?(int)"))
    path = "./Layer Outputs/Perplexity " + str(perp) + "/"
    if(not os.path.exists(path)):
        os.mkdir(path)

    for i in range(len(listOfLayers)):
        X = np.array(features[listOfLayers[i]])
        
        print("Layer Name:", listOfLayers[i])
        print("Layer Shape:", X.shape)    
        
        tsne = TSNE(n_components = 2 , random_state = 1 , perplexity = perp , n_iter = 5000 , learning_rate = 50 , verbose = 5)
        tsne_data = tsne.fit_transform(X)
        tsne_df_1 = pd.DataFrame(data = tsne_data, columns = ['dim-1' , 'dim-2'])
        tsne_df_1["label"] = labels

        sns.set_style("darkgrid")
        sns.FacetGrid(data = tsne_df_1 , hue = "label" , height = 5).map(plt.scatter , 'dim-1' , 'dim-2').add_legend()
        plt.title(listOfLayers[i])
        plt.savefig(path + str(i) + " " + listOfLayers[i] + '.png')
        plt.show()

    img_paths = [] 
    for img_names in os.listdir(path):
        img_paths.append(os.path.join(path, img_names))

    def create_video(images, output_path, frame_rate=1):
        # Get dimensions of the first image
        image = cv2.imread(images[0])
        height, width, layers = image.shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        for image_path in tqdm(images):
            frame = cv2.imread(image_path)
            out.write(frame)

        # Release the VideoWriter
        out.release()

    # List of image file paths
    output_video_path = path + "train.avi"
    create_video(images=img_paths, output_path=output_video_path)

    exit("finished!")
