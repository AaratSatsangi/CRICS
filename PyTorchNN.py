import torch.nn as nn
from torchvision import utils as vutils
import torch
from collections import OrderedDict
from torchinfo import summary
import json
from sklearn.metrics import classification_report
from tqdm import tqdm

class Layers():

    def __init__(self, keys_path = "./Model Architecture/keys.json"):
        with open(keys_path, "r") as f:
            self.keys = json.load(f)["keys"]
    
    def getParams(self, keys, values):
        return {keys[i]:values[i] for i in range(len(values))}

    def getConvTranspose2dLayer(self, name, params):
        return {name: nn.ConvTranspose2d(in_channels = params.get("in_channels",0), 
                                                           out_channels = params.get("out_channels",0),
                                                           kernel_size = params.get("k_size",0),
                                                           stride = params.get("stride",1),
                                                           padding = params.get("padding",0), 
                                                           output_padding = params.get("out_padding",0),
                                                           groups=params.get("groups",1), 
                                                           bias=params.get("bias",True), 
                                                           dilation=params.get("dilation",1), 
                                                           padding_mode='zeros',
                                                           device=None,
                                                           dtype=None)}
    
    def getConv2dLayer(self, name, params):
        return {name: nn.Conv2d(in_channels = params.get("in_channels",0),
                                out_channels = params.get("out_channels",0),
                                kernel_size = params.get("k_size",0),
                                stride = params.get("stride",1),
                                padding = params.get("padding",0), 
                                groups=params.get("groups",1), 
                                bias=params.get("bias",True), 
                                dilation=params.get("dilation",1), 
                                padding_mode='zeros',
                                device=None,
                                dtype=None)}

    def getConv3dLayer(self, name, params):
        return {name: nn.Conv3d(in_channels = params.get("in_channels",0),
                                out_channels = params.get("out_channels",0),
                                kernel_size = params.get("k_size",0),
                                stride = params.get("stride",1),
                                padding = params.get("padding",0), 
                                groups=params.get("groups",1), 
                                bias=params.get("bias",True), 
                                dilation=params.get("dilation",1), 
                                padding_mode='zeros',
                                device=None,
                                dtype=None)
        }

    def getMaxpool2dLayer(self, name, params):
        return {name: nn.MaxPool2d(kernel_size=params["kernel_size"], 
                                   stride=params.get("stride",None), 
                                   padding=params.get("padding",0), 
                                   dilation=params.get("dilation",1), 
                                   return_indices=params.get("return_indices",False), 
                                   ceil_mode=params.get("ceil_mode", False))}

    def getLinearLayer(self, name, params):
        return {name: nn.Linear(in_features=params.get("in_features",1),
                                out_features=params.get("out_features",1),
                                bias=params.get("bias",True))}
    
    def getFlattenLayer(self, name, params):
        return {name: nn.Flatten(start_dim=params.get("start_dim",1),
                                 end_dim=params.get("end_dim",-1))}
    
    def getUnflattenLayer(self, name, params):
        return {name: nn.Unflatten(dim=params.get("input_shape", -1), unflattened_size=params.get("output_shape",-1))}
    
    def getBatchNorm2dLayer(self, name, params):
        return {name: nn.BatchNorm2d(params.get("num_features",0))}
    
    def getBatchNorm1dLayer(self, name, params):
        return {name: nn.BatchNorm1d(params.get("num_features",0))}
    
    def getActivationLayer(self, name, params = {}):
        activation = name.split("_")[3]
        
        if(activation == "ReLU"):
            layer = nn.ReLU(params.get("inplace",False))
        elif(activation == "Tanh"):
            layer = nn.Tanh()
        elif(activation == "LeakyReLU"):
            layer = nn.LeakyReLU(params.get("slope",0.01), params.get("inplace",False))
        elif(activation == "Sigmoid"):
            layer = nn.Sigmoid()
        elif(activation == "Softmax"):
            layer = nn.Softmax(dim=params.get("dim",1))
        
        else:
            raise Exception(f"The Activation Function, {activation} is not defined")
            
        return {name: layer}
    
    def getDropout2dLayer(self, name, params):
        return {name: nn.Dropout2d(p=params.get("p",0.5))}
    
    def getDropoutLayer(self, name, params):
        return {name: nn.Dropout(p=params.get("p",0.5))}
    
    def _hasNoWeights(self, name):
        keywords = ['Dropout',
                    'Flatten',
                    'Unflatten',
                    'Maxpool']
        return any([keyword in name for keyword in keywords])
    
    def init_weights(self, model, name):
        print("Initializing Weights & Biases for Model:", name, "...")
        for model in model.children():
            names_layers = [(name,layer) for name, layer in model.named_children()]
            j=0
            for i in range(len(names_layers)):
                if("Activation" in names_layers[i][0]):
                    while (j < i):
                        if(self._hasNoWeights(names_layers[j][0])):
                            j+=1
                            continue
                        if("LeakyReLU" in names_layers[i][0]):
                            nn.init.kaiming_normal_(names_layers[j][1].weight,a = names_layers[i][1].negative_slope, mode='fan_out') if ("Conv" in names_layers[j][0]) else nn.init.normal_(names_layers[j][1].weight, 1.0, 0.02)
                        elif("ReLU" in names_layers[i][0]):
                            nn.init.kaiming_normal_(names_layers[j][1].weight, mode='fan_out') if ("Conv" in names_layers[j][0]) else nn.init.normal_(names_layers[j][1].weight, 1.0, 0.02)
                        else:
                            nn.init.xavier_normal_(names_layers[j][1].weight)    
                        nn.init.constant_(names_layers[j][1].bias, 0.1)
                        j+=1
                    j+=1
                                         
class SequentialNN(nn.Module):

    def __init__(self, arch=None, input_shape=None, transformations=None, device='cpu', keys_path = "./Model Architecture/keys.json"):
        super(SequentialNN, self).__init__()
        self.Layers = Layers(keys_path)
        self.input_shape = input_shape
        self.device = device
        self.architecture = None
        self.compiled_models = None
        self.model_name = None
        self.model = None
        self.training_losses = []
        self.val_losses = []
        self.transformations = transformations
        self.startEpoch = 0
        if(arch):
            self.compile(arch)
            if(len(self.compiled_models) == 1):
                self.model_name = list(self.compiled_models.keys())[0]
                self.model = list(self.compiled_models.values())[0]
                
    def compile(self, arch):
        """"
        Provided the Architecture of models in a dictionary form, this method compiles all the models and the architecture in self.architecture and self.compiled_models
        The input arch should be in the format - 
        {
            "_1_ModelName":{
                "_1_LayerName_": _1_LayerParams,
                "_2_LayerName_": _2_LayerParams,
                ...
                },
            "_2_ModelName":{...},
            ...
        }
        """
        
        compiled_arch = {}
        compiled_models = {}
        for model in arch.keys():
            
            layers = []
            compiled_layers = OrderedDict()
            for layer_name in arch[model].keys():
                layer_params = arch[model][layer_name]
                
                if("Conv2d" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["Conv2d"], layer_params)
                    compiled_layers.update(self.Layers.getConv2dLayer(layer_name, layer_params))
                elif("Conv3d" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["Conv3d"], layer_params)
                    compiled_layers.update(self.Layers.getConv3dLayer(layer_name, layer_params))
                elif("ConvTranspose2d" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["ConvTranspose2d"], layer_params)
                    compiled_layers.update(self.Layers.getConvTranspose2dLayer(layer_name, layer_params))
                elif("Maxpool2d" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["Maxpool2d"], layer_params)
                    compiled_layers.update(self.Layers.getMaxpool2dLayer(layer_name, layer_params))
                elif("Activation" in layer_name):
                    if("_ReLU_" in layer_name):
                        layer_params = self.Layers.getParams(self.Layers.keys["ReLU"], layer_params)
                    elif("_Sigmoid_" in layer_name or "_Tanh_" in layer_name):
                        layer_params = {}
                    elif("_LeakyReLU_" in layer_name):
                        layer_params = self.Layers.getParams(self.Layers.keys["LeakyReLU"], layer_params)
                    elif("_Softmax_" in layer_name):
                        layer_params = self.Layers.getParams(self.Layers.keys["Softmax"], layer_params)
                    compiled_layers.update(self.Layers.getActivationLayer(layer_name, layer_params))
                elif("Linear" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["Linear"], layer_params)
                    compiled_layers.update(self.Layers.getLinearLayer(layer_name, layer_params))
                elif("Unflatten" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["Unflatten"], layer_params)
                    compiled_layers.update(self.Layers.getUnflattenLayer(layer_name, layer_params))
                elif("Flatten" in layer_name):
                    layer_params = {}
                    compiled_layers.update(self.Layers.getFlattenLayer(layer_name, layer_params))
                elif("BatchNorm" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["BatchNorm"], layer_params)
                    if("1d" in layer_name):
                        compiled_layers.update(self.Layers.getBatchNorm1dLayer(layer_name, layer_params))
                    else:
                        compiled_layers.update(self.Layers.getBatchNorm2dLayer(layer_name, layer_params))
                elif("Dropout" in layer_name):
                    layer_params = self.Layers.getParams(self.Layers.keys["Dropout"], layer_params)
                    if("2d" in layer_name):
                        compiled_layers.update(self.Layers.getDropout2dLayer(layer_name, layer_params))
                    else:
                        compiled_layers.update(self.Layers.getDropoutLayer(layer_name, layer_params))
                else:
                    raise Exception("Layer not named correctly!", layer_name)
                
                layers.append((layer_name, layer_params))
                    
            compiled_arch[model] = layers
            compiled_models[model] = nn.Sequential(compiled_layers)
            self.Layers.init_weights(compiled_models[model], model)
        
        self.architecture = compiled_arch
        self.compiled_models = compiled_models
        return  
    
    def _getXY(self, data):
        return data[0].to(self.device), data[1].to(self.device)

    def _doAfterStep(self, **kwargs):
        return
        
    def _doAfterEpoch(self, **kwargs):
        return

    def _binarizeUsingMax(self, t:torch.tensor):
        max_values, _ = t.max(dim=1, keepdim=True)
        return torch.where(t == max_values, torch.tensor(1.0), torch.tensor(0.0)).numpy()

    def _calcPerformMetrics(self, y_pred, y_true, class_names, path_saveDict):
        y_pred = self._binarizeUsingMax(y_pred)
        y_true = self._binarizeUsingMax(y_true)
        report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True, zero_division=0)
        with open(path_saveDict, 'w') as f:
            json.dump(report, f, indent=4)
            print("Results Written in:", path_saveDict)

        print("\nResults:")
        print(json.dumps(report, indent=4))
        return
    
    def _isDecreasing(self, lst):
        for i in range(len(lst) - 1):
            if lst[i] <= lst[i + 1]:
                return False
        return True
    
    def testSequential(self, test_loader, path_modelSave, loss, labels):
        best_model = torch.load(path_modelSave).to(self.device)
        y_trueTensor = torch.empty(0,3)
        y_predTensor = torch.empty(0,3)
        with torch.no_grad():
            test_loss = 0.0
            for test_data in tqdm(test_loader):
                x,y = self._getXY(test_data)
                y_pred = best_model(x)
                test_loss += loss(y_pred, y).item()
                
                y_true = torch.zeros(y.shape[0],3)
                for row in range(y.shape[0]):
                    y_true[row, y[row]] = 1
                y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
                y_predTensor = torch.vstack([y_predTensor, nn.functional.softmax(y_pred, dim=1).cpu()])

        del best_model
        test_loss /= len(test_loader)
        print("\t" +"Test Loss:", round(test_loss,5))
        path_saveDict = "/".join(path_modelSave.split("/")[:-1] + ["performance/" + path_modelSave.split("/")[-1].split(".")[0] + ".json"])
        self._calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor,class_names=labels, path_saveDict=path_saveDict)
        return

    def trainSequential(self, loss, optimizer, train_loader, val_loader, start_epoch, total_epoch, path_modelSave, persistence=0, test_loader=None, labels=None):
        
        print("\n", "#"*80, sep="")
        print("\t\t\tTraining:", str(self.model_name))
        print("#"*80, "\n")
        
        total_steps = len(train_loader)
        p_counter = 0
        
        for epoch in range(start_epoch, total_epoch):
            
            self.startEpoch = epoch
            
            # Training
            self.model.train()
            step_losses = []
            min_valLoss = min(self.val_losses) if(len(self.val_losses)) else 100
            print("\t" + "-"*100)
            print("\t" + "EPOCH: [%d/%d]\t\t\t\t\t\t\t\t\tp_counter: %d" % (epoch+1, total_epoch, p_counter))
            print("\t" +"-"*100)
            for i, train_data in enumerate((train_loader), 0):
                self.model.zero_grad()
                x,y = self._getXY(train_data)
                y_pred = self.model(x)
                error = loss(y_pred,y)
                error.backward()
                optimizer.step()
                step_losses.append(error.item())
                print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (i+1,total_steps,step_losses[-1]), end = "\r")
                self._doAfterStep()

            self.training_losses.append(round(sum(step_losses)/len(step_losses), 5))
            self._doAfterEpoch()
            print("\n\n\t" +"\tTraining Loss: %0.5f" % (self.training_losses[-1]))
            
            if(val_loader):
            # Validation
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for val_data in val_loader:
                        x,y = self._getXY(val_data)
                        y_pred = self(x)
                        val_loss += loss(y_pred, y).item()
                val_loss /= len(val_loader)
                self.val_losses.append(round(val_loss,5))
                print("\t" +"\tValidation Loss: %0.5f" % (self.val_losses[-1]), end = "\n\n")
            
                p_counter += 1
                if(len(self.val_losses) > 1):
                    if(self.val_losses[-1] < min_valLoss):
                        min_valLoss = self.val_losses[-1]
                        torch.save(self, path_modelSave)
                        p_counter = 0

                if(persistence and len(self.val_losses) >= persistence and p_counter == persistence):
                    print("Validation Loss constant for", persistence, "Epochs")
                    if(self._isDecreasing(self.training_losses[-persistence:])):
                        print("Stoping Training: Overfitting Detected")
                        break

                    else:
                        print("Training Loss fluctuating -- ", self.training_losses[-persistence:])
                        while(True):
                            flag = input("keep training?(y/n)")
                            if(flag == "y" or flag =="n"):
                                p_counter = 0
                                break
                            else:
                                print("Wrong Input")
                        if(flag == "n"):
                            break
            else:
                torch.save(self, path_modelSave)

        if(test_loader):       
            self.testSequential(test_loader, path_modelSave, loss, labels)
        return 
    
    def forward(self, input):
        return self.model(input)

    def getSummary(self, model=None):
        if(model):
            print(str(summary(model, self.input_shape, col_names=["input_size","output_size","num_params"])))
        else:
            print(str(summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])))
        return

class AutoEncoder(SequentialNN):
    
    def __init__(self, arch=None, input_shape=None, keys_path = "./Model Architecture/keys.json"):
        
        super(AutoEncoder, self).__init__(arch, input_shape, keys_path)
        self.encoder_model = self.decoder_model = None
        if(arch):
            self.encoder_model = self.compiled_models["encoder"]
            self.decoder_model = self.compiled_models["decoder"]
            self.model = nn.Sequential(OrderedDict({"encoder": self.encoder_model,
                                                 "decoder": self.decoder_model
                                                 }))
        print("Complete!")

    def getSummary(self, encoder = True, decoder = True):
        if(encoder and decoder):
            print(str(summary(self.model, self.input_shape, col_names=["input_size","output_size","num_params"])))
        elif(encoder):
            print(str(summary(self.encoder_model, self.input_shape, col_names=["input_size","output_size","num_params"])))
        else:
            print(str(summary(self.decoder_model, self.input_shape, col_names=["input_size","output_size","num_params"])))

    def forward(self, input):
        return self.model(input)
    
class DenoisingAutoEncoder(AutoEncoder):
    
    def __init__(self, arch=None, input_shape=None, keys_path = "./Model Architecture/keys.json", noise_factor = 0.8):
        super(DenoisingAutoEncoder, self).__init__(arch, input_shape, keys_path)
        self.noise_factor = noise_factor

    def forward(self, x):
        if (self.model.training):
            x += self.noise_factor*torch.randn_like(x)
        return self.model(x)

class LayerOutputs():
    def __init__(self, model:SequentialNN, listOfLayers:list):
        self.model = model
        self.outputs = {}
        self.labels = []
        self.outputLayerNames = listOfLayers

        # Register a forward hook for each layer
        for name, layer in self.model.model.named_children():
            if(name in self.outputLayerNames):
                layer.register_forward_hook(self.hook_fn(name))

    def hook_fn(self, name):
        def fn(module, input, output):
            out = output.detach().cpu().numpy()
            if(name not in self.outputs.keys()):
            #     for x in out:
            #         self.outputs[name].append(x)
            # else:
                self.outputs[name] = []
                # for x in out:
                #     self.outputs[name].append(x)
            for x in out:
                self.outputs[name].append(x)

        return fn
    
    def getOutputs(self, dataloader):
        for data in tqdm(dataloader):
            x,y = self.model._getXY(data)
            self(x)
            for label in y.detach().cpu().numpy():
                self.labels.append(label)

    def makeFeatures(self):
        features = {}
        for key in self.outputs.keys():
            temp = []
            for encoding in self.outputs[key]:
                temp.append(encoding.flatten())
            features[key] = temp
        return features

    def __call__(self, x):
        self.model(x)
        return
