
import Unet_basic_model
from Unet_basic_model import unet_model

import torch

import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from skimage.transform import resize
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob, os

use_gpu = -1 #-1 = use cpu, 0 = use first gpu, 1 = second gpu,...
img_size = (224, 224)
#imgpath = 'L:/COCO_Dataset/train2017/000000000257.jpg'
class_count = 10
class_list = ['car', 'chair', 'book', 'bottle', 'cup', 'dining table', 'traffic light', 'bowl', 'handbag', 'bird']
modelpath = 'Objects.pt'


if (torch.cuda.is_available() and (use_gpu!=-1)):
    print ("Using gpu: ", torch.cuda.get_device_name(use_gpu))
else:
    print ("USing cpu")

device = torch.device("cuda:"+str(use_gpu) if (torch.cuda.is_available() and (use_gpu!=-1)) else "cpu")

def predict_img (imgpath, model):
    image = Image.open(imgpath)
    image = image.resize (img_size)
    img = TF.to_tensor(image)
    img.unsqueeze_(0)
    img = img.to(device)
    output = model(img)
    #output = torch.argmax(output, 1)
    
    return output


def visualize (imgpath, prediction):
    image = Image.open(imgpath)
    image = image.resize (img_size)
    image = np.array(image, dtype=np.uint8)
    
    temp = np.zeros(prediction[0].shape, dtype = prediction[0].dtype)
    temp[temp == 0] = np.min (prediction)
    prediction = np.insert (prediction, 0, temp, axis = 0)
    niggar = np.argmax (prediction, 0)
    plt.imsave("Result/" + str(os.path.basename(imgpath)) + "_OG.png", image)
    for i in range (0, prediction.shape[0]-1):
        nigga = np.copy(niggar)
        nigga[nigga!=i+1] = 0
        if (np.all (nigga == 0) == False):
            nigga = (nigga*255/np.max(nigga))
        else:
            continue
        image[:,:,0] = nigga
        plt.imsave(("Result/"+ str(os.path.basename(imgpath)) +"_"+str(i)+"_"+class_list[i]+".png"), image)
    nigga = np.copy(niggar)
    nigga[nigga!=0] = 1
    nigga[nigga!=1] = 255
    nigga[nigga!=255] = 0
    image[:,:,0] = nigga
    plt.imsave(("Result/"+ str(os.path.basename(imgpath)) +"_background.png"), image)
    return niggar

def predict_folder (folder_path, model):
    for i in glob.glob (folder_path):
        prediction = predict_img (i, model)
        visualize (i, prediction.detach().numpy()[0])

if __name__ == "__main__":
    model = Unet_basic_model.unet_model.UNet(n_channels = 3, n_classes = class_count)
    model.load_state_dict(torch.load(modelpath))
    model = model.to(device)
    model.eval()
    predict_folder ("./Image/*.jpg", model)
    """print ("input '0' for default path (testImg.png)")
    print ("input '-1' to exit")
    while (True):
        img_path = input("Input image path: ")
        if (img_path == '0'):
            print ("Default path selected")
            img_path = imgpath
        elif (img_path == '-1'):
            break
        prediction = predict_img (model, img_path)
        visualize (img_path, prediction.detach().numpy()[0])"""
