import Unet_basic_model
from Unet_basic_model import unet_model

import os, glob

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

import numpy as np
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt

use_gpu = -1 #-1 = use cpu, 0 = use first gpu, 1 = second gpu,...
img_size = (224, 224)
#imgpath = 'L:/COCO_Dataset/train2017/000000000257.jpg'
class_count = 1
class_list = ['Human']
modelpath = 'Human.pt'


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
    print (img.shape)
    output = model(img)
    #output = torch.argmax(output, 1)
    
    return output


def visualize (imgpath, prediction):
    image = Image.open(imgpath)
    image = image.resize (img_size)
    image = np.array(image, dtype=np.uint8)
    
    plt.imsave("Result/" + str(os.path.basename(imgpath)) + "_0_OG.png", image)
    if (np.min(prediction [0]) < 0):
        prediction[0] = prediction[0] + np.min(prediction[0])
    prediction[0] = prediction[0]/np.max(prediction[0]) * 255
    image[:,:,0] = prediction[0]
    plt.imsave(("Result/"+ str(os.path.basename(imgpath)) +"_"+str(0)+"_"+class_list[0]+".png"), image)
    return image
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
        print ("File chosen: ", img_path)
        prediction = predict_img (img_path, model)
        visualize (img_path, prediction.detach().numpy()[0])"""
