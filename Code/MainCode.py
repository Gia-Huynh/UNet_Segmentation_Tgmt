
import Dataset_func
import Unet_basic_model
from Unet_basic_model import unet_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from skimage.transform import resize
from sklearn import metrics
import cv2

#print ("Testing: ")
#print (100,'%,',5,'/',5,',Loss: %.3f'%(0.1 * 32),',ETA: ',1)

load_best = 0
epochs = 4
batch_size = 44
use_gpu = 0
class_count = 1
start_class = 0 #skip first N class
num_workers = 4

img_size = (224, 224)
learning_rate = 0.02
img_dir = "L:/COCO_Dataset/train2017"
val_dir = "L:/COCO_Dataset/val2017"
annotation_dir = "L:/COCO_Dataset/COCO_annotations_trainval2017/annotations"
best_model_params_path = "./BestModel_HUMAN_ONLY.pt"

print ("Use gpu: ", use_gpu)
if (torch.cuda.is_available() and (use_gpu!=-1)):
    print ("Using gpu: ", torch.cuda.get_device_name(use_gpu))
device = torch.device("cuda:"+str(use_gpu) if (torch.cuda.is_available() and (use_gpu!=-1)) else "cpu")
cudnn.benchmark = True
plt.ion()   # interactive mode

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(img_size[0], interpolation = transforms.InterpolationMode.NEAREST, antialias = False),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation (5),
        transforms.ColorJitter(brightness = 0.1, contrast=0.1),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size[0], interpolation = transforms.InterpolationMode.NEAREST, antialias = False),
        transforms.CenterCrop(img_size[0]),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#cap = Dataset_func.CocoDetection(root = val_dir,
#                        annFile = os.path.join(annotation_dir, "instances_val2017.json"))
#cap = Dataset_func.CocoDetection(root = img_dir,
#                        annFile = os.path.join(annotation_dir, "instances_train2017.json"))

def plot_count_per_class (coco):
    class_count = dict(sorted(coco.get_count_per_class ().items(), key=lambda item: item[1]))
    lists = sorted(class_count.items(), key=lambda item: item[1], reverse=True)
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.xticks(rotation=90)

    
#plot_count_per_class(cap)
#a, b = cap[4]
#print("Number of images containing all the classes:", cap.__len__())

#PAUSE
#print("Image Size: ", img.size())
#print(target)
    
#instances_train2017
#img_dir

if __name__ == "__main__":
    image_datasets = {         'val': Dataset_func.CocoDetection(root = val_dir,
                                        annFile = os.path.join(annotation_dir, "instances_val2017.json"),
                                        transform = data_transforms['val'],
                                        target_transform = data_transforms['val'],
                                        class_count=class_count,
                                        start_class=start_class),
                               'train': Dataset_func.CocoDetection(root = img_dir,
                                        annFile = os.path.join(annotation_dir, "instances_train2017.json"),
                                        transform = data_transforms['val'],
                                        target_transform = data_transforms['val'],
                                        class_count=class_count,
                                        start_class=start_class)}
    print('Number of samples: ', len(image_datasets['train']))
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                 shuffle=False, num_workers=num_workers)
                   ,'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
                                                 shuffle=False, num_workers=0)
                   }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
#image_datasets['train'].coco
#image_datasets['train'][5]
#PAUSE               
def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs):
    since = time.time()
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_loss = 99999
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                i = 0
                prev_i = 0
                starting_time = 0
                iter_steps = len(image_datasets[phase])//batch_size
                gay_time = time.time()
                for inputs, labels_mask in dataloaders[phase]:
                    if (starting_time == 0):
                        starting_time = time.time()
                    #print ("Ay")
                    inputs = inputs.to(device)
                    labels_mask = labels_mask.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        #print (outputs.shape)
                        #_, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels_mask)
                        #print (outputs)
                        #print (labels_mask)
                        #print (outputs.shape,' ',outputs.dtype)
                        #print (labels_mask.shape,' ',labels_mask.dtype)                     
                        #if (i == prev_i):
                        #    try:
                        #        print (loss)
                        #    except:
                        #        pass
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        else:
                            pass

                    # statistics
                    
                    i=i+1
                    if ((phase == 'train') and (i == prev_i + iter_steps//10)) or ((phase == 'val') and (i == prev_i + len(image_datasets[phase])//2)):
                        print (int(i/iter_steps * 1000)/10,'%,',i,'/',iter_steps,',Average loss:%.3f'%(running_loss/(i*batch_size)),
                               ',ETA:',int((time.time() - starting_time)/i*(iter_steps-i)),'Time taken:',int(time.time() - starting_time))
                        prev_i = i
                    
                    gay_time = time.time()
                    running_loss = running_loss + (loss.item() * batch_size)
                    #print ("Running_loss time: ", time.time() - gay_time)  
                    #running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                #epoch_acc = running_corrects.double() / dataset_sizes[phase]
                #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(f'{phase} Loss: {epoch_loss:.4f}')
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    #best_acc = epoch_acc
                    print ("Best current model found, saving...")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), best_model_params_path)
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}, acc: {best_acc:4f} ')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        torch.save(model.state_dict(), best_model_params_path + "_bak")
    return model
#PAUSE
if __name__ == "__main__":
    print ("Building Model")

    model = Unet_basic_model.unet_model.UNet(n_channels = 3, n_classes = class_count)
    if (load_best == 1):
        model.load_state_dict(torch.load("./BestModel.pt"))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MultiLabelMarginLoss() 
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.95)
    # Decay LR by a factor of 0.05 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.05)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epochs)
    
    #for param in model.parameters():
    #        param.requires_grad = True
    torch.cuda.empty_cache()
    #model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
    #                       num_epochs=epochs)
    
    #print (model.layer4[2].conv3.weight.grad)
    #visualize_grad_cam (model)
    torch.save(model.state_dict(), 'wtf.model')
    #from torchsummary import summary
    #summary (model, (3, 512, 512))
    #visualize_model(model)
