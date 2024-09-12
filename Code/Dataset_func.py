#Reference: https://github.com/mkisantal/backboned-unet/tree/master

import numpy as np
import glob, os, json
import gc

from pycocotools.coco import COCO
from pycocotools.mask import decode as decocode
from PIL import Image
from matplotlib import pyplot as plt
import torch.utils.data as data
import torch
annotation_dir = "L:/COCO_Dataset/COCO_annotations_trainval2017/annotations"
img_dir = "L:/COCO_Dataset/train2017"


#https://pytorch.org/docs/0.4.0/_modules/torchvision/datasets/coco.html#CocoDetection
class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def get_top_N_classes (self, num_class = 10, starting_class = 0):
        assert num_class > 0, "Num_Class should be bigger than 0"
        assert starting_class >= 0, "starting_class should be equal or bigger than 0"
        topN = sorted(self.get_count_per_class(self.annFile_path).items(), key=lambda item: item[1], reverse=True)
        return [topN[x][0] for x in range (starting_class, (num_class+starting_class))]
    def get_top_N_classes_ID (self):
        selected_classes = self.get_top_N_classes(self.class_count, self.start_class)
        return self.coco.getCatIds(catNms=selected_classes)
        
    def getImgIds (self):
        selected_cat_Ids = self.get_top_N_classes_ID()
        result_id = []
        for single_cat_id in selected_cat_Ids:
            result_id.append (self.coco.getImgIds(catIds=single_cat_id))
        return np.array(list(set().union(*result_id)))
    

    def __init__(self, root, annFile, transform=None, target_transform=None, class_count = 10, start_class = 1):
        from pycocotools.coco import COCO
        self.root = root
        self.annFile_path = annFile
        self.coco = COCO(annFile)
        self.transform = transform
        self.target_transform = target_transform
        self.class_count = class_count
        self.start_class = start_class
        self.ids = self.getImgIds()
        self.catIds = self.get_top_N_classes_ID()

    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.catIds)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(int(img_id))[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        
        mask = np.zeros((len(self.catIds), *coco.annToMask(target[0]).shape), dtype = np.uint8)
        
        for i in target:
            mask[self.catIds.index(i['category_id'])] += np.array(coco.annToMask(i))
        
        if self.target_transform is not None:
            list_temp = []
            for i in range (0, mask.shape[0]):
                temp = self.target_transform(mask[i])
                temp[temp>0] = 1
                list_temp += temp
        target = torch.stack(list_temp)
        del list_temp
        #gc.collect()
        
        """
        mask = np.zeros(coco.annToMask(target[0]).shape, dtype = np.uint8)
        #print ("Target: ",target)
        #print (coco.annToMask(target[0]))
        #print (np.all(coco.annToMask(target[0])==0))
        
        for i in target:
            #print ("i: ",i)
            #print (coco.annToMask(i))
            #print (np.all(coco.annToMask(i)==0))
            gay = np.array(coco.annToMask(i))
            #print (np.all(gay==0),' ',gay.shape)
            print (self.catIds.index(i['category_id']))
            gay [gay>0] = self.catIds.index(i['category_id'])
            mask = mask + gay
        print ("mask.shape ", mask.shape)
        plt.imshow (mask)
        #test_mask = np.copy(mask)
        #test_mask[test_mask > 0] = 255
        #plt.imshow (test_mask)
        print("Unique mask: ",np.unique (mask))
        target = self.target_transform(mask)
        if (index < 10):
            print (np.all(mask==0))
            print (np.all(target.numpy()==0))
            print (np.all(target.squeeze().numpy()==0))
            print (np.all(target.squeeze().long().numpy()==0))
            #print (mask.shape)
            #print (target.squeeze().shape)
        """
        return img, target


    def __len__(self):
        #return len(self.ids)
        return (np.array(self.ids).shape[0])

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def get_count_per_class(self, annFile):
        """
        Count per class given COCO annotations in dict format.
        https://stackoverflow.com/a/75559680
        """
        with open(annFile) as f:
            coco_dict = json.load (f)
        id_to_class_name = {x['id']: x['name'] for x in coco_dict['categories']} 
        annotations = coco_dict['annotations']
        
        counts = {}
        for annotation in annotations:
            class_name = id_to_class_name[annotation['category_id']]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts



def ShowCOCOImage (coco, img_dir, image_id = 1):
    img = coco.imgs[image_id]
    image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
    #plt.imshow(image, interpolation='nearest')
    
    cat_ids = coco.getCatIds()
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    coco.showAnns(anns)
    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(image)
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(mask)
    fig.show()

def readJsonDataset (path_to_coco):
    coco = COCO (os.path.join(path_to_coco, "instances_train2017.json"))
    return coco
if __name__ == "__main__":
    #coco = readJsonDataset (annotation_dir)
    #ShowCOCOImage (coco, img_dir, 74)
    pass
