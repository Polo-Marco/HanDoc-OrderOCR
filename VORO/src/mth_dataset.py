from torch.utils.data import Dataset, DataLoader
import torch
import json
import os
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

class MTHv2PairDataset(Dataset):
    '''MTHv2PairDataset'''
    def __init__(self, data_path,RegionType="line", transform=None):
        """
        Args:
            data_path (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path 
        with open(data_path,'r') as f:
            self.data = json.load(f)
        self.RegionType = RegionType
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        data = self.data[idx]
        normalized_feature = [i/data["width"] if idx%2==0 else i/data["height"] for idx,i in enumerate(data["feature"])]
        sample = {"feature":torch.tensor(normalized_feature,dtype=torch.float64),
                "label":torch.tensor(data["label"],dtype=torch.float64),
                "index":str(data["index"]),
                "image":data["image"]}
        return sample
    
class MTHv2PairImageDataset(Dataset):
    """MTHv2PairImageDataset"""

    def __init__(self, data_path,image_path,image_flag,
                 RegionType="line", augmentation=False,image_size=224):
        """
        Args:
            data_path (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path 
        with open(data_path,'r') as f:
            self.data = json.load(f)
        self.RegionType = RegionType
        self.image_size = image_size
        self.image_flag = image_flag
        self.feature_len = len(self.data[0]["feature"])
        for data in self.data:
             data["feature"] = self.resize_coor(data["feature"],(data["width"],data["height"]))
        if augmentation:
            #self.augmentation = T.Compose([T.RandomAffine(degrees= 20 ,scale=(0.7,1))])
            self.augmentation = T.Compose([T.RandomAffine(degrees=0,scale=(1,1),translate=(0,0.1))])
        else:
            self.augmentation = None
        self.images={}
        if self.image_flag:
            print("loading images...")
            transform = T.Resize((image_size,image_size))
            for image in tqdm(set([d["image"] for d in self.data])):
                self.images[image]=np.array(transform(
                    Image.open(image_path+image+".jpg").convert("L")
                ))#read all images
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        data = self.data[idx]
        poly1 = data["feature"][:self.feature_len//2]
        poly2 = data["feature"][self.feature_len//2:]
        mask1,mask2 = self.__get_polygon_mask__(poly1),self.__get_polygon_mask__(poly2)
        #label = [[0,1] if data["label"] else [1,0] ][0]
        if self.image_flag:
            image = self.images[data["image"]]
            feature = torch.tensor(np.array([image/255,mask1/255,mask2/255]),dtype=torch.float64)
        else:
            mask3 = np.zeros((self.image_size,self.image_size))
            feature = torch.tensor(np.array([mask3,mask1/255,mask2/255]),dtype=torch.float64)
        if self.augmentation:
                feature = self.augmentation(feature)
        sample = {"feature":feature,
                "label":torch.tensor(data["label"],dtype=torch.float64),
                "index":str(data["index"]),
                "image":data["image"]}
        return sample
    def _transform_img(self,img):
        '''
        input: PIL image format
        output: transformed np array
        '''
        return np.asarray((img.convert('L')))
    
    def resize_coor(self,coors, ori_size):
        '''
        input: list of boxes, tuple(w,h) of ori, tgt size
        output: list of resized boxes
        turn longest side into tgt_size
        '''
        x_ratio, y_ratio = float(self.image_size)/float(ori_size[0]),float(self.image_size)/float(ori_size[1])
        return [round(i*x_ratio) if idx%2==0 else round(i*y_ratio) for idx,i in enumerate(coors)]

    def __get_polygon_mask__(self,poly_box):
        contours = np.array(poly_box).reshape(4,2)
        mask = np.zeros((self.image_size,self.image_size))
        cv2.fillPoly(mask, pts = [contours], color =(255,255,255))
        return mask
        
    
if __name__ == "__main__":
    dpath = "../../datasets/MTH_data/pair/sent_pair_test_3.json" 
    ipath = "../../datasets/MTH_data/imgs/"
    ds = MTHv2PairImageDataset(dpath,ipath,augmentation=True)
    dl = DataLoader(ds,batch_size=2)
    #print((ds[0]))
    for i in dl:
        print(i)
        exit()    




