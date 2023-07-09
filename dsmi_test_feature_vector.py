
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch
import torchvision
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
import sklearn
import math
from matplotlib import patches
from PIL import Image, ImageDraw
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils
import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.nn.functional as F
import collections
from torch.jit.annotations import Tuple, List, Dict, Optional
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
class UNET(nn.Module):
    
    def __init__(self):
        super(UNET, self).__init__()
        self.num_classes = 2
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=256)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=256, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out,contracting_42_out

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Dataset_(Dataset):
    def __init__(self, image_dir, fmask_dir,bmask_dir, transform=None):
        self.image_dir = image_dir
        self.fore_mask_dir = fmask_dir
        self.back_mask_dir = bmask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.mask1_dir= os.listdir(fmask_dir)
        self.mask2_dir = os.listdir(bmask_dir)

    def __len__(self):
        return len(self.fore_mask_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        fore_mask_path = os.path.join(self.fore_mask_dir, self.images[index])
        back_mask_path = os.path.join(self.back_mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        image2=image
        fore_mask = np.array(Image.open(fore_mask_path).convert("RGB"))
        # fore_mask[fore_mask == 255.0] = 1.0
        back_mask = np.array(Image.open(back_mask_path).convert("RGB"))
        # back_mask[back_mask == 255.0] = 1.0
        # print(type(image)," ",type(fore_mask)," ",type(back_mask))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=fore_mask)
            # ,back_mask=back_mask)
            image = augmentations["image"]
            fore_mask = augmentations["mask"]
            augmentations2 = self.transform(image=image2, mask=back_mask)
            # ,back_mask=back_mask)
            # image = augmentations2["image"]
            back_mask = augmentations2["mask"]
            # back_mask = augmentations["back_mask"]
            # image=self.transform(image=image)
            # fore_mask=self.transform(image=fore_mask)
            # back_mask=self.transform(image=back_mask)
            
        # print(type(image)," ",type(fore_mask)," ",type(back_mask))
        # print(image.shape," ",fore_mask.shape," ",back_mask.shape)
        return image,fore_mask,back_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_new=UNET().to(device)
model_new = UNET().to(device)

model_new.load_state_dict(torch.load('unet_model_modified.pth',map_location=torch.device('cpu'))["state_dict"])

class DSMI(nn.Module):
    def __init__(self):
        
        super(DSMI, self).__init__()
        self.model_new=model_new 
        self.encoder=self.model_new.downs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        for down in self.encoder:
            x = down(x)
            x = self.pool(x)
        
        # for down in self.downs:
        #     y = down(y)
        #     y = self.pool(y)
          
        # for down in self.downs:
        #     z = down(z)
        #     z = self.pool(z)
        # print(type(x))
        return x

import torch
import torchvision
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="unet_model_modified.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_fmaskdir,
    train_bmaskdir,
    batch_size,
    train_transform,
    mask_transform,
    # val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Dataset_(
        image_dir=train_dir,
        fmask_dir=train_fmaskdir,
        bmask_dir=train_bmaskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # val_ds = Dataset(
    #     image_dir=val_dir,
    #     mask_dir=val_maskdir,
    #     transform=val_transform,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    return train_ds

import torch
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Data/Images"
TRAIN_FMASK_DIR = "Data/foreground2"
TRAIN_BMASK_DIR = "Data/background2"
train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

dataset=get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_FMASK_DIR,
        TRAIN_BMASK_DIR,
        # VAL_IMG_DIR,
        # VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        # val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )



"""#FASTER RCNN with UNET

##Import required packages and libraries
"""

import torch
import torchvision
import pandas as pd
import os
torch.manual_seed(0)
from sklearn.model_selection import train_test_split
import torch.nn as nn
import sklearn
from matplotlib import patches
from PIL import Image, ImageDraw
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils
import warnings
from sys import maxunicode
from tqdm import tqdm
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms

"""##Dataloader

"""


class Dataset_2(torch.utils.data.Dataset):
    def __init__(self, root,image_dir, fmask_dir,bmask_dir,folder='dataset',transforms=None, transform=None):
        self.transforms=[]
        if transforms!=None:
            self.transforms.append(transforms)
        self.image_dir = image_dir
        self.fore_mask_dir = fmask_dir
        self.back_mask_dir = bmask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.mask1_dir= os.listdir(fmask_dir)
        self.mask2_dir = os.listdir(bmask_dir)
        self.root=root
        self.folder=folder
        box_data=pd.read_csv("Data/DSMI.csv")
        self.box_data=pd.concat([box_data,box_data.bbox.str.split('[').str.get(1).str.split(']').str.get(0).str.split(',',expand=True)],axis=1)
        # print(self.box_data)
        self.imgs=list(os.listdir(os.path.join(root,self.folder)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path=os.path.join(os.path.join(self.root,self.folder),self.imgs[index])
        fore_mask_path = os.path.join(self.fore_mask_dir,self.imgs[index])
        back_mask_path = os.path.join(self.back_mask_dir,self.imgs[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        image2=image
        fore_mask = np.array(Image.open(fore_mask_path).convert("RGB"))
        # fore_mask[fore_mask == 255.0] = 1.0
        back_mask = np.array(Image.open(back_mask_path).convert("RGB"))
        # back_mask[back_mask == 255.0] = 1.0
        

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=fore_mask)
          
            image = augmentations["image"]
            fore_mask = augmentations["mask"]
            augmentations2 = self.transform(image=image2, mask=back_mask)
           
            back_mask = augmentations2["mask"]
            
        
        
        img=Image.open(img_path)
        df=self.box_data[self.box_data['image_id']==self.imgs[index].split('.')[0]]
       
        if df.shape[0]!=0:
        #   df[2]=df[0].astype(float)+df[2].astype(float)
        #   df[3]=df[1].astype(float)+df[3].astype(float)
          boxes=df[[0,1,2,3]].astype(float).values
          if self.folder=="Modifiedpng/Train/benign/":
              labels=np.ones(len(boxes))
          elif self.folder=="Modifiedpng/Train/malignant/":
              labels=np.full(len(boxes),0)
          else:    
              labels=np.full(len(boxes),0)
        else:
            boxes=np.asarray([[0,0,0,0]])
            labels=np.full(len(boxes),0)
        for i in self.transforms:
            img=i(img)

        targets={}
        targets['boxes']=torch.from_numpy(boxes).double()
        targets['labels']=torch.from_numpy(labels).type(torch.int64)

        
        return img.double(),targets,img,fore_mask,back_mask

class Dataset_2test(torch.utils.data.Dataset):
    def __init__(self, root,image_dir, fmask_dir,bmask_dir,folder='dataset',transforms=None, transform=None):
        self.transforms=[]
        if transforms!=None:
            self.transforms.append(transforms)
        self.image_dir = image_dir
        self.fore_mask_dir = fmask_dir
        self.back_mask_dir = bmask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.mask1_dir= os.listdir(fmask_dir)
        self.mask2_dir = os.listdir(bmask_dir)
        self.root=root
        self.folder=folder
        box_data=pd.read_csv("Data/DSMI.csv")
        self.box_data=pd.concat([box_data,box_data.bbox.str.split('[').str.get(1).str.split(']').str.get(0).str.split(',',expand=True)],axis=1)
        # print(self.box_data)
        self.imgs=list(os.listdir(os.path.join(root,self.folder)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path=os.path.join(os.path.join(self.root,self.folder),self.imgs[index])
        fore_mask_path = os.path.join(self.fore_mask_dir,self.imgs[index])
        back_mask_path = os.path.join(self.back_mask_dir,self.imgs[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        image2=image
        fore_mask = np.array(Image.open(fore_mask_path).convert("RGB"))
        # fore_mask[fore_mask == 255.0] = 1.0
        back_mask = np.array(Image.open(back_mask_path).convert("RGB"))
        # back_mask[back_mask == 255.0] = 1.0
        

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=fore_mask)
          
            image = augmentations["image"]
            fore_mask = augmentations["mask"]
            augmentations2 = self.transform(image=image2, mask=back_mask)
           
            back_mask = augmentations2["mask"]
            
        
        
        img=Image.open(img_path)
        df=self.box_data[self.box_data['image_id']==self.imgs[index].split('.')[0]]
       
        if df.shape[0]!=0:
        #   df[2]=df[0].astype(float)+df[2].astype(float)
        #   df[3]=df[1].astype(float)+df[3].astype(float)
          boxes=df[[0,1,2,3]].astype(float).values
          if self.folder=="Modifiedpng/Test/benign/":
              labels=np.ones(len(boxes))
          elif self.folder=="Modifiedpng/Test/malignant/":
              labels=np.full(len(boxes),0)
          else:    
              labels=np.full(len(boxes),0)
        else:
            boxes=np.asarray([[0,0,0,0]])
            labels=np.full(len(boxes),0)
        for i in self.transforms:
            img=i(img)

        targets={}
        targets['boxes']=torch.from_numpy(boxes).double()
        targets['labels']=torch.from_numpy(labels).type(torch.int64)

        
        return img.double(),targets,img,fore_mask,back_mask

"""##Prepare Train set and Test set """

root=''


train_transforms = transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.0,p=1.0),
    transforms.RandomEqualize(p=1.0),
    transforms.ToTensor()
])

dataset=Dataset_2(root,TRAIN_IMG_DIR, TRAIN_FMASK_DIR,TRAIN_BMASK_DIR,'Modifiedpng/Train/benign/',transforms=train_transforms,transform=train_transform)

image,target,img,fmask,bmask =dataset[2]

root=''


dataset_benign_train=Dataset_2(root,TRAIN_IMG_DIR, TRAIN_FMASK_DIR,TRAIN_BMASK_DIR,'Modifiedpng/Train/benign/',transforms=train_transforms,transform=train_transform)
dataset_benign_test=Dataset_2test(root,TRAIN_IMG_DIR, TRAIN_FMASK_DIR,TRAIN_BMASK_DIR,'Modifiedpng/Test/benign/',transforms=train_transforms,transform=train_transform)
image,target,img,fmask,bmask = dataset_benign_train.__getitem__(1)
print(image.shape)
plt.imshow(image.permute(1, 2, 0))


dataset_malignant_train=Dataset_2(root,TRAIN_IMG_DIR, TRAIN_FMASK_DIR,TRAIN_BMASK_DIR,'Modifiedpng/Train/malignant/',transforms=train_transforms,transform=train_transform)
dataset_malignant_test=Dataset_2test(root,TRAIN_IMG_DIR, TRAIN_FMASK_DIR,TRAIN_BMASK_DIR,'Modifiedpng/Test/malignant/',transforms=train_transforms,transform=train_transform)

benign_tens=[]
malignant_tens=[]

for i in range(len(dataset_benign_train)):
    benign_tens.append(list(dataset_benign_train[i]))

for i in range(len(dataset_malignant_train)):
    malignant_tens.append(list(dataset_malignant_train[i]))

print(len(benign_tens),len(malignant_tens))
dataset_train=[]
dataset_train.extend(benign_tens)
dataset_train.extend(malignant_tens)
print(len(dataset_train))

benign_tens=[]
malignant_tens=[]

for i in range(len(dataset_benign_test)):
    benign_tens.append(list(dataset_benign_test[i]))

for i in range(len(dataset_malignant_test)):
    malignant_tens.append(list(dataset_malignant_test[i]))

print(len(benign_tens),len(malignant_tens))
dataset_test=[]
dataset_test.extend(benign_tens)
# dataset_test.extend(malignant_tens)
print(len(dataset_test))

data_loader_train = torch.utils.data.DataLoader(dataset_train,batch_size=1,shuffle=True,collate_fn=lambda x:list(zip(*x)))
data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,collate_fn=lambda x:list(zip(*x)))

# images,labels=next(iter(data_loader_train))

"""##Function to view and save results"""

images,labels,img,fmask,bmask=next(iter(data_loader_train))

# images,labels=next(iter(data_loader_train))
def view(images,labels,k,string,std=1, mean=0):
    figure = plt.figure(figsize=(30,30))
    images=list(images)
    labels=list(labels)
    if(len(labels) == 0): return
    for i in range(k):
        out=torchvision.utils.make_grid(images[i])
        inp=out.cpu().numpy().transpose(1,2,0)
        inp=np.array(std)*inp+np.array(mean)
        inp=np.clip(inp,0,1)
        ax=figure.add_subplot(2,2,i+1)
        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
        l=labels[i]['boxes'].cpu().numpy()

         # Load the image tensor
        img_tensor =images[i]

        # Convert the tensor to a PIL image
        img_pil = Image.fromarray((img_tensor * 255).permute(1, 2, 0).byte().cpu().numpy(), mode='RGB')

        # Create a new PIL image for drawing the bounding box
        draw = ImageDraw.Draw(img_pil)
     
        
        # l[:,2]=l[:,2]-l[:,0]
        # l[:,3]=l[:,3]-l[:,1]
        l2=l
        for j in range(len(l)):
            ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=2,edgecolor='b',facecolor='none'))
            ax.set_xlabel(labels[i]['labels'][0])

            # Define the coordinates of the bounding box
            x1, y1, x2, y2 = l2[j][0],l2[j][1],l2[j][2],l2[j][3]
            # x1, y1, x2, y2 = l2[j][0],l2[j][1],l2[j][2]+l2[j][0],l2[j][3]+l2[j][1]
            # Draw the bounding box on the image
            if x1<x2 and y1<y2:
                draw.rectangle((x1, y1, x2, y2), outline='red')
            else: return
               

        # Save the image with the bounding box
        img_pil.save(f"{string}.png")
        
        # Load the image with the bounding box
        # img = plt.imread(f"/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn/MYimage.png")

        # # Show the image with the bounding box
        # plt.imshow(img)
        # plt.axis('off')
        # plt.show()
        # figure.savefig(f"/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn/Test_results_5Mar/Pretrained_and_Finetuned/Malignant/Predictions/{string}.png")
# view(images,labels,1)

def nms(boxes, scores, iou_threshold):
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()

    # to(): Performs Tensor dtype and/or device conversion
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
   
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    keep = keep.nonzero().squeeze(1)
    return keep


def clip_boxes_to_image(boxes, size):
   
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # x1, x2
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):

    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def permute_and_flatten(layer, N, A, C, H, W):
    
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    
    box_cls_flattened = []
    box_regression_flattened = []

    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # [batch_size, anchors_num_per_position * classes_num, height, width], class_num is equal 2
        N, AxC, H, W = box_cls_per_level.shape
        # [batch_size, anchors_num_per_position * 4, height, width]
        Ax4 = box_regression_per_level.shape[1]
        # anchors_num_per_position
        A = Ax4 // 4
        # classes_num
        C = AxC // A

        # [N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # [N, -1, C]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class BalancedPositiveNegativeSampler(object):
    

    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        

        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            # positive sample if index >= 1
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            # negative sample if index == 0
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            # number of positive samples
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples, used all positive samples
            num_pos = min(positive.numel(), num_pos)

            # number of negative samples
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples, used all negative samples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


def encode_boxes(reference_boxes, proposals, weights):
    

    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # Returns a new tensor with a dimension of size one inserted at the specified position.
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1

    # center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        

        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
       

        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
       

        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)

        boxes_per_image = [b.size(0) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        # map regression parameters into anchors to get coordinate
        pred_boxes = self.decode_single(
            rel_codes.reshape(box_sum, -1), concat_boxes
        )
        return pred_boxes.reshape(box_sum, -1, 4)

    def decode_single(self, rel_codes, boxes):
        
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor width
        heights = boxes[:, 3] - boxes[:, 1]  # anchor height
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor center x coordinate
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor center y coordinate

        wx, wy, ww, wh = self.weights  # default is 1
        dx = rel_codes[:, 0::4] / wx   # predicated anchors center x regression parameters
        dy = rel_codes[:, 1::4] / wy   # predicated anchors center y regression parameters
        dw = rel_codes[:, 2::4] / ww   # predicated anchors width regression parameters
        dh = rel_codes[:, 3::4] / wh   # predicated anchors height regression parameters

        # limit max value, prevent sending too large values into torch.exp()
        # self.bbox_xform_clip=math.log(1000. / 16)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes


def set_low_quality_matches_(matches, all_matches, match_quality_matrix):
    
    # For each gt, find the prediction with which it has highest quality
    highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)  # the dimension to reduce.

    # Find highest quality match available, even if it is low, including ties
    gt_pred_pairs_of_highest_quality = torch.nonzero(
        match_quality_matrix == highest_quality_foreach_gt[:, None]
    )
    

    pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
    matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


class Matcher(object):
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        

        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
   
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
import torch.nn.functional as F
from torch import Tensor
from torch.jit.annotations import List, Dict, Tuple

# import utils.boxes_utils as box_op
# from utils.det_utils import *


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    
    # print(labels)
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

   
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = smooth_l1_loss(box_regression[sampled_pos_inds_subset, labels_pos],
                              regression_targets[sampled_pos_inds_subset],
                              beta=1 / 9,
                              size_average=False,
                              ) / labels.numel()

    return classification_loss, box_loss


def add_gt_proposals(proposals, gt_boxes):
    

    proposals = [
        torch.cat((proposal, gt_box))
        for proposal, gt_box in zip(proposals, gt_boxes)
    ]
    return proposals


def check_targets(targets):
    assert targets is not None
    assert all(["boxes" in t for t in targets])
    assert all(["labels" in t for t in targets])


class RoIHeads(torch.nn.Module):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,

                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,

                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detection_per_img):
        super(RoIHeads, self).__init__()

        self.box_similarity = box_iou

        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 0.5
            bg_iou_thresh,  # 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,  # 512
            positive_fraction)  # 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        

        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                # iou of bbox and anchors
                match_quality_matrix = box_iou(gt_boxes_in_image, proposals_in_image)

                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def select_training_samples(self,
                                proposals,
                                targets
                                ):

        check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        proposals = add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,
                               box_regression,
                               proposals,
                               image_shapes
                               ):
        
        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            boxes = clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            keep = remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            keep = batched_nms(boxes, scores, labels, self.nms_thresh)

            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None
                ):
        

        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                # assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses
class TwoMLPHead(nn.Module):
   
    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        # x = x.to(torch.float)
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

class RPNHead(nn.Module):
    

    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()
        # 3x3 conv
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # background/foreground score
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        # bbox regression parameters
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        cls_scores = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            cls_scores.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return cls_scores, bbox_reg

from torch import nn
from torch.jit.annotations import List, Tuple


@torch.jit.script
class ImageList(object):
    
    def __init__(self, tensors, image_sizes):
        
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)
    
def torch_choice(l):
    index = int(torch.empty(1).uniform_(0., float(len(l))).item())
    return l[index]


def max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_images(images, size_divisible=32):
    """
    batched images
    :param images: a set of images
    :param size_divisible: ratio of height/width to be adjusted
    :return: batched tensor image
    """

    max_size = max_by_axis([list(img.shape) for img in images])

    stride = float(size_divisible)

    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    # [batch, channel, height, width]
    batch_shape = [len(images)] + max_size

    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN model.
    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    :param min_size: minimum size of input image
    :param max_size: maximum size of input image
    :param image_mean: image mean
    :param image_std: image std
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        """
        resize input image to specified size and transform for target
        :param image: input image
        :param target: target related info, like bbox
        :return:
            image: resized image
            target: resized target
        """

        # image shape is [channel, height, width]
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        if self.training:
            size = float(torch_choice(self.min_size))
        else:
            size = float(self.min_size[-1])
        scale_factor = size / min_size

        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size

        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def postprocess(self, result, image_shapes, original_image_sizes):
        """
        post process of predictions, mainly map bboxed coordinates to original image
        :param result: predictions result
        :param image_shapes: image size after preprocess
        :param original_image_sizes: original image size
        :return:
        """

        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # save resized image size
        image_sizes = [img.shape[-2:] for img in images]
        images = batch_images(images)
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


def resize_boxes(boxes, original_size, new_size):
    
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

import torch
from torch import nn


def generate_anchors(scales, aspect_ratios, dtype=torch.float32, device="cpu"):
    """
     generate anchor template based on sizes and ratios, generated template is centered at [0, 0]
     :param scales: anchor sizes, in tuple[int]
     :param aspect_ratios: anchor ratios, in tuple[float]
     :param dtype: data type
     :param device: date device
     :return:
     """

    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1.0 / h_ratios

    # [r1, r2, r3]' * [s1, s2, s3]
    # number of elements is len(ratios)*len(scales)
    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    # left-top, right-bottom coordinate relative to anchor center(0, 0)
    # anchor template is centered at [0, 0], shape [len(ratios)*len(scales), 4]
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    return base_anchors.round()  # anchor will lose some precision here


class AnchorsGenerator(nn.Module):
    """
    anchor generator for feature maps according to anchor sizes and ratios
    :param sizes: anchor sizes, in tuple[int]
    :param aspect_ratios: anchor ratios, in tuple[float]
    :return:
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()

        # assert len(sizes) == len(aspect_ratios), 'anchor sizes must equal to anchor ratios!'

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def set_cell_anchors(self, dtype, device):
        """
        generate template template
        :param dtype: data type
        :param device: data device
        :return:
        """
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None

        # generate anchor template
        cell_anchors = [generate_anchors(sizes, aspect_ratios, dtype, device)
                        for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # calculate the number of anchors per feature map, for k in origin paper
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, feature_map_sizes, strides):
        """
        compute anchor coordinate list in origin image, mapped from feature map
        :param feature_map_sizes: feature map sizes
        :param strides: strides between origin image and anchor
        :return:
        """

        anchors = []
        cell_anchors = self.cell_anchors  # anchor template
        assert cell_anchors is not None

        # for every resolution feature map, like fpn
        for size, stride, base_anchors in zip(feature_map_sizes, strides, cell_anchors):
            f_p_height, f_p_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center...]
            # x_center in origin image
            shifts_x = torch.arange(0, f_p_width, dtype=torch.float32, device=device) * stride_width

            # y_center in origin image
            shifts_y = torch.arange(0, f_p_height, dtype=torch.float32, device=device) * stride_height

            # torch.meshgrid will output grid
            # shape: [grid_height, grid_width]
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors  # List[Tensor(all_num_anchors, 4)]

    def cached_grid_anchors(self, feature_map_size, strides):
        """
        cached all anchor information
        :param feature_map_size: feature map size after backbone feature extractor
        :param strides: strides between origin image size and feature map size
        :return:
        """

        key = str(feature_map_size) + str(strides)
        # self._cache is a dictionary type
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(feature_map_size, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list, feature_maps):
        """
        get feature map sizes
        :param image_list:
        :param feature_maps:
        :return:
        """

        feature_map_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # get input image sizes
        image_size = image_list.tensors.shape[-2:]

        # get dtype and device
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # compute map stride between feature_maps and input images
        strides = [[torch.tensor(image_size[0] / g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] / g[1], dtype=torch.int64, device=device)] for g in feature_map_sizes]

        # get anchors template according size and aspect_ratios
        self.set_cell_anchors(dtype, device)

        # get anchor coordinate list in origin image, according to map
        anchors_over_all_feature_maps = self.cached_grid_anchors(feature_map_sizes, strides)

        anchors = []
        # for every image and feature map in a batch
        for i, (_, _) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # for every resolution feature map like fpn
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)

        # concat every resolution anchors, like fpn
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]

        self._cache.clear()
        return anchors

from torch import nn
from torch.jit.annotations import Dict
from torch.nn import functional as F



class RPNHead(nn.Module):
    

    def __init__(self, in_channels, num_anchors):

        super(RPNHead, self).__init__()
        # 3x3 conv
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        # background/foreground score
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)

        # bbox regression parameters
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        cls_scores = []
        bbox_reg = []
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            cls_scores.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return cls_scores, bbox_reg


class RegionProposalNetwork(torch.nn.Module):
    

    def __init__(self, anchor_generator, head, fg_iou_thresh, bg_iou_thresh, batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh):

        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # function for computing iou between anchor and true bbox
        self.box_similarity = box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # foreground threshold, if IOU > threshold(0.7), is positive samples
            bg_iou_thresh,  # background threshold, if IOU < threshold(0.3), is negative samples
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1e-3

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
       

        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                # compute iou of anchors and real bbox
                match_quality_matrix =box_iou(gt_boxes, anchors_per_image)
                # calculate index of anchors and gt iou（iou<0.3 is -1，0.3<iou<0.7 is -2）
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        

        result = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            result.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(result, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        
        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size, size filled with fill_value
        levels = [torch.full((n,), idx, dtype=torch.int64, device=device)
                  for idx, n in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]  # [batch_size, 1]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, lvl, img_shape in zip(proposals, objectness, levels, image_shapes):
            # adjust predicted bbox, make boxes outside of the image in image
            boxes = clip_boxes_to_image(boxes, img_shape)

            # Remove boxes which contains at least one side smaller than min_size.
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            # keep only top k scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
       

        # selective positive and negative samples
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # bbox regression loss
        box_loss = smooth_l1_loss(pred_bbox_deltas[sampled_pos_inds], regression_targets[sampled_pos_inds],
                                  beta=1 / 9, size_average=False, ) / (sampled_inds.numel())

        # classification loss
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(self, images, features, targets=None):
        

        # RPN uses all feature maps that are available
        features = list(features.values())

        # Two fc layers to compute the fg/bg scores and bboxs regressions
        fg_bg_scores, pred_bbox_deltas = self.head(features)

        # get all anchors of images based on features
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # numel() Returns the total number of elements in the input tensor.
        num_anchors_per_level_shape_tensors = [o[0].shape for o in fg_bg_scores]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # adjust tensor order and reshape
        fg_bg_scores, pred_bbox_deltas = concat_box_prediction_layers(fg_bg_scores, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        # remove small bboxes, nms process, get post_nms_top_n target
        boxes, scores = self.filter_proposals(proposals, fg_bg_scores, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)

            # encode parameters based on the bboxes and anchors
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                fg_bg_scores, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg}

        return boxes, losses
from collections import OrderedDict
"""##Training and Validation"""
class DSMI_(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.Unet = UNET()
        self.m = nn.Conv2d(512,256,kernel_size=2,padding=5)
       
    def forward(self, x,y,z):
        _,I_Embed=self.Unet(x[0].unsqueeze(0))
        _,F_Embed=self.Unet(y[0].permute(2,0,1).unsqueeze(0))
        _,B_Embed=self.Unet(z[0].permute(2,0,1).unsqueeze(0))
        FB_Embed=torch.cat((F_Embed, B_Embed), 1)
        I_Embed_T=torch.transpose(I_Embed,1,2)
        I_Embed_T2=torch.transpose(I_Embed,1,3)
        tensor2 = I_Embed_T2
        tensor1 = I_Embed_T
        tens=torch.einsum('bikj,bijk->bij',tensor1 , tensor2)
        tensor1 = tens
        tensor2 = FB_Embed
        tens2=torch.einsum('bij,bmij->bmij',tensor1 , tensor2)    
        output=self.m(tens2)
        return output
    
class FasterRCNNBase(nn.Module):
    

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        # self.fpn = fpn
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.counter = 0
        self.dsmi = DSMI_()
        self.conv = nn.Conv2d(512,256,kernel_size=1)
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None,im=None,fm=None,bm=None):
        plt.clf()
        
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features_np = images.tensors.cpu().detach().numpy()
        fig, axes = plt.subplots(nrows=1, ncols=1)
        for i in range(min(features_np.shape[1], 1)):
            ax = np.ravel(axes)[i]
            ax.imshow(features_np[0, i, :, :], cmap='gray')
        fig.savefig(f"i{self.counter}.png")
        features = self.backbone(images.tensors)
        x1 = features['3']
        if im is not None:
            x2 = self.dsmi(im,fm,bm)
            x3 = torch.cat((x1,x2),1)
            x4 = self.conv(x3)
            backbone_output = {'0': features['0'], '1': features['1'] , '2': features['2'], '3': x4.double()}
            backbone_output = collections.OrderedDict(backbone_output) 
            features = backbone_output 
        else:
            x4 = x1
        # plt.clf()
        # features_np = x2.cpu().detach().numpy()
        # fig, axes = plt.subplots(nrows=1, ncols=1)
        # for i in range(min(features_np.shape[1], 1)):
        #     ax = np.ravel(axes)[i]
        #     ax.imshow(features_np[0, i, :, :], cmap='gray')
        # fig.savefig(f"di{self.counter}.png")
        # plt.clf()
        # features_np = x1.cpu().detach().numpy()
        # fig, axes = plt.subplots(nrows=1, ncols=1)
        # for i in range(min(features_np.shape[1], 1)):
        #     ax = np.ravel(axes)[i]
        #     ax.imshow(features_np[0, i, :, :], cmap='gray')
        # fig.savefig(f"f{self.counter}.png")
        # x2 = 0.5*x2
        
        # plt.clf()
        # features_np = x2.cpu().detach().numpy()
        # fig, axes = plt.subplots(nrows=1, ncols=1)
        # for i in range(min(features_np.shape[1], 1)):
        #     ax = np.ravel(axes)[i]
        #     ax.imshow(features_np[0, i, :, :], cmap='gray')
        # fig.savefig(f"decdsmi{self.counter}.png")
        
        
        # features = self.fpn(backbone_output)
        x11 = features['3']
        # print(x11.shape)
        # plt.clf()
        # features_np = x11.cpu().detach().numpy()
        # fig, axes = plt.subplots(nrows=1, ncols=1)
        # for i in range(min(features_np.shape[1], 1)):
        #     ax = np.ravel(axes)[i]
        #     ax.imshow(features_np[0, i, :, :], cmap='gray')
        # fig.savefig(f"a{self.counter}.png")

        # if isinstance(features, torch.Tensor):
        #     features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        # print(proposals)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # print(detections)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        # string = ("detections"+str(self.counter))
       
        self.counter=self.counter+1
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)
    
from torchvision.ops import MultiScaleRoIAlign
class FastRCNNPredictor(nn.Module):
   

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
    
class FasterRCNN(FasterRCNNBase):

    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=300, max_size=800,  # preprocess minimum and maximum size
                 image_mean=None, image_std=None,  # mean and std in preprocess

                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # kept proposals before nms
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # kept proposals after nms
                 rpn_nms_thresh=0.7,  # iou threshold during nms
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # bg/fg threshold
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # number of samples and fraction

                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,

                 # remove low threshold target
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None
                 ):

     

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # output channels of the backbone
        out_channels = 256

        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # two fc layer after roi pooling
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # get prediction
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        # fpn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).backbone.fpn
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)


image,target,img,fmask,bmask =dataset[2]
import time
from torchvision import ops          

def create_model(num_classes):
    global backbone, model
    
    anchor_sizes = tuple((f,) for f in  [32, 64, 128, 256, 512] )
    aspect_ratios = tuple((f,) for f in [0.25, 0.5, 1, 2.0, 4.0]) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)
  
    backbone = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).backbone

    roi_pooler = ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=[7, 7],
        sampling_ratio=2)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                        # transform parameters
                        min_size=800, max_size=1000,
                        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
                        # rpn parameters
                        rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                        rpn_pre_nms_top_n_train= 2000,
                        rpn_pre_nms_top_n_test=2000,
                        rpn_post_nms_top_n_train=1000,
                        rpn_post_nms_top_n_test=1000,
                        rpn_nms_thresh=0.7,
                        rpn_fg_iou_thresh=0.7,
                        rpn_bg_iou_thresh=0.3,
                        rpn_batch_size_per_image=256,
                        rpn_positive_fraction=0.5,
                        # Box parameters
                        box_head=None, box_predictor=None,

                        # remove low threshold target
                        box_score_thresh=0.05,
                        box_nms_thresh=0.5,
                        box_detections_per_img=100,
                        box_fg_iou_thresh=0.5,
                        box_bg_iou_thresh=0.5,
                        box_batch_size_per_image=512,
                        box_positive_fraction=0.25,
                        bbox_reg_weights=None
                        )

       
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

#We'll take pretrained Faster R-CNN model from torchvision and change the roi head according to our needs

# print(FastRCNNPredictor)
num_classes = 2
# model=FasterRCNNwithDSMI(num_classes)
model = create_model(num_classes)

# model=model._modules["backbone"]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

detection_threshold=0.5
rpn_pre_nms_top_n_train=2000
rpn_pre_nms_top_n_test=1000
rpn_nms_thresh=0.7

# print(model)
model=model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params,lr=1e-4,weight_decay=1e-4)


class Averager:
    def __init__(self):
        self.current_total=0.0
        self.iterations = 0.0
    def send(self,value):
        self.current_total+=value
        self.iterations+=1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0*self.current_total/self.iterations
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# model.train()
# import time
# itr=1
# import logging
# start_time=time.time()
# total_train_loss=[]
# loss_hist=Averager()
# epoch_loss_val=[]
# from tqdm.notebook import tqdm
# # set up checkpointing parameters
# checkpoint_path = 'model_checkpoint_dsmi_scratch2.pth'
# checkpoint_frequency = 5

# # initialize early stopping parameters
# early_stop = False
# best_val_loss = float('inf')
# epochs_no_improve = 0
# epoch_train_loss=0
# patience=15
# num_epochs=1000
# test_size=0.2
# startepoch=0
# train_data, val_data = train_test_split(dataset_train, test_size=test_size, random_state=42)
# # print(len(dataset_test))
# data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=lambda x:list(zip(*x)))
# data_loader_val = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=lambda x:list(zip(*x)))
# # Check if the checkpoint file exists
# if os.path.exists(checkpoint_path):
#   # Load the checkpoint
#   checkpoint = torch.load(checkpoint_path)
#   model.load_state_dict(checkpoint['model_state_dict'])
#   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#   startepoch = checkpoint['epoch']
#   epoch_train_loss = checkpoint['loss']
#   print('Checkpoint found and loaded successfully.')
# else:
#   startepoch = 0
#   print('No checkpoint found, starting training from scratch.')
# for epoch in tqdm(range(startepoch,num_epochs)):
#     model.train()
#     print("epoch number:",epoch)
#     loss_hist.reset()
#     train_loss = []
#     val_loss =[]
#     for images,targets,imgs,fmasks,bmasks in tqdm(data_loader_train):
        
#         images = list(image.type(torch.cuda.DoubleTensor) for image in images)
#         targets = [{k:v.type(torch.cuda.DoubleTensor) for k,v in t.items()} for t in targets]
#         imgs = list(img.type(torch.cuda.DoubleTensor) for img in imgs)
#         fmasks = list(fmask.type(torch.cuda.DoubleTensor) for fmask in fmasks)
#         bmasks = list(bmask.type(torch.cuda.DoubleTensor) for bmask in bmasks)
#         box=targets[0]["boxes"][0]
#         if box[0] >= box[2] or box[1] >= box[3]:
#            print('')
#         else:

#            model = model.double()

#            loss_dict = model(images,targets,imgs,fmasks,bmasks)
#            losses = sum(loss for loss in loss_dict.values())
#            loss_value = losses.item()

#            loss_hist.send(loss_value)

#            optimizer.zero_grad()
#            losses.backward()
#            optimizer.step()
#            train_loss.append(loss_value)
                
#         if itr % 25 == 0:
#           print(f"\n Iteration #{itr}  loss:{loss_dict}\n")
#         #print(f"\n Iteration #{itr}  loss:{loss_dict}\n")
#         itr += 1
#         logging.basicConfig(filename='dsmi_scratch.log', level=logging.DEBUG)
#         logger = logging.getLogger()
#         #logging.debug('Epoch: {},  Loss: {}'.format(epoch, losses.item()))
    
#     #lr_scheduler.step()    
          
   
#     # evaluate the model on the validation set
    
#     with torch.no_grad():
#         for images,targets,imgs,fmasks,bmasks in data_loader_val:
#             images = list(image.type(torch.cuda.DoubleTensor) for image in images)
#             targets = [{k:v.type(torch.cuda.DoubleTensor) for k,v in t.items()} for t in targets]
#             imgs = list(img.type(torch.cuda.DoubleTensor) for img in imgs)
#             fmasks = list(fmask.type(torch.cuda.DoubleTensor) for fmask in fmasks)
#             bmasks = list(bmask.type(torch.cuda.DoubleTensor) for bmask in bmasks)
#             model = model.double()
#             loss_dict= model.forward(images,targets,imgs,fmasks,bmasks)
            
#             losses = sum(loss for loss in loss_dict.values())
            
#             val_loss.append(loss_value)
        
#     epoch_train_loss = np.mean(train_loss)
#     epoch_val_loss = np.mean(val_loss)
#     total_train_loss.append(epoch_train_loss)
#     print(f'Epoch train loss is {epoch_train_loss:.4f}')
#     print(f'Epoch val loss is {epoch_val_loss:.4f}')

#     # check for improvement in validation loss
#     if epoch_val_loss < best_val_loss:
#         best_val_loss = epoch_val_loss
#         epochs_no_improve = 0
#         torch.save({
#              'epoch': epoch,
#              'model_state_dict': model.state_dict(),
#              'optimizer_state_dict': optimizer.state_dict(),
#              'loss': epoch_train_loss,
#           },'dsmi_scratch.pth')
#         checkpoint = {
#              'epoch': epoch,
#              'model_state_dict': model.state_dict(),
#              'optimizer_state_dict': optimizer.state_dict(),
#              'loss': epoch_train_loss,
#           }
#         torch.save(checkpoint, checkpoint_path)
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve == patience:
#             print(f'Early stopping after {epoch+1} epochs')
#             early_stop = True
#             break

#     epoch_loss_val.append(epoch_train_loss)
#     time_elapsed = time.time() - start_time
#     print("Time elapsed: ",time_elapsed)
#     logging.debug('Epoch: {}, epoch_train_loss: {}, time:{}'.format(epoch, epoch_train_loss,time_elapsed))
   

# if not early_stop:
#         print("Training completed after all epochs")

# plt.clf()
# plt.plot(epoch_loss_val)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')

# # Save the plot as a PNG image
# plt.savefig('dsmi_scratch.png')
model=model.double()
mdl=torch.load('dsmi_scratch.pth',map_location=torch.device('cpu'))
model.load_state_dict(mdl["model_state_dict"])

"""##Testing and Evaluation"""

def non_max_suppression(predictions, confidence_threshold, iou_threshold):
   
    # Filter out predictions with low confidence scores
    filtered_predictions = predictions[predictions[:, 4] > confidence_threshold]

    # Sort predictions by confidence score (highest first)
    sorted_indices = torch.argsort(filtered_predictions[:, 4], descending=True)

    selected_predictions = []
    while len(sorted_indices) > 0:
        # Select the prediction with highest confidence score
        best_index = sorted_indices[0]
        # print(filtered_predictions[best_index])
        selected_predictions.append(filtered_predictions[best_index])

        # Remove the selected prediction from the list
        sorted_indices = sorted_indices[1:]

        # Calculate IoU (Intersection over Union) between the selected prediction and the remaining predictions
        ious = calculate_ious(filtered_predictions[best_index], filtered_predictions[sorted_indices])

        # Filter out predictions that overlap significantly with the selected prediction
        overlapping_indices = sorted_indices[ious > iou_threshold]
        sorted_indices = sorted_indices[ious <= iou_threshold]

    return selected_predictions

def calculate_ious(prediction, predictions):
   
    # Calculate the area of each bounding box
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    predictions_area = (predictions[:, 2] - predictions[:, 0]) * (predictions[:, 3] - predictions[:, 1])

    # Calculate the coordinates of the intersection rectangle
    x1 = torch.max(prediction[0], predictions[:, 0])
    y1 = torch.max(prediction[1], predictions[:, 1])
    x2 = torch.min(prediction[2], predictions[:, 2])
    y2 = torch.min(prediction[3], predictions[:, 3])

    # Calculate the area of the intersection rectangle
    intersection_area = torch.max(torch.tensor(0.0), x2 - x1) * torch.max(torch.tensor(0.0), y2 - y1)

    # Calculate the IoU between the prediction and each prediction in the list
    ious = intersection_area / (prediction_area + predictions_area - intersection_area)

    return ious

def box_accuracy(gt_boxes, pred_boxes, threshold=0.5):
    
    num_boxes = gt_boxes.size(0)
    iou = torch.zeros((num_boxes,), dtype=torch.float32)

    for i in range(num_boxes):
        gt_box = gt_boxes[i]
        if i>=pred_boxes.size(0) :
           break
        pred_box = pred_boxes[i]

        # calculate intersection-over-union (IoU)
        x1 = torch.max(gt_box[0], pred_box[0])
        y1 = torch.max(gt_box[1], pred_box[1])
        x2 = torch.min(gt_box[2], pred_box[2])
        y2 = torch.min(gt_box[3], pred_box[3])
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        union = area_gt + area_pred - intersection
        iou[i] = intersection / union

    # calculate accuracy as percentage of boxes with IoU > threshold
    correct_boxes = torch.sum(iou >= threshold)
    accuracy = correct_boxes.item() / num_boxes * 100
    
    return accuracy,correct_boxes,num_boxes

def box_precision_recall_mAP(gt_boxes, pred_boxes,threshold=0.5):
    
    num_boxes = gt_boxes.size(0)
    iou = torch.zeros((num_boxes,), dtype=torch.float32)
    sorted_idx = torch.argsort(-pred_boxes[:, 0])

    # Sort predicted boxes by confidence score
    sorted_pred_boxes = pred_boxes[sorted_idx]

    tp = torch.zeros((num_boxes,), dtype=torch.float32)
    fp = torch.zeros((num_boxes,), dtype=torch.float32)
    ttp=0
    ffp=0

    # Calculate true positives and false positives
    for i in range(num_boxes):
        gt_box = gt_boxes[i]
        if i>=pred_boxes.size(0) :
           break
        pred_box = sorted_pred_boxes[i]

        # Calculate IoU
        x1 = torch.max(gt_box[0], pred_box[0])
        y1 = torch.max(gt_box[1], pred_box[1])
        x2 = torch.min(gt_box[2], pred_box[2])
        y2 = torch.min(gt_box[3], pred_box[3])
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        union = area_gt + area_pred - intersection
        iou[i] = intersection / union

        if iou[i] >= threshold:
            ttp=ttp+1
            tp[i] = 1
        else:
          ffp=ffp+1
          fp[i] = 1

  
    # Calculate precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
  
    recall = tp_cumsum / num_boxes
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Calculate mean Average Precision (mAP)
    recall = torch.cat([torch.tensor([0]), recall, torch.tensor([1])])
    precision = torch.cat([torch.tensor([0]), precision, torch.tensor([0])])
    precision = torch.flip(precision, dims=[0])
    ap = torch.trapz(precision, recall)

    return precision, recall, ap.item(), tp_cumsum[-1], fp_cumsum[-1],num_boxes

TP=0
FP=0
FN=0
TN=0

total_correct=torch.zeros((1,),dtype=torch.float32)
total_boxes=torch.zeros((1,),dtype=torch.float32)
total_tp=torch.zeros((1,),dtype=torch.float32)
total_fp=torch.zeros((1,),dtype=torch.float32)
tbox=torch.zeros((1,),dtype=torch.float32)
detection_threshold=0.36
counter=0

for images,targets,imgs,fmasks,bmasks in tqdm(data_loader_test):
        images = list(image.type(torch.cuda.DoubleTensor).to(device) for image in images)
        targets = [{k: v.type(torch.cuda.DoubleTensor).to(device) for k, v in t.items()} for t in targets]    
        model.eval()
        output=model(images)
     
        new_output=output
        print(new_output)
        for j in range(len(new_output)):
          bbox=()
          score=()
          lbl=()
          predictions=()
                   
          boxes = new_output[j]['boxes']
          scores = new_output[j]['scores']
          labels= new_output[j]['labels']

          # Filter predictions with low score
          labels=labels[scores>=detection_threshold]
          boxes = boxes[scores >= detection_threshold]
          scores = scores[scores >= detection_threshold]
          new_output[j]['boxes']=boxes
          number_of_predictions=len(boxes)
          predictions=torch.randn(number_of_predictions,6)
          for i in range(len(new_output[j]['boxes'])):
            temp_tens=torch.tensor([new_output[j]['boxes'][i][0],new_output[j]['boxes'][i][1],new_output[j]['boxes'][i][2],new_output[j]['boxes'][i][3],new_output[j]['scores'][i],new_output[j]['labels'][i]])
            predictions[i]=(temp_tens)
          result=non_max_suppression(predictions,0,0.5)
          l=len(result)
          bbox=torch.randn(l,4)
          score=torch.randn(l)
          lbl=torch.randn(l)
          for i in range(len(result)):
            bb=torch.tensor([result[i][0],result[i][1],result[i][2],result[i][3]])
            bbox[i]=bb
            score[i]=torch.tensor(result[i][4])
            lbl[i]=torch.tensor(result[i][5])
          new_output[j]['scores']=score
          new_output[j]['boxes']=bbox
          new_output[j]['labels']=lbl
          #print(new_output)
          precision, recall, ap, tp_cumsum, fp_cumsum ,nbox= box_precision_recall_mAP(targets[j]['boxes'], new_output[j]['boxes'], threshold=0.0)
          total_tp=torch.add(total_tp,tp_cumsum)
          total_fp=torch.add(total_fp,fp_cumsum)
          tbox = torch.add(tbox,nbox)
          accuracy,correct_boxes,num_boxes = box_accuracy(targets[j]['boxes'], new_output[j]['boxes'], threshold=0.1)
          total_correct = torch.add(total_correct,correct_boxes)
          total_boxes = torch.add(total_boxes,num_boxes)

        string = ("output"+str(counter))
        counter=counter+1
        with torch.no_grad():
           view(images,new_output,1,string)

recall = total_tp / tbox
precision =  total_tp / (total_tp + total_fp)
accuracy = total_correct.item() / total_boxes * 100
recall = torch.cat([torch.tensor([0]), recall, torch.tensor([1])])
precision = torch.cat([torch.tensor([0]), precision, torch.tensor([0])])
precision = torch.flip(precision, dims=[0])
ap = torch.trapz(precision, recall)
print("\nAccuracy:",accuracy[0].item()," Recall: ",recall[1].item()," Precision:",precision[1].item()," mAP:",ap.item())
