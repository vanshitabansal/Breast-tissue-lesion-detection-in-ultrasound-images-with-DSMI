import torch
import torchvision
import pandas as pd
import os
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
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms

class lesiondataset(torch.utils.data.Dataset):
    def __init__(self,root,folder='dataset',transforms=None):
        self.transforms=[]
        if transforms!=None:
            self.transforms.append(transforms)
        self.root=root
        self.folder=folder
        box_data=pd.read_csv("dl_info_deeplesion.csv")
        self.box_data=pd.concat([box_data,box_data.bbox.str.split(',',expand=True)],axis=1)
        # print(self.box_data)
        # print(self.box_data)
        self.imgs=list(os.listdir(os.path.join(root,self.folder)))
        # print(list(os.listdir(os.path.join(root,self.folder))))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self,idx):
        img_path=os.path.join(os.path.join(self.root,self.folder),self.imgs[idx])
        img=Image.open(img_path)
        # print(img.size)
        # print(self.imgs[idx])
        df=self.box_data[self.box_data['image_id']==self.imgs[idx]]
        # print("df",df)
        if df.shape[0]!=0:
          # print("match")
          # print(df)
          boxes=df[[0,1,2,3]].astype(float).values
          # print(df[0]," ",df[1]," ",df[2]," ",df[3])
          # if self.folder=="Modifiedpng/benign/":
          labels=np.ones(len(boxes))
          # elif self.folder=="Modifiedpng/malignant/":
              # labels=np.full(len(boxes),2)
          # else:    
              # labels=np.full(len(boxes),0)
        else:
            boxes=np.asarray([[0,0,0,0]])
            labels=np.full(len(boxes),0)
        for i in self.transforms:
            img=i(img)

        targets={}
        targets['boxes']=torch.from_numpy(boxes).double()
        targets['labels']=torch.from_numpy(labels).type(torch.int64)

        return img.double(),targets


import torchvision.transforms as transforms

train_transforms = transforms.Compose([
    #transforms.RandomAutocontrast(p=0.2),
    #transforms.ColorJitter(brightness=0.7,contrast=0.5),
    #transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomAdjustSharpness(sharpness_factor=1.0,p=1.0),
    transforms.RandomEqualize(p=1.0),
    transforms.ToTensor()
])

# root='/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn'
# dataset_benign=lesiondataset(root,'Modifiedpng/benign/',transforms=torchvision.transforms.ToTensor())

#image,target = dataset.__getitem__(1)


#plt.imshow(image.permute(1, 2, 0))

root=''
dataset=lesiondataset(root,'DeepLesion/Images',transforms=train_transforms)
image,target = dataset.__getitem__(1)
print(image.shape)

plt.imshow(image.permute(1, 2, 0))

# dataset_benign[0]

print(len(dataset))

# root='/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn'

# dataset_malignant=lesiondataset(root,'Modifiedpng/malignant/',transforms=train_transforms)
# # (dataset_malignant[0])

# root='/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn'

# dataset_normal=lesiondataset(root,'Modifiedpng/normal/',transforms=train_transforms)
# # dataset_normal[0]

# print((dataset_normal[0][1]))
# print(type(dataset_benign[0][1]))
# print(len(dataset_benign))

lesion_tens=[]


for i in range(len(dataset)):
    lesion_tens.append(list(dataset[i]))


print(len(lesion_tens))
dataset=[]
# dataset.extend(benign_tens)
dataset.extend(lesion_tens)
# dataset.extend(normal_tens)
print(len(dataset))

"""#split improvement 80:20"""

torch.manual_seed(0)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
dataset_train, dataset_test = torch.utils.data.random_split(dataset, [train_size, test_size])

data_loader_train = torch.utils.data.DataLoader(dataset_train,batch_size=4,shuffle=True,collate_fn=lambda x:list(zip(*x)))
data_loader_test = torch.utils.data.DataLoader(dataset_test,batch_size=1,shuffle=False,collate_fn=lambda x:list(zip(*x)))
print(len(data_loader_test))

import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

images,labels=next(iter(data_loader_train))
from matplotlib import patches
def view(images,labels,k,std=1, mean=0):
    # print(labels)
    figure = plt.figure(figsize=(30,30))
    images=list(images)
    labels=list(labels)
    for i in range(k):
        out=torchvision.utils.make_grid(images[i])
        inp=out.cpu().numpy().transpose(1,2,0)
        inp=np.array(std)*inp+np.array(mean)
        inp=np.clip(inp,0,1)
        ax=figure.add_subplot(2,2,i+1)
        ax.imshow(images[i].cpu().numpy().transpose((1,2,0)))
        l=labels[i]['boxes'].cpu().numpy()
        
        print(l)
        print(len(l))
        l[:,2]=l[:,2]-l[:,0]
        l[:,3]=l[:,3]-l[:,1]
        for j in range(len(l)):
            ax.add_patch(patches.Rectangle((l[j][0],l[j][1]),l[j][2],l[j][3],linewidth=4,edgecolor='g',facecolor='none'))
            ax.set_xlabel(labels[i]['labels'][0])
        # figure.savefig(f"/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn/Test_results_5Mar/Pretrained_and_Finetuned/Malignant/Predictions/{string}.png")
#view(images,labels,4)

"""#Try different baseline"""

#We'll take pretrained Faster R-CNN model from torchvision and change the roi head according to our needs
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 2
detection_threshold=0.5
rpn_pre_nms_top_n_train=2000
rpn_pre_nms_top_n_test=1000
rpn_nms_thresh=0.7
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)
model=model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer=torch.optim.SGD(params,lr=1e-4)

"""#check losses
#print different losses
"""

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
# checkpoint_path = 'model_checkpoint_deeplesion_true.pth'
# checkpoint_frequency = 1

# # initialize early stopping parameters
# early_stop = False
# best_val_loss = float('inf')
# epochs_no_improve = 0
# epoch_train_loss=0
# patience=30
# num_epochs=200
# test_size=0.2
# startepoch=0
# train_data, val_data = train_test_split(dataset_train, test_size=test_size, random_state=42)
# data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=lambda x:list(zip(*x)))
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
#     for images,targets in tqdm(data_loader_train):
#         images = list(image.to(device) for image in images)
#         targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
#         box=targets[0]["boxes"][0]
#         if box[0] >= box[2] or box[1] >= box[3]:
#           print('')
#         elif len(targets)>1:
#           box=targets[1]["boxes"][0]
#           if box[0] >= box[2] or box[1] >= box[3]:
#             print('')
 
#           elif len(targets)>2:
#               box=targets[2]["boxes"][0]
#               if box[0] >= box[2] or box[1] >= box[3]:
#                 print('')

#               elif len(targets)>3:
#                    box=targets[3]["boxes"][0]
#                    if box[0] >= box[2] or box[1] >= box[3]:
#                      print('')

#                    else:

#                     model = model.double()
#                     loss_dict = model(images,targets)
                    
#                     losses = sum(loss for loss in loss_dict.values())
#                     loss_value = losses.item()

#                     loss_hist.send(loss_value)

#                     optimizer.zero_grad()
#                     losses.backward()
#                     optimizer.step()
#                     train_loss.append(loss_value)
                
#           if itr % 25 == 0:
#               print(f"\n Iteration #{itr} loss: {loss_dict} \n")

#           itr += 1
#           logging.basicConfig(filename='pretrained_deeplesion.log', level=logging.DEBUG)
#           logger = logging.getLogger()
#           #logging.debug('Epoch: {},  Loss: {}'.format(epoch, losses.item()))
    
#     #lr_scheduler.step()    
         
          
             
   
#     # evaluate the model on the validation set
    
#     with torch.no_grad():
#         for images, targets in data_loader_val:
#             images = list(image.to(device) for image in images)
#             targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
#             model = model.double()
#             loss_dict= model(images,targets)
            
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
#           },'pretrained_deeplesion.pth')
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
# plt.savefig('train_loss_epoch_pretrained_deeplesion.png')

# model.load_state_dict(torch.load('pretrained_deeplesion.pth',map_location=torch.device('cpu')))


# print("Training done!")

"""#Testing"""
model = model.double()
mdl=torch.load('pretrained_deeplesion.pth',map_location=torch.device('cpu'))
model.load_state_dict(mdl['model_state_dict'])


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


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#create_dir("/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn/Test_results_5Mar/Pretrained_and_Finetuned/Malignant/Predictions")
#create_dir("/content/drive/MyDrive/Implementation of Detectron2 based faster rcnn/Test_results_5Mar/Pretrained_and_Finetuned/Malignant/GroundTruth")


from sys import maxunicode
from tqdm import tqdm

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
for images,targets in tqdm(data_loader_test):
        images = list(image.type(torch.cuda.DoubleTensor).to(device) for image in images)
        targets = [{k: v.type(torch.cuda.DoubleTensor).to(device) for k, v in t.items()} for t in targets]    
        model.eval()
        output=model(images)
     
        new_output=output
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
  
          precision, recall, ap, tp_cumsum, fp_cumsum ,nbox= box_precision_recall_mAP(targets[j]['boxes'], new_output[j]['boxes'], threshold=0.5)
          total_tp=torch.add(total_tp,tp_cumsum)
          total_fp=torch.add(total_fp,fp_cumsum)
          tbox = torch.add(tbox,nbox)
          accuracy,correct_boxes,num_boxes = box_accuracy(targets[j]['boxes'], new_output[j]['boxes'], threshold=0.5)
          total_correct = torch.add(total_correct,correct_boxes)
          total_boxes = torch.add(total_boxes,num_boxes)

        string = ("image"+str(counter))
        counter=counter+1
        with torch.no_grad():
            view(images,new_output,1)

recall = total_tp / tbox
precision =  total_tp / (total_tp + total_fp)
accuracy = total_correct.item() / total_boxes * 100
recall = torch.cat([torch.tensor([0]), recall, torch.tensor([1])])
precision = torch.cat([torch.tensor([0]), precision, torch.tensor([0])])
precision = torch.flip(precision, dims=[0])
ap = torch.trapz(precision, recall)
print("\nAccuracy:",accuracy[0].item()," Recall: ",recall[1].item()," Precision:",precision[1].item()," mAP:",ap.item())
