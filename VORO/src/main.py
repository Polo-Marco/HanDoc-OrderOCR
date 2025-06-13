import os
import glob
import sys
import argparse

import torch
from torch.utils.data import DataLoader

import json
import numpy as np


import metrics
import decode
from mth_dataset import MTHv2PairDataset,MTHv2PairImageDataset
from utils import mkdir, files_exist,save_log
from custom_models.MLP import MLP

import datetime
from custom_models.mobilenetv3 import mobilenetv3_small
from custom_models.SCNN import CNNNet
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer,scheduler , device):
    """
    Train the model for 1 epoch
    """
    model.train()
    g_loss = 0
    for batch, sample in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        x = sample["feature"].to(device)
        t = sample["label"].to(device)
        y = model(x.float())
        loss = criterion(y.squeeze(), t.float())
        g_loss += loss.data
        loss.backward()
        optimizer.step()
    scheduler.step(g_loss / (batch + 1))
    return g_loss / (batch + 1)


def validate(model, dataloader, criterion, device,mode="eval"):
    """
    evaluate the model
    """
    model.eval()
    g_loss = 0
    predictions=[]
    re_idx=[]
    re_image=[]
    for batch, sample in enumerate(tqdm(dataloader)):
        x = sample["feature"].to(device)
        t = sample["label"].to(device)
        y = model(x.float())
        if mode == "eval":
            loss = criterion(y.squeeze(), t.float())
            g_loss += loss.data
        predictions+=y.to('cpu').tolist()
        re_idx+=sample["index"]
        re_image+=sample["image"]
    return g_loss / (batch + 1),predictions,re_idx,re_image

def predict(model, dataloader, device):
    """
    predict with the model
    """
    model.eval()
    predictions=[]
    re_idx=[]
    re_image=[]
    for batch, sample in enumerate(tqdm(dataloader)):
        x = sample["feature"].to(device)
        t = sample["label"].to(device)
        y = model(x.float())
        predictions+=y.to('cpu').tolist()
        re_idx+=sample["index"]
        re_image+=sample["image"]
    return predictions,re_idx,re_image

def decode_pairs(preds,idxs,images):
    img_dic = {elem:[] for elem in set(images)}
    # save all pairs within images
    for pred,idx,img in zip(preds,idxs,images):
        i = idx.replace("[","").replace("]","").split(",")
        idx1,idx2 = int(i[0]),int(i[1])
        img_dic[img].append([idx1,idx2,pred[0]])
    #loop thru and gen matrix for decoder
    best_paths = {}
    for img in img_dic:
        length = int(np.sqrt(len(img_dic[img])))+1
        matrix = np.zeros((length,length))
        for elem in img_dic[img]:
            matrix[elem[0]][elem[1]]=elem[2]
        decoder = decode.CountDecoder(matrix)
        decoder.run()
        best_paths[img] = [int(elem) for elem in (list(decoder.best_path))]
    return best_paths
                
def evaluate(img_seqs):
    scores={"F":[],"K":[],"image":[]}
    for img in img_seqs:
        seq = img_seqs[img]
        scores["K"].append(float(metrics.kendall_tau_distance(seq,
            list(range(len(seq))),normalized=False)))
        scores["F"].append(float(metrics.spearman_footrule_distance(seq,
            list(range(len(seq))),normalized=False)))
        scores["image"].append(img)
    return scores



def main():
    parser = argparse.ArgumentParser(description='CNN classifier for Pair-wise Reading order')
    #model variables
    parser.add_argument('--seed', 
            type=float, 
            default=42,
            help='Random Seed',)
    parser.add_argument('--learning_rate',
            type=float,
            default=0.001,
            help='Learning Rate',)
    parser.add_argument('--augmentation',
            type=bool,
            default=False,
            help='use augmentation or not',)
    parser.add_argument('--batch_size',
            type=int,
            default=256,
            help='Number samples per batch',)
    parser.add_argument('--epochs',
            type=int,
            default=1200,
            help='Number of training epochs',)
    parser.add_argument('--model',
            type=str,
            default="mobilenetv3",
            help='model to use: CNN, mobilenetv3',)
    parser.add_argument('--evaluate_rate',
            type=int,
            default=300,
            help='Evaluate Validation set each number of epochs',)
    parser.add_argument('--echo_rate',
            type=int,
            default=1,
            help='Rate of info displayed',)
    parser.add_argument('--max_nondecreasing_epochs',
            type=int,
            default=5,
            help="early stop when no performance gain in n epochs",)
    parser.add_argument('--image_size',
            type=int,
            default=224,
            help="set image size",)
    parser.add_argument('--image_flag',
            type=bool,
            default=False,
            help="use image as extra feature",)
    parser.add_argument('--gpu',
            type=int,
            default=0,
            help="which gpu to use",)
    
    #path variables
    parser.add_argument('--eval_only',
            type=bool,
            default=False,
            help='eval only flag',)
    parser.add_argument('--model_path',
            type=str,
            default='./best.pth',
            help='model path for training or evaluating',)
    parser.add_argument('--image_path',
            type=str,
            default='./image/',
            help='image path for image data',)
    parser.add_argument('--out_folder',
            type=str,
            default='./',
            help='Output Folder',)
    parser.add_argument('--tr_data',
            type=str,
            default='./',
            help='Pointer to training data files',)
    parser.add_argument('--te_data',
            type=str,
            default='./',
            help='Pointer to test data files',)
    parser.add_argument('--val_data',
            type=str,
            default='./',
            help='Pointer to validation data files',)
    parser.add_argument('--exp_id',
            type=str,
            default='',
            help='Id assigned to the experiment.',)
    

    args = parser.parse_args()
    print(args)
    print(args.exp_id)
    args.out_folder = os.path.join(args.out_folder, args.exp_id)
    mkdir(args.out_folder)
    save_log(args.out_folder+"log.txt",str(args))
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    print("using device: ",device)
    
    #num_features = 16#tr_dataset.get_num_features()
    #hidden_size = num_features * 2
    #model = MLP(num_features, hidden_size, 1).to(device)
    if args.model == "mobilenetv3":
        model = mobilenetv3_small(num_classes=1).to(device)
    else:
        model = CNNNet(num_classes=1).to(device)
    if not args.eval_only:
        tr_dataset = MTHv2PairImageDataset(args.tr_data,args.image_path,
                augmentation=args.augmentation,RegionType="line",image_size=args.image_size,
                                           image_flag=args.image_flag
        )
    
        tr_dataloader = DataLoader(
            tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
        )
        print("Train num. samples: ", len(tr_dataset))
        criterion = torch.nn.BCELoss().to(device)#torch.nn.CrossEntropyLoss().to(device)#BCEloss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    else:
        print("evaluation only")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        #model.load_state_dict(torch.load(args.model_path), map_location=device)

    # --- val data
    val_dataset = MTHv2PairImageDataset(args.val_data,args.image_path,
            RegionType="line",image_size=args.image_size,image_flag=args.image_flag)
    print("Val num. pairs: ", len(val_dataset))
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10,
    )
    te_dataset = MTHv2PairImageDataset(args.te_data,args.image_path,
            RegionType="line",image_size=args.image_size,image_flag=args.image_flag)
    print("test num. pairs: ", len(te_dataset))

    # --- train the model
    best_val_loss = 10000
    best_k_score = 1000
    best_epoch = 0

    
    if not args.eval_only:
        for epoch in range(args.epochs):
            t_loss = train(model, tr_dataloader, criterion, optimizer,scheduler, device)
            v_loss,preds,idxs,images = validate(model, val_dataloader, criterion, device)
            img_seqs = decode_pairs(preds,idxs,images)
            scores = evaluate(img_seqs)
            k_score = np.mean(scores["K"])
            f_score = np.mean(scores["F"])
            info_str="Val F(s,t), K(s,t) after {} epochs: {}, {}"
            print(info_str.format(epoch,
                    f_score,k_score))
            save_log(args.out_folder+"log.txt",info_str.format(epoch,f_score,k_score))
            
            if not epoch % args.echo_rate:
                info_str = "Epoch {} : train-loss: {} val-loss: {}"
                print(
                    info_str.format(
                        epoch, t_loss, v_loss
                    )
                )
                save_log(args.out_folder+"log.txt",info_str.format(epoch, t_loss, v_loss))
            if best_k_score > k_score:
                best_k_score = k_score
                best_epoch = epoch
                print("saving best model with best k score: ",k_score)
                torch.save(
                    model.state_dict(), 
                    os.path.join(args.out_folder, "model.pth")
                )
    
            if (epoch - best_epoch) == args.max_nondecreasing_epochs:
                print(
                    "Loss DID NOT decrease after {} consecutive epochs.".format(
                        args.max_nondecreasing_epochs
                    )
                )
                break
            #eval mode
            # if ((epoch) % args.evaluate_rate) == 0:
            #     #preds,idxs,images = predict(model, val_dataloader, device)
            #     img_seqs = decode_pairs(preds,idxs,images)
            #     scores = evaluate(img_seqs)
            #     print("Val F(s,t), K(s,t) after {} epochs: {}, {}".format(epoch,
            #         np.mean(scores["F"]),np.mean(scores["K"])))
        #load best model:
        model.load_state_dict(torch.load(os.path.join(args.out_folder, "model.pth")))
    print("evaluating...")
    te_dataloader = DataLoader(
        te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10,
    )
    _,preds,idxs,images =  validate(model, te_dataloader, None, device,mode="prediction")
    img_seqs = decode_pairs(preds,idxs,images)
    scores = evaluate(img_seqs)
    k_score = np.mean(scores["K"])
    f_score = np.mean(scores["F"])
    print("average Footrules score: ",f_score)
    print("average Kendall score: ",k_score)
    save_log(args.out_folder+"log.txt","result on test set: ")
    save_log(args.out_folder+"log.txt","average Footrules score: "+str(f_score))
    save_log(args.out_folder+"log.txt","average Kendall score: "+str(k_score))

    with open(args.out_folder+"evaluation.json",'w')as wf:
        json.dump(scores,wf)
    with open(args.out_folder+"prediction.json",'w')as wf:
        json.dump(img_seqs,wf)

if __name__ == "__main__":
    main()
