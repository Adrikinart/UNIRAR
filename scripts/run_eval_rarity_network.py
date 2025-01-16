
import os
import cv2
import numpy as np
import argparse
import sys 
import time
import json

sys.path.append('..')
from utils import metrics

import torch
import matplotlib.pyplot as plt
from model import RarityNetwork

from utils.helper import show_saliency
from model import load_model, load_dataloader, run_model  

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

def run_dataset_PO3(args,name,directory ,saliency_model, rarity_model, layers_index, file_opener):
    files = os.listdir(directory)
    start_time_global = time.time()
    results = []
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # open images
        img = cv2.imread(go_path)
        if img is None:
            continue

        targ = cv2.imread(go_path.replace("images","targ_labels"),0)
        dist = cv2.imread(go_path.replace("images","dist_labels"),0)

        size = (412,412)
        targ = cv2.resize(targ, size)
        dist = cv2.resize(dist, size)

        # run model
        start_time = time.time()
        saliency, layers = run_model(args, saliency_model,file_opener ,go_path, DEFAULT_DEVICE)

        # run rarity network
        rarity_map, groups = rarity_model(
            layers_input= layers,
            layers_index=layers_index
        ) 
 
        process_time = time.time() - start_time


        # create and save maps
        maps = {
            'saliency': saliency,
            'rarity': rarity_map,
        }

        maps['saliency_Add']= rarity_model.add_rarity(saliency, rarity_map)
        # maps['saliency_Sub']= rarity_model.sub_rarity(saliency, rarity_map)
        maps['saliency_Prod']= rarity_model.prod_rarity(saliency, rarity_map)
        maps['saliency_Itti']= rarity_model.fuse_rarity(saliency, rarity_map)

        data = {
            "filename": filename,
            'path' : directory + "images",
            'metrics' : 
            {
            }
        }

        for key, v in maps.items():
            map_ = v.permute(1,2,0).detach().cpu().numpy()

            map_ = cv2.resize(map_, (targ.shape[1], targ.shape[0]))
            map_ = cv2.normalize(map_, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            msr = metrics.compute_msr(map_, targ, dist)
            data['metrics'][key] = {    
                'msrt': float(round(msr['msrt'], 4)),
                'msrb': float(round(msr['msrb'], 4)),
            }

        results.append(data)

        process_time_global = time.time() - start_time_global

        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (index + 1) / process_time_global
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()

        # # show results
        if args.show == "true":
            show_saliency(
                img= img,
                maps= maps,
                details= groups
            )
            plt.show()



            # Save results as JSON file
    print()
    results_path = os.path.join(args.path, f"{name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)


def run_dataset_salicon(args, saliency_model, rarity_model, layers_index, file_opener):
    files = os.listdir(args.salicon)
    start_time_global = time.time()
    results = []
    for index, filename in enumerate(files):
        go_path = os.path.join(args.salicon , filename)

        # open images
        img = cv2.imread(go_path)
        if img is None:
            continue

        # fix = cv2.imread(os.path.join(args.mit1003.replace("images","fixation") , filename.replace(".jpeg" , "_fixMap.jpg")),0)
        sal = cv2.imread(os.path.join(args.salicon.replace("images","saliency") , filename.replace(".jpg" , ".png")),0)

        size = (412,412)
        # fix = cv2.resize(fix, size)
        sal = cv2.resize(sal, size)

        # run model
        start_time = time.time()
        saliency, layers = run_model(args, saliency_model,file_opener ,go_path, DEFAULT_DEVICE)

        # run rarity network
        rarity_map, groups = rarity_model(
            layers_input= layers,
            layers_index=layers_index
        ) 
 
        process_time = time.time() - start_time



        start_time = time.time()

        # create and save maps
        maps = {
            'GT' : torch.tensor(sal).unsqueeze(0).float().to(DEFAULT_DEVICE),
            'saliency': saliency.clone(),
            'rarity': rarity_map.clone(),
        }

        maps['saliency_Add']= rarity_model.add_rarity(saliency.clone(), rarity_map.clone())
        # maps['saliency_Sub']= rarity_model.sub_rarity(saliency.clone(), rarity_map.clone())
        maps['saliency_Prod']= rarity_model.prod_rarity(saliency.clone(), rarity_map.clone())
        maps['saliency_Itti']= rarity_model.fuse_rarity(saliency.clone(), rarity_map.clone())

        # Normalize fix and sal between 0 and 1
        sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        data = {
            "filename": filename,
            'path' : args.mit1003 + "images",
            'metrics' : 
            {
            }
        }

        for key, v in maps.items():
            map_ = v.permute(1,2,0).detach().cpu().numpy()

            data['metrics'][key] = {    
                # 'NSS': float(round(metrics.NSS_score(map_.copy(), fix.copy()), 4)),
                'CC': float(round(metrics.CC_score(map_.copy(), sal.copy()), 4)),
                # 'KLD': float(round(metrics.KLdiv(map_.copy(), fix.copy()), 4)),
                'SIM': float(round(metrics.SIM(map_.copy(), sal.copy()), 4)),
                # 'AUC-J': round(metrics.AUC_Judd(map_, fix), 2),
            }
        process_time2 = time.time() - start_time
        # print(f"Time to compute model: {process_time:.4f}s")
        # print(f"Time to compute metrics: {process_time2:.4f}s")
        

        results.append(data)

        process_time_global = time.time() - start_time_global

        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (index + 1) / process_time_global
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()

        # # show results
        if args.show == "true":
            print()
            for key, value in data['metrics'].items():
                print(f"{key}: {value}")

            show_saliency(
                img= img,
                maps= maps,
                details= groups
            )
            plt.show()


            # Save results as JSON file
    print()
    results_path = os.path.join(args.path, f"mit1003_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)



def run_dataset_mit1003(args, saliency_model, rarity_model, layers_index, file_opener):
    files = os.listdir(args.mit1003)
    start_time_global = time.time()
    results = []
    for index, filename in enumerate(files):
        go_path = os.path.join(args.mit1003 , filename)

        # open images
        img = cv2.imread(go_path)
        if img is None:
            continue

        fix = cv2.imread(os.path.join(args.mit1003.replace("images","fixation") , filename.replace(".jpeg" , "_fixMap.jpg")),0)
        sal = cv2.imread(os.path.join(args.mit1003.replace("images","saliency") , filename.replace(".jpeg" , "SM.jpg")),0)

        size = (412,412)
        fix = cv2.resize(fix, size)
        sal = cv2.resize(sal, size)

        # run model
        start_time = time.time()
        saliency, layers = run_model(args, saliency_model,file_opener ,go_path, DEFAULT_DEVICE)

        # run rarity network
        rarity_map, groups = rarity_model(
            layers_input= layers,
            layers_index=layers_index
        ) 
 
        process_time = time.time() - start_time



        start_time = time.time()

        # create and save maps
        maps = {
            'Fixation' : torch.tensor(fix).unsqueeze(0).float().to(DEFAULT_DEVICE),
            'GT' : torch.tensor(sal).unsqueeze(0).float().to(DEFAULT_DEVICE),
            'saliency': saliency.clone(),
            'rarity': rarity_map.clone(),
        }

        maps['saliency_Add']= rarity_model.add_rarity(saliency.clone(), rarity_map.clone())
        # maps['saliency_Sub']= rarity_model.sub_rarity(saliency.clone(), rarity_map.clone())
        maps['saliency_Prod']= rarity_model.prod_rarity(saliency.clone(), rarity_map.clone())
        maps['saliency_Itti']= rarity_model.fuse_rarity(saliency.clone(), rarity_map.clone())

        # Normalize fix and sal between 0 and 1
        fix = cv2.normalize(fix, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        data = {
            "filename": filename,
            'path' : args.mit1003 + "images",
            'metrics' : 
            {
            }
        }

        for key, v in maps.items():
            if key == "Fixation":
                continue
            map_ = v.permute(1,2,0).detach().cpu().numpy()

            data['metrics'][key] = {    
                'NSS': float(round(metrics.NSS_score(map_.copy(), fix.copy()), 4)),
                'CC': float(round(metrics.CC_score(map_.copy(), sal.copy()), 4)),
                'KLD': float(round(metrics.KLdiv(map_.copy(), fix.copy()), 4)),
                'SIM': float(round(metrics.SIM(map_.copy(), sal.copy()), 4)),
                # 'AUC-J': round(metrics.AUC_Judd(map_, fix), 2),
            }
        process_time2 = time.time() - start_time
        # print(f"Time to compute model: {process_time:.4f}s")
        # print(f"Time to compute metrics: {process_time2:.4f}s")
        

        results.append(data)

        process_time_global = time.time() - start_time_global

        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (index + 1) / process_time_global
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()

        # # show results
        if args.show == "true":
            print()
            for key, value in data['metrics'].items():
                print(f"{key}: {value}")

            show_saliency(
                img= img,
                maps= maps,
                details= groups
            )
            plt.show()


            # Save results as JSON file
    print()
    results_path = os.path.join(args.path, f"mit1003_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    parser.add_argument(
        "--finetune", 
        type=str, 
        default="true",
        help="true for finetune, false for no finetune",
        choices = ["true", "false"]
    )

    parser.add_argument(
        "--model", 
        type=str, 
        default="Unisal", 
        help="select model",
        choices= ["Unisal", "TranSalNetDense" , "TranSalNetRes", "TempSal"]
    )

    parser.add_argument(
        "--path", 
        type=str, 
        default="../results/",
        help="path to save results"
    )

    parser.add_argument(
        "--mit1003", 
        type=str, 
        default="C:/Users/lelon/Documents/Dataset/MIT1003/images/",
        help="path directory images"
    )

    parser.add_argument(
        "--salicon", 
        type=str, 
        default="C:/Users/lelon/Documents/Dataset/salicon/images/val/",
        help="path directory images"
    )

    parser.add_argument(
        "--P3", 
        type=str, 
        default="C:/Users/lelon/Documents/Dataset/P3_Data/",
        help="path directory images"
    )

    parser.add_argument(
        "--O3", 
        type=str, 
        default="C:/Users/lelon/Documents/Dataset/O3_Data/",
        help="path directory images"
    )

    parser.add_argument(
        "--show", 
        type=str, 
        default="false",
        help="true for show, false for no ",
        choices = ["true", "false"]
    )


    # get arguments
    args = parser.parse_args()

    # rarity network
    rarity_model= RarityNetwork(
        threshold=args.threshold
    )

    # load model
    model, layers_index = load_model(args, DEFAULT_DEVICE)

    # load file opener
    file_opener = load_dataloader(args)

    # load model on default device
    rarity_model= rarity_model.to(DEFAULT_DEVICE)
    model = model.to(DEFAULT_DEVICE)
    args.path+= f"{args.model}_finetune_{args.finetune.lower()}_threshold_{args.threshold}/"

    os.makedirs(args.path, exist_ok=True)
    print(f"Results will be saved in {args.path}")
    print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)

    #  run dataset test
    # run_dataset_salicon(
    #     saliency_model= model,
    #     rarity_model= rarity_model,
    #     file_opener= file_opener,
    #     layers_index= layers_index,
    #     args= args,
    # )

    #  run dataset test
    run_dataset_mit1003(
        saliency_model= model,
        rarity_model= rarity_model,
        file_opener= file_opener,
        layers_index= layers_index,
        args= args,
    )

    #  run dataset test
    run_dataset_PO3(
        saliency_model= model,
        directory= args.P3 + "colors/images/",
        name= "p3_colors",
        rarity_model= rarity_model,
        file_opener= file_opener,
        layers_index= layers_index,
        args= args,
    )

    #  run dataset test
    run_dataset_PO3(
        saliency_model= model,
        directory= args.P3 + "sizes/images/",
        name= "p3_sizes",
        rarity_model= rarity_model,
        file_opener= file_opener,
        layers_index= layers_index,
        args= args,
    )

    #  run dataset test
    run_dataset_PO3(
        saliency_model= model,
        directory= args.P3 + "orientations/images/",
        name= "p3_orientations",
        rarity_model= rarity_model,
        file_opener= file_opener,
        layers_index= layers_index,
        args= args,
    )

    #  run dataset test
    run_dataset_PO3(
        saliency_model= model,
        directory= args.O3 + "/images/",
        name= "o3",
        rarity_model= rarity_model,
        file_opener= file_opener,
        layers_index= layers_index,
        args= args,
    )
