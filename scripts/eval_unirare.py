
import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys 
import time
sys.path.append('..')
from model import UNIRARE
from data import FileOpener
from utils import metrics
import json

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)

def post_process(map, img):
    smap = np.exp(map)
    smap = np.squeeze(smap)
    smap = smap
    map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
    return cv2.resize(map_ , (img.shape[1] , img.shape[0]))


def show_images(img, saliency, saliency_rare, saliency_fusion_add, saliency_fusion_rs,saliency_fusion_prod, saliency_rare_details, targ, dist):

    saliency_fusion_add_color = cv2.applyColorMap(saliency_fusion_add, cv2.COLORMAP_JET)
    saliency_fusion_add_color = cv2.cvtColor(saliency_fusion_add_color, cv2.COLOR_BGR2RGB)

    saliency_fusion_rs_color = cv2.applyColorMap(saliency_fusion_rs, cv2.COLORMAP_JET)
    saliency_fusion_rs_color = cv2.cvtColor(saliency_fusion_rs_color, cv2.COLOR_BGR2RGB)

    saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)

    saliency_rare_color = cv2.applyColorMap(saliency_rare, cv2.COLORMAP_JET)
    saliency_rare_color = cv2.cvtColor(saliency_rare_color, cv2.COLOR_BGR2RGB)


    saliency_color_img = cv2.addWeighted(img, 0.6, saliency_color, 0.4, 0)
    saliency_rare_color_img = cv2.addWeighted(img, 0.6, saliency_rare_color, 0.4, 0)

    saliency_fusion_add_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_add_color, 0.4, 0)
    saliency_fusion_rs_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_rs_color, 0.4, 0)


    
    saliency_fusion_add_color = cv2.applyColorMap(saliency_fusion_add, cv2.COLORMAP_JET)
    saliency_fusion_add_color = cv2.cvtColor(saliency_fusion_add_color, cv2.COLOR_BGR2RGB)

    saliency_fusion_rs_color = cv2.applyColorMap(saliency_fusion_rs, cv2.COLORMAP_JET)
    saliency_fusion_rs_color = cv2.cvtColor(saliency_fusion_rs_color, cv2.COLOR_BGR2RGB)

    saliency_fusion_prod_color = cv2.applyColorMap(saliency_fusion_prod, cv2.COLORMAP_JET)
    saliency_fusion_prod_color = cv2.cvtColor(saliency_fusion_prod_color, cv2.COLOR_BGR2RGB)


    saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)

    saliency_rare_color = cv2.applyColorMap(saliency_rare, cv2.COLORMAP_JET)
    saliency_rare_color = cv2.cvtColor(saliency_rare_color, cv2.COLOR_BGR2RGB)


    saliency_color_img = cv2.addWeighted(img, 0.6, saliency_color, 0.4, 0)
    saliency_rare_color_img = cv2.addWeighted(img, 0.6, saliency_rare_color, 0.4, 0)

    saliency_fusion_add_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_add_color, 0.4, 0)
    saliency_fusion_rs_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_rs_color, 0.4, 0)
    saliency_fusion_prod_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_prod_color, 0.4, 0)



    plt.figure()

    plt.subplot(151)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Initial Image')

    plt.subplot(152)
    plt.imshow(saliency_rare)
    plt.axis('off')
    plt.title('saliency_rare')


    plt.subplot(153)
    plt.imshow(dist)
    plt.axis('off')
    plt.title('saliency_rare')

    plt.subplot(154)
    plt.imshow(targ)
    plt.axis('off')
    plt.title('saliency')


    plt.figure(figsize=(10,10))

    plt.subplot(461)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Initial Image')

    plt.subplot(462)
    plt.imshow(saliency_rare)
    plt.axis('off')
    plt.title('saliency_rare')

    plt.subplot(463)
    plt.axis('off')
    plt.imshow(saliency_color)
    plt.title('saliency')

    plt.subplot(464)
    plt.imshow(saliency_fusion_add_color)
    plt.axis('off')
    plt.title('fusion_add')

    plt.subplot(465)
    plt.imshow(saliency_fusion_rs_color)
    plt.axis('off')
    plt.title('fusion_sub')


    plt.subplot(466)
    plt.imshow(saliency_fusion_prod_color)
    plt.axis('off')
    plt.title('fusion_prod')


    plt.subplot(467)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Initial Image')

    plt.subplot(468)
    plt.imshow(saliency_rare_color_img)
    plt.axis('off')
    plt.title('saliency_rare')

    plt.subplot(469)
    plt.imshow(saliency_color_img)
    plt.axis('off')
    plt.title('saliency')

    plt.subplot(4,6,10)
    plt.imshow(saliency_fusion_add_color_img)
    plt.axis('off')
    plt.title('fusion_add')

    plt.subplot(4,6,11)
    plt.imshow(saliency_fusion_rs_color_img)
    plt.axis('off')
    plt.title('fusion_sub')


    plt.subplot(4,6,12)
    plt.imshow(saliency_fusion_prod_color_img)
    plt.axis('off')
    plt.title('fusion_prod')

    for i in range(0,saliency_rare_details.shape[0]):
        plt.subplot(4,6,13 + i)
        plt.imshow(saliency_rare_details[i,:, :])
        plt.axis('off')
        plt.title(f'Level {i}Saliency Map')

    plt.show()



def run_dataset(name,directory, model, args, path_save, show = False):


    files = os.listdir(directory)
    opener = FileOpener()
    
    results = []
    start_time_global = time.time()
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # load FileOpener
        if args.type == "image":

            # open images
            img = cv2.imread(go_path)

            # check if img is none
            if img is None:
                continue

            if name == "MIT1003":
                directory_fixation = directory.replace("images","fixation")
                directory_saliency = directory.replace("images","saliency")

                fix = cv2.imread(os.path.join(directory_fixation , filename.replace(".jpeg" , "_fixMap.jpg")),0)
                sal = cv2.imread(os.path.join(directory_saliency , filename.replace(".jpeg" , "SM.jpg")),0)

                targ = cv2.imread(os.path.join(directory_fixation , filename.replace(".jpeg" , "_fixMap.jpg")),0)
                dist = cv2.imread(os.path.join(directory_saliency , filename.replace(".jpeg" , "SM.jpg")),0)

            else:
                targ = cv2.imread(go_path.replace("images","targ_labels"),0)
                dist = cv2.imread(go_path.replace("images","dist_labels"),0)

            # open image tensor
            tensor_image = opener.open_image(
                file = go_path , 
                size = (412,412)
            )

            # process image
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0).to(DEFAULT_DEVICE)
            start_time = time.time()
            saliency, saliency_rare, saliency_rare_details = model(tensor_image, source="SALICON")
            end_time = time.time()
            process_time = end_time - start_time
            # print(f"Processing time: {process_time:.2f} seconds")

            saliency= saliency.squeeze(0).squeeze(0).detach().cpu()
            saliency_rare = saliency_rare.squeeze(0).detach().cpu()
            saliency_rare_details = saliency_rare_details.squeeze(0).squeeze(0).detach().cpu()

            saliency = post_process(saliency.detach().cpu().numpy(),img)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) * 255
            saliency_rare = (saliency_rare - saliency_rare.min()) / (saliency_rare.max() - saliency_rare.min()) * 255

            saliency = cv2.resize(saliency, (img.shape[1], img.shape[0]))
            saliency_rare = cv2.resize(saliency_rare.permute(1, 2, 0).numpy(), (img.shape[1], img.shape[0]))
            saliency_rare = cv2.GaussianBlur(saliency_rare, (5, 5), 0)

            saliency_fusion_add = saliency_rare + saliency
            saliency_fusion_add = (saliency_fusion_add - saliency_fusion_add.min()) / (saliency_fusion_add.max() - saliency_fusion_add.min()) * 255

            saliency_fusion_sub = np.abs(saliency_rare - saliency)
            saliency_fusion_sub = (saliency_fusion_sub - saliency_fusion_sub.min()) / (saliency_fusion_sub.max() - saliency_fusion_sub.min()) * 255

            saliency_fusion_prod = saliency_rare * saliency
            saliency_fusion_prod = (saliency_fusion_prod - saliency_fusion_prod.min()) / (saliency_fusion_prod.max() - saliency_fusion_prod.min()) * 255

            saliency = (saliency).astype(np.uint8)
            saliency_rare = (saliency_rare).astype(np.uint8)
            saliency_fusion_add = (saliency_fusion_add).astype(np.uint8)
            saliency_fusion_sub = (saliency_fusion_sub).astype(np.uint8)
            saliency_fusion_prod = (saliency_fusion_prod).astype(np.uint8)


            if show:
                show_images(
                    img,
                    saliency,
                    saliency_rare,
                    saliency_fusion_add,
                    saliency_fusion_sub,
                    saliency_fusion_prod,
                    saliency_rare_details,
                    targ,
                    dist
                )

            if name == "MIT1003":
                fix = cv2.normalize(fix, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
                sal = cv2.normalize(sal, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
                saliency = saliency.astype(np.float64) / 255.0

                # Resize all images to 212x212
                fix = cv2.resize(fix, (212, 212))
                sal = cv2.resize(sal, (212, 212))
                saliency = cv2.resize(saliency, (212, 212))
                saliency_rare = cv2.resize(saliency_rare, (212, 212))
                saliency_fusion_add = cv2.resize(saliency_fusion_add, (212, 212))
                saliency_fusion_sub = cv2.resize(saliency_fusion_sub, (212, 212))
                saliency_fusion_prod = cv2.resize(saliency_fusion_prod, (212, 212))


                results.append({
                    "filename": filename,
                    'path' : directory,
                    'metrics' : 
                    {
                        'saliency': {
                            'NSS': round(metrics.NSS_score(saliency, fix), 4),
                            'CC': round(metrics.CC_score(saliency, sal), 4),
                            'KLD': round(metrics.KLdiv(saliency, fix), 4),
                            'SIM': round(metrics.SIM(saliency, sal), 4),
                            'AUC-J': round(metrics.AUC_Judd(saliency, fix), 2),
                        },
                        "saliency_rare": {
                            'NSS': round(metrics.NSS_score(saliency_rare, fix), 4),
                            'CC': round(metrics.CC_score(saliency_rare, sal), 4),
                            'KLD': round(metrics.KLdiv(saliency_rare, fix), 4),
                            'SIM': round(metrics.SIM(saliency_rare, sal), 4),
                            'AUC-J': round(metrics.AUC_Judd(saliency_rare, fix), 2),
                        },
                        "saliency_fusion_add": {
                            'NSS': round(metrics.NSS_score(saliency_fusion_add, fix), 4),
                            'CC': round(metrics.CC_score(saliency_fusion_add, sal), 4),
                            'KLD': round(metrics.KLdiv(saliency_fusion_add, fix), 4),
                            'SIM': round(metrics.SIM(saliency_fusion_add, sal), 4),
                            'AUC-J': round(metrics.AUC_Judd(saliency_fusion_add, fix), 2),
                        },
                        "saliency_fusion_sub": {
                            'NSS': round(metrics.NSS_score(saliency_fusion_sub, fix), 4),
                            'CC': round(metrics.CC_score(saliency_fusion_sub, sal), 4),
                            'KLD': round(metrics.KLdiv(saliency_fusion_sub, fix), 4),
                            'SIM': round(metrics.SIM(saliency_fusion_sub, sal), 4),
                            'AUC-J': round(metrics.AUC_Judd(saliency_fusion_sub, fix), 2),
                        },
                        "saliency_fusion_prod": {
                            'NSS': round(metrics.NSS_score(saliency_fusion_prod, fix), 4),
                            'CC': round(metrics.CC_score(saliency_fusion_prod, sal), 4),
                            'KLD': round(metrics.KLdiv(saliency_fusion_prod, fix), 4),
                            'SIM': round(metrics.SIM(saliency_fusion_prod, sal), 4),
                            'AUC-J': round(metrics.AUC_Judd(saliency_fusion_prod, fix), 2),
                        }

                    }
                })


            else:

                results.append({
                    "filename": filename,
                    'path' : directory,
                    'metrics' : 
                    {
                        'saliency': metrics.compute_msr(saliency, targ, dist),
                        'saliency_rare': metrics.compute_msr(saliency_rare, targ, dist),
                        'saliency_fusion_add': metrics.compute_msr(saliency_fusion_add, targ, dist),
                        'saliency_fusion_rs': metrics.compute_msr(saliency_fusion_rs, targ, dist),

                    }
                })

        # Print loading bar with FPS information
        process_time_global = time.time() - start_time_global

        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (index + 1) / process_time_global
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()

    # Save results as JSON file
    print()
    results_path = os.path.join(path_save, f"{name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--type", 
        type=str, 
        default="image", 
        help="image or video"
    )

    parser.add_argument(
        "--name", 
        type=str, 
        default='unirare', 
        help="Name information"
    )

    parser.add_argument(
        "--source", 
        type=str, 
        default="SALICON", 
        help="SALICON / DFH1K / UCF"
    )

    parser.add_argument(
        "--finetune", 
        type=str, 
        default="true",
        help="true for finetune, false for no finetune",
        choices = ["true", "false"]
    )

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    args = parser.parse_args()

    # create model unirare
    model = UNIRARE(
        bypass_rnn=False,
        threshold=args.threshold
    ).to(DEFAULT_DEVICE)

    if args.finetune.lower() == "true":
        path_ = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists("../model/weights/" + "weights_best.pth"):
            print("Loading model")
            model.load_weights( "../model/weights/" + "weights_best.pth")
        else:
            print("Model not found")

    # Create folder in "../res" with name unirare and args information
    if args.finetune.lower() == "true":
        res_dir = os.path.join("..", "res", f"{args.name}_finetune_threshold_{args.threshold}_{args.type}")
    else:
        res_dir = os.path.join("..", "res", f"{args.name}_threshold_{args.threshold}_{args.type}")
    os.makedirs(res_dir, exist_ok=True)
    print(f"Results will be saved in {res_dir}")

    # run_dataset(
    #     name= "O3_data" ,
    #     directory = "/Users/coconut/Documents/Dataset/Saliency/O3_data/images/" , 
    #     model= model,
    #     args= args,
    #     path_save= res_dir
    # )
    # run_dataset(
    #     name= "P3_data_sizes" ,
    #     directory = "/Users/coconut/Documents/Dataset/Saliency/P3_data/sizes/images/" , 
    #     model= model,
    #     args= args,
    #     path_save= res_dir
    # )
    # run_dataset(
    #     name= "P3_data_orientations" ,
    #     directory = "/Users/coconut/Documents/Dataset/Saliency/P3_data/orientations/images/" , 
    #     model= model,
    #     args= args,
    #     path_save= res_dir
    # )
    # run_dataset(
    #     name= "P3_data_colors" ,
    #     directory = "/Users/coconut/Documents/Dataset/Saliency/P3_data/colors/images/" , 
    #     model= model,
    #     args= args,
    #     path_save= res_dir
    # )

    run_dataset(
        name= "MIT1003" ,
        directory = "/Users/coconut/Documents/Dataset/Saliency/MIT1003/images/" , 
        model= model,
        args= args,
        path_save= res_dir
    )