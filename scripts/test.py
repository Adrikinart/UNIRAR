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
from model import TempSal
from utils import metrics
import json

from PIL import Image

DEFAULT_DEVICE = (
    torch.device("cuda:0") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {DEFAULT_DEVICE}")


# Transformations for the input images
img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])
        ])
gt_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            #transforms.Normalize([0.5],[0.5])
        ])

def to_np(tensor):
    return tensor.permute( 1, 2, 0).detach().cpu().numpy()


def get_image(img_path):
    """
    Load and preprocess an RGB image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [3, 256, 256].
    """
    img = Image.open(img_path).convert('RGB')
    transformed_img = img_transform(img)
    return transformed_img

def get_gt_tensor(img_path):
    """
    Load and preprocess a grayscale ground-truth image.

    Args:
        img_path (str): Path to the ground truth image file.

    Returns:
        torch.Tensor: Preprocessed ground truth tensor of shape [1, 256, 256].
    """
    img = Image.open(img_path).convert('L')
    return gt_transform(img)




def show_images(img, saliency, saliency_rare, saliency_fusion_add, saliency_fusion_rs, saliency_rare_details, args, index):
    """
    Display and save saliency maps with visualizations.

    Args:
        img (np.ndarray): Original image in NumPy format with shape [H, W, 3].
        saliency (np.ndarray): Predicted saliency map.
        saliency_rare (np.ndarray): Rare feature saliency map.
        saliency_fusion_add (np.ndarray): Additive fusion saliency map.
        saliency_fusion_rs (np.ndarray): Residual fusion saliency map.
        saliency_rare_details (torch.Tensor): Rare saliency maps for different time slices.
        args (argparse.Namespace): Parsed arguments.
        index (int): Index of the image.
    """
    def apply_color_map(saliency_map):
            if saliency_map.ndim != 2:
                raise ValueError("Saliency map must be a 2D array.")
            colored = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def overlay_image(base, overlay):
        if base.shape[:2] != overlay.shape[:2]:
            raise ValueError("Base image and overlay must have the same dimensions.")
        return cv2.addWeighted(base, 0.6, overlay, 0.4, 0)

    maps = {
        "Initial Image": img,
        "Saliency Rare": apply_color_map(saliency_rare),
        "Saliency": apply_color_map(saliency),
        "Fusion Add": apply_color_map(saliency_fusion_add),
        "Fusion RS": apply_color_map(saliency_fusion_rs)
    }

    plt.figure(figsize=(15, 10))
    for i, (title, map_) in enumerate(maps.items(), 1):
        plt.subplot(2, 3, i)
        plt.imshow(map_ if "Image" in title else overlay_image(img, map_))
        plt.axis('off')
        plt.title(title)

    num_imgs = saliency_rare_details.shape[0]
    plt.subplots_adjust(wspace=0.5)
    for i in range(1, num_imgs + 1):
        plt.subplot(1, num_imgs, i)
        plt.imshow(saliency_rare_details[i - 1].cpu().squeeze().permute(1, 2, 0).detach().numpy())
        plt.axis('off')
        plt.title(f"Time {i - 1}-{i} s")

    output_dir = "./outputs/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"image_{index}_{args.name}_threshold_{args.threshold}.jpeg")
    plt.savefig(output_path)
    plt.show()

def run_dataset(name,directory, model, args):
    """
    Process the dataset and evaluate the model.

    Args:
        name (str): Dataset name.
        directory (str): Path to the dataset directory.
        model (torch.nn.Module): Saliency model.
        args (argparse.Namespace): Parsed arguments.
    """

    print(f"Running dataset: {name}")

    files = os.listdir(directory)
    
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # load FileOpener
        if args.type == "image":
            
            image = get_image(go_path)

            # shape:
            dimension = image.shape

            if image.dim() == 3:  # Shape: [Height, Width, Channels]
                # Add a batch dimension: [1, Height, Width, Channels]
                image = image.unsqueeze(0)
            
            #print(f'dimension before permutation: {image.shape}') # torch.Size([1, 3, 412, 412])
            image = image.to(DEFAULT_DEVICE)

            start_time = time.time()

            #print(f'dimension after: {image.shape}')

            with torch.no_grad():
                saliency, saliency_time, layers = model(image)
            
            print(f'saliency_rare_details: {saliency_time.shape}')
            print(f'saliency dimension: {saliency.shape}')

            saliency = saliency.squeeze(0).detach().cpu()
            saliency_time = saliency_time.squeeze(0).detach().cpu()

            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) * 255
        
            #saliency = cv2.resize(saliency, (dimension[1], dimension[2]))
            #plt.imshow(saliency)
            #plt.show()
            end_time = time.time()
            process_time = end_time - start_time

            plt.subplot(1,7, 1)
            plt.imshow(saliency)
            plt.axis('off')
            plt.title("Saliency")

            for i in range(saliency_time.shape[0]):
                plt.subplot(1, 7, i+1)
                plt.imshow(saliency_time[i])
                plt.axis('off')
                plt.title(f"Time {i}-{i+1} s")

            plt.show()

            break




            """ 

            # plot saliency map
            
        
            saliency_fusion_add = saliency_rare + saliency
            saliency_fusion_add = (saliency_fusion_add - saliency_fusion_add.min()) / (saliency_fusion_add.max() - saliency_fusion_add.min()) * 255

            saliency_fusion_rs = np.abs(saliency_rare - saliency)
            saliency_fusion_rs = (saliency_fusion_rs - saliency_fusion_rs.min()) / (saliency_fusion_rs.max() - saliency_fusion_rs.min()) * 255


            saliency = (saliency).astype(np.uint8)
            saliency_rare = (saliency_rare).astype(np.uint8)
            saliency_fusion_add = (saliency_fusion_add).astype(np.uint8)
            saliency_fusion_rs = (saliency_fusion_rs).astype(np.uint8)


            show_images(
                image,
                saliency,
                saliency_rare,
                saliency_fusion_add,
                saliency_fusion_rs,
                saliency_rare_details, # here it's sec 1, 2, ... 5
                args,
                index
            )
        """


        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test saliency prediction on images")
    parser.add_argument("--type", type=str, default="image", choices=["image", "video"], help="Input type")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for saliency map")
    parser.add_argument("--source", type=str, default="SALICON", help="Data source")
    parser.add_argument("--finetune", type=str, choices=["true", "false"], default="true", help="Use finetuned model")
    parser.add_argument("--name", type=str, default="TempSal", help="Experiment name")
    parser.add_argument("--time_slices", type=int, default=5, help="Number of time slices")
    parser.add_argument("--train_model", type=int, default=0, help="Flag for training mode")
    args = parser.parse_args()

    # create model TempSal
    model_checkpoint_path = "./weights/multilevel_tempsal.pt"
    time_slices = args.time_slices
    train_model = args.train_model

    model = TempSal(
        device=DEFAULT_DEVICE,
        model_path=model_checkpoint_path,
        model_vol_path= model_checkpoint_path,
        time_slices=args.time_slices,
        train_model=args.train_model
    ).to(DEFAULT_DEVICE)

    run_dataset(
        name="test",
        directory="inputs/images",
        model=model,
        args=args
    )