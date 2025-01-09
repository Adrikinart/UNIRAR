
import os
import cv2
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys 
sys.path.append('..')
from model import DeepRare
import time


if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)



def process_image(image):
        # Load an example image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(img).unsqueeze(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DeepRare")

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    args = parser.parse_args()


    rarity_network = DeepRare(
        threshold=args.threshold,
    ) # instantiate class
    directory = r'inputs/images/'

    rarity_network= rarity_network.to(DEFAULT_DEVICE)

    for filename in os.listdir(directory):
        print(filename)
        go_path = os.path.join(directory, filename)

        img = cv2.imread(go_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        input_image = process_image(img).to(DEFAULT_DEVICE)


        start_time = time.time()
        SAL, groups = rarity_network(input_image)
        end_time = time.time()

        print(f"Processing time for rarity network: {end_time - start_time:.4f} seconds")


        groups = groups.squeeze(0).detach().cpu()
        SAL = SAL.squeeze(0).detach().cpu()

        plt.figure(1)

        plt.subplot(421)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Initial Image')

        plt.subplot(422)
        plt.imshow(SAL.permute(1, 2, 0))
        plt.axis('off')
        plt.title('Final Saliency Map')

        for i in range(0,groups.shape[0]):

            plt.subplot(423 + i)
            plt.imshow(groups[i,:, :])
            plt.axis('off')
            plt.title(f'Level {i}Saliency Map')

        plt.show()