import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from model import UNISAL, UNIRARE, DeepRare
from data import FileOpener

class Runner:
    
    def __init__(self, model_type, model_path, source="SALICON", threshold=None):
        self.model_type = model_type
        self.model_path = model_path
        self.source = source
        self.threshold = threshold
        self.model = self.load_model()

    def load_model(self):
        if self.model_type == "unisal":
            model = UNISAL(bypass_rnn=False)
        elif self.model_type == "unirare":
            model = UNIRARE(bypass_rnn=False)
        elif self.model_type == "deeprare":
            model = DeepRare(threshold=self.threshold)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if os.path.exists(self.model_path):
            model.load_weights(self.model_path)
        return model

    def post_process(self, map, img):
        smap = np.exp(map)
        smap = np.squeeze(smap)
        map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
        return cv2.resize(map_, (img.shape[1], img.shape[0]))

    def run_inference(self, input_path, input_type="image"):
        opener = FileOpener()
        files = os.listdir(input_path)

        for filename in files:
            go_path = os.path.join(input_path, filename)
            if input_type == "image":
                tensor_image = opener.open_image(file=go_path, size=(412, 412))
                img = cv2.imread(go_path)
                tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
                if self.model_type == "unirare":
                    map_, SAL, groups = self.model(tensor_image, source=self.source)
                    map_ = map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
                    map_ = self.post_process(map_, img)
                    self.display_results(img, map_, SAL, groups)
                else:
                    map_ = self.model(tensor_image, source=self.source)
                    map_ = map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
                    map_ = self.post_process(map_, img)
                    self.display_results(img, map_)
            elif input_type == "video":
                tensors_frames = opener.open_video(file=go_path, fps=15, seq_len=12, size=(412, 412))
                # Process video frames
                # ...

    def display_results(self, img, map_, SAL=None, groups=None):
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Initial Image')

        plt.subplot(122)
        plt.imshow(map_)
        plt.axis('off')
        plt.title('Final Saliency Map')

        if SAL is not None and groups is not None:
            plt.subplot(422)
            plt.imshow(SAL)
            plt.axis('off')
            plt.title('Final Saliency Map')

            for i in range(groups.shape[-1]):
                plt.subplot(423 + i)
                plt.imshow(groups[:, :, i])
                plt.axis('off')
                plt.title(f'Level {i} Saliency Map')

        plt.show()
