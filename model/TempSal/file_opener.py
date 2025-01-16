

from PIL import Image
import torchvision.transforms as transforms



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


def open_image(file):
    """
    Load and preprocess an RGB image.

    Args:
        img_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [3, 256, 256].
    """
    img = Image.open(file).convert('RGB')
    transformed_img = img_transform(img).unsqueeze(0)
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


