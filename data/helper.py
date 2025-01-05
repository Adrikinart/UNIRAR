import torch

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    tensor.fill_(1. / tensor.numel())
    return tensor


import re

def extract_decimal(filename):
    decimal_pattern = re.compile(r'(\d+)\.jpg$')

    match = decimal_pattern.search(filename)
    if match:
        return float(match.group(1))  # Convertir le nombre en float
    return float('inf')  # Si aucun nombre trouvé, renvoyer une valeur très élevée pour les placer à la fin
