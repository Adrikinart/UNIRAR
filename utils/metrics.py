import torch
from sklearn.metrics import roc_auc_score

import cv2
import random
import numpy as np
import skimage.morphology as morph

def count_fixations(saliency_map, target_mask):
    """
    Compte le nombre de fixations nécessaires pour atteindre la cible.
    
    Args:
        saliency_map (torch.Tensor): Carte de saillance (H x W).
        target_mask (torch.Tensor): Masque binaire (H x W) de la cible.
    
    Returns:
        int: Nombre de fixations.
    """
    fixations = 0
    current_map = saliency_map.clone()
    while not target_reached(current_map, target_mask):
        fixations += 1
        max_pos = torch.argmax(current_map)
        current_map.view(-1)[max_pos] = 0
    return fixations

def target_reached(saliency_map, target_mask):
    """
    Vérifie si la cible a été atteinte.
    
    Args:
        saliency_map (torch.Tensor): Carte de saillance (H x W).
        target_mask (torch.Tensor): Masque binaire (H x W) de la cible.
    
    Returns:
        bool: True si la cible est atteinte, sinon False.
    """
    max_pos = torch.argmax(saliency_map.view(-1))
    return target_mask.view(-1)[max_pos].item() == 1

def compute_gsi(target_saliency, distractor_saliency):
    """
    Calcule le GSI (Global Saliency Index).
    
    Args:
        target_saliency (torch.Tensor): Saillance de la cible.
        distractor_saliency (torch.Tensor): Saillance des distracteurs.
    
    Returns:
        float: GSI.
    """
    return target_saliency.mean().item() - distractor_saliency.mean().item()




def correlation_coefficient(predicted, ground_truth):
    """
    Calcule le coefficient de corrélation (CC) entre deux cartes.
    
    Args:
        predicted (torch.Tensor): Carte prédite (H x W).
        ground_truth (torch.Tensor): Carte réelle (H x W).
    
    Returns:
        float: Coefficient de corrélation.
    """
    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()
    mean_pred = predicted.mean()
    mean_gt = ground_truth.mean()
    numerator = ((predicted - mean_pred) * (ground_truth - mean_gt)).sum()
    denominator = torch.sqrt(((predicted - mean_pred) ** 2).sum() * ((ground_truth - mean_gt) ** 2).sum())
    return (numerator / denominator).item()

def kl_divergence(predicted, ground_truth):
    """
    Calcule la divergence de Kullback-Leibler (KL).
    
    Args:
        predicted (torch.Tensor): Carte prédite (H x W).
        ground_truth (torch.Tensor): Carte réelle (H x W).
    
    Returns:
        float: KL divergence.
    """
    predicted = predicted / predicted.sum()
    ground_truth = ground_truth / ground_truth.sum()
    return (ground_truth * (ground_truth.log() - predicted.log())).sum().item()

def auc_score(predicted, fixations):
    """
    Calcule l'AUC pour les fixations binaires.
    
    Args:
        predicted (torch.Tensor): Carte prédite (H x W).
        fixations (torch.Tensor): Masque binaire des fixations (H x W).
    
    Returns:
        float: Score AUC.
    """
    predicted = predicted.flatten().cpu().numpy()
    fixations = fixations.flatten().cpu().numpy()
    return roc_auc_score(fixations, predicted)

def nss(saliency_map, fixation_map):
    """
    Calcule le NSS (Normalized Scan-path Saliency).
    
    Args:
        saliency_map (torch.Tensor): Carte de saillance (H x W).
        fixation_map (torch.Tensor): Carte des fixations (H x W).
    
    Returns:
        float: NSS.
    """
    saliency_map = (saliency_map - saliency_map.mean()) / saliency_map.std()
    return saliency_map[fixation_map > 0].mean().item()

def similarity(predicted, ground_truth):
    """
    Calcule la similarité entre deux cartes.
    
    Args:
        predicted (torch.Tensor): Carte prédite (H x W).
        ground_truth (torch.Tensor): Carte réelle (H x W).
    
    Returns:
        float: Similarité.
    """
    predicted = predicted / predicted.sum()
    ground_truth = ground_truth / ground_truth.sum()
    return torch.min(predicted, ground_truth).sum().item()


def compute_msr(salmap, targmap, distmap):
    """
    Calcule les ratios de saillance maximale (MSRt, MSRb).
    
    Args:
        target_saliency (torch.Tensor): Saillance de la cible.
        distractor_saliency (torch.Tensor): Saillance des distracteurs.
        background_saliency (torch.Tensor): Saillance de l'arrière-plan.
    
    Returns:
        tuple: (MSRt, MSRb).
    """

    msrt = MSR_targ(salmap,targmap,distmap)
    msrb = MSR_bg(salmap,targmap,distmap)
    return {'msrt':msrt, 'msrb':msrb}


def MSR_targ(salmap, targmap, distmap, dilate=0, add_eps=False):
    if isinstance(salmap, str):
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE) # assume that this is a grayscale binary map with white for target and black for non-target
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE) # assume that this is a grayscale binary map with white for distractors and black for non-distractors

    if add_eps:
        randimg = [random.uniform(0, 1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_copy = targmap.copy()
    distmap_copy = distmap.copy()

    # dilate the target and distractor maps to allow for saliency bleed
    if dilate > 0:
        targmap_copy = morph.dilation(targmap_copy.astype(np.uint8), morph.disk(dilate))
        distmap_copy = morph.dilation(distmap_copy.astype(np.uint8), morph.disk(dilate))

    # convert the target and distractor masks into arrays with 0 and 1 for values
    targmap_normalized = targmap_copy / 255
    distmap_normalized = distmap_copy / 255
    salmap_normalized = salmap/255

    maxt = np.max(np.multiply(salmap_normalized, targmap_normalized))
    maxd = np.max(np.multiply(salmap_normalized, distmap_normalized))

    if maxd > 0:
        score = maxt/maxd
    else:
        score = -1

    return float( score )


def MSR_bg(salmap, targmap, distmap, dilate=0, add_eps=False):
    if isinstance(salmap, str):
        salmap = cv2.imread(salmap, cv2.IMREAD_GRAYSCALE) # we only want the grayscale version, since saliency maps should all be grayscale
    if isinstance(targmap, str):
        targmap = cv2.imread(targmap, cv2.IMREAD_GRAYSCALE) # a grayscale binary map with white for target and black for non-target
    if isinstance(distmap, str):
        distmap = cv2.imread(distmap, cv2.IMREAD_GRAYSCALE) # a grayscale binary map with white for distractors and black for non-distractors

    if add_eps:
        randimg = [random.uniform(0,1/100000) for _ in range(salmap.size)]
        randimg = np.reshape(randimg, salmap.shape)
        salmap = salmap + randimg

    targmap_copy = targmap.copy()
    distmap_copy = distmap.copy()

    # dilate the target and distractor maps to allow for saliency bleed
    if dilate > 0:
        targmap_copy = morph.dilation(targmap_copy.astype(np.uint8), morph.disk(dilate))
        distmap_copy = morph.dilation(distmap_copy.astype(np.uint8), morph.disk(dilate))

    # convert the target and distractor masks into arrays with 0 and 1 for values
    targmap_normalized = targmap_copy / 255
    distmap_normalized = distmap_copy / 255
    salmap_normalized = salmap / 255
    # compute background mask from the target and distractor masks
    bgmap_normalized = 1 - np.logical_or(targmap_normalized > 0.5, distmap_normalized > 0.5)

    maxt = np.max(np.multiply(salmap_normalized, targmap_normalized))
    maxb = np.max(np.multiply(salmap_normalized, bgmap_normalized))

    if maxt > 0:
        score = maxb/maxt
    else:
        score = -1

    return float( score )
