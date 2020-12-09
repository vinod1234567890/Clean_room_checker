import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# from torchsummary import summary
from PIL import Image
from PIL import ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.nn.functional import normalize as l2norm
from sklearn.metrics.pairwise import cosine_similarity

transform = transforms.Compose(
    [
        transforms.Resize((224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def open_image(path):
    im = cv2.imread(path)
    cv_img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv_img_rgb)
    return transform(pil_im).unsqueeze(0)


def get_distance(model, query_image_path):
    model.eval()
    # loading the original image finger prints
    original_image_finger_print = torch.tensor(
        np.load(
            os.path.join(
                os.getcwd(),
                "dataset",
                "Image_features",
                "original_image_finger_print.npy",
            )
        )
    )
    query_image_finger_print = model(open_image(query_image_path))
    norm_original = (
        l2norm(original_image_finger_print, p=2, dim=1)
        .flatten(start_dim=1)
        .detach()
        .numpy()
    )
    norm_query = (
        l2norm(query_image_finger_print, p=2, dim=1)
        .flatten(start_dim=1)
        .detach()
        .numpy()
    )
    return round(cosine_similarity(norm_original, norm_query)[0][0], 4)


def get_cosine_similarity():
    resnet18 = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
    for param in resnet18.parameters():
        param.requires_grad_(False)
    # getting the file name
    file_nme = os.listdir(os.path.join(os.getcwd(), "dataset", "Images"))[0]
    res18_dist = get_distance(
        resnet18,
        os.path.join(
            os.getcwd(),
            "dataset",
            "Images",
            file_nme,
        ),
    )
    del resnet18
    return res18_dist
