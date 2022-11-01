# [0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import numpy as np
from PIL import Image
from torchvision import models, transforms
from scipy import spatial
import torch


class AgeClassifier:
    def __init__(self):
        self.model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
        self.transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')
        self.ages = [2, 9, 19, 29, 39, 49, 59, 69, 70]

    def get_age(self, img_array):
        im = Image.fromarray(img_array)
        inputs = self.transforms(im, return_tensors='pt')
        output = self.model(**inputs)

        # Predicted Class probabilities
        proba = output.logits.softmax(1)

        # Predicted Classes
        preds = proba.argmax(1)
        return self.ages[preds.cpu().numpy()[0]]  # return age


class PersonMatcher:
    def __init__(self):
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.inv_norm = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.preprocess = transforms.Compose([transforms.Resize(256), transforms.ToTensor(), self.norm])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ResNet50 = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        # self.ResNet50 = models.resnet18(pretrained=True).to(self.device)
        self.ResNet50 = torch.nn.Sequential(*(list(self.ResNet50.children())[:-1])).to(self.device)  # drop last classification layer
        self.ResNet50.eval()

    def get_embds(self, person_img):
        im = Image.fromarray(person_img)
        image_tensor = self.preprocess(im)
        input_tensor = image_tensor.unsqueeze(0)  # single-image batch as wanted by model
        input_tensor = input_tensor.to(self.device)  # send tensor to TPU
        outputs = self.ResNet50(input_tensor)
        embds = outputs.cpu().detach().numpy()[0, :,0,0]
        return embds
    def get_sim_score(self, embds1, embds2):
        sim = 1 - spatial.distance.cosine(embds1, embds2)
        return sim