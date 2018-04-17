import os

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image
from imageio import imread, imwrite
from skimage.transform import resize
from sklearn.cluster import KMeans
from torchvision import models
from tqdm import tqdm
import skimage

class vgg_feature_extractor():
    def __init__(self, n_clusters=6, random_state=719):
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.extractor = self.vgg_extractor()
        
    def get_vgg_clusters(self, meta, is_train=False):
        img_filepaths = meta['file_path_image'].values

        features = []
        for filepath in tqdm(img_filepaths):
            img = imread(filepath)
            img = img / 255.0
            
            try:
                img = img[:, :, :3]
            except:
                print(filepath)
                print(img.shape)
                print(img)
                img = img.astype('uint8')
                img = skimage.color.gray2rgb(img)
                print('new\n', img)
                img = img.astype('float')/255.
                
            
            x = self.preprocess_image(img)
            feature = self.extractor(x)
            feature = np.ndarray.flatten(feature.cpu().data.numpy())
            features.append(feature)
        features = np.stack(features, axis=0)

        labels = self.cluster_features(features,is_train=is_train)

        return labels


    def vgg_extractor(self):
        model = models.vgg16(pretrained=True)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return torch.nn.Sequential(*list(model.features.children())[:-1])


    def preprocess_image(self, img, target_size=(128, 128)):
        img = resize(img, target_size, mode='constant')
        x = np.expand_dims(img, axis=0)
        x = x.transpose(0, 3, 1, 2)
        x = torch.FloatTensor(x)
        if torch.cuda.is_available():
            x = torch.autograd.Variable(x, volatile=True).cuda()
        else:
            x = torch.autograd.Variable(x, volatile=True)
        return x


    def cluster_features(self, features, is_train=False):
        if is_train:
            self.cluster_model.fit(features)
        
        labels = self.cluster_model.predict(features)
        return labels
