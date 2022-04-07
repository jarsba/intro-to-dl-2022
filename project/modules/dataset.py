import os
import torch
from torchvision.io import ImageReadMode
import glob
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import pickle

class ProjectDataset(Dataset):

    def __init__(self, annotations_dir, img_dir, transform=None, target_transform=None, rebuildAnnotations=False):
        self.imagePaths = glob.glob(f'{img_dir}{os.sep}*.jpg')
        self.annotationPaths = glob.glob(f'{annotations_dir}{os.sep}*.txt')
        self.annotationPaths.sort()

        self.annotations_dir = annotations_dir
        self.img_dir = img_dir

        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = None
        if rebuildAnnotations == False:
            try:
                self.img_labels = pd.read_pickle('project/annotations.pickle')
            except FileNotFoundError:
                self.img_labels = self._annotate_images()
        else:
            self.img_labels = self._annotate_images()

    def _annotate_images(self):
        NUM_CLASSES = 14
        annotation_file_labels = []
        imgids = []
        onehots = []
        ann_dict = {}
        for anfi in self.annotationPaths:
            with open(anfi, 'r') as f:
                lines = [l.strip() for l in f.readlines()]
                annotation_file_labels.append(set(lines))
        for i, imfi in enumerate(self.imagePaths):
            onehot = np.zeros(NUM_CLASSES, dtype=int)
            fileid = imfi.split(os.sep)[-1][2:-4] #extract only the number from the filename
            for i,j in enumerate(annotation_file_labels):
                if fileid in j:
                    onehot[i] = 1
            #imgids.append(int(fileid))
            #onehots.append(onehot)
            ann_dict[int(fileid)] = onehot
        #annotate_df = pd.DataFrame({'classes':onehots}, index=imgids)
        #annotate_df = annotate_df.sort_index()
        #annotate_df.to_pickle('project/annotations.pickle')
        pickle.dump(ann_dict, open('project/annotations.pickle', 'wb'))
        return ann_dict

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if idx == 0: #there is no image with index 0, return img1 instead
            return self.__getitem__(1)
        img_path = os.sep.join([self.img_dir, f'im{idx}.jpg'])
        image = read_image(img_path, mode= ImageReadMode.RGB)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_weights(self):
        lbls =  torch.tensor(np.array([*self.img_labels.values()]), dtype=torch.float)
        weights = lbls.sum(0) / lbls.shape[1]
        return weights / weights.sum()

class ProjectTestDataset(Dataset):

    def __init__(self, img_dir='test_data/images') -> None:
        super().__init__()
        self.imgDir = img_dir

    def __getitem__(self, index):
        picture_id = 20000 + index + 1
        path = f'{self.imgDir}{os.sep}im{picture_id}.jpg'
        return read_image(path , mode=ImageReadMode.RGB) 

    def __len__(self):
        return len(glob.glob(f'{self.imgDir}{os.sep}*.jpg'))


if __name__ == "__main__":
    ds = ProjectDataset(img_dir='project/data/images',\
                        annotations_dir='project/data/annotations', rebuildAnnotations=True)