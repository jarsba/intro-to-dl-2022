import os
import torch
from torchvision.io import ImageReadMode
import glob
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np

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
            imgids.append(int(fileid))
            onehots.append(onehot)
        annotate_df = pd.DataFrame({'classes':onehots}, index=imgids)
        annotate_df = annotate_df.sort_index()
        annotate_df.to_pickle('project/annotations.pickle')
        return annotate_df

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if idx == 0: #there is no image with index 0, return img1 instead
            return self.__getitem__(1)
        img_path = os.sep.join([self.img_dir, f'im{idx}.jpg'])
        image = read_image(img_path, mode= ImageReadMode.RGB)
        label = self.img_labels.iloc[idx].values[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == "__main__":
    ds = ProjectDataset(img_dir='project/data/images',\
                        annotations_dir='project/data/annotations', rebuildAnnotations=True)