import pandas as pd
import PIL

from os.path import join
from torch.utils.data.dataset import Dataset


class FlickrDataset(Dataset):
    def __init__(self, annotations_path, images_path, transform):
        self.images_path = images_path
        self.df = pd.read_csv(annotations_path,
                              sep=" ",
                              header=None,
                              names=["file_name", "logo", 1, 2, 3, 4, 5, 6])
        self.labels_dict = {label: idx for
                            idx, label in
                            enumerate(self.df["logo"].unique())}
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.loc[index, :]
        fpath = join(self.images_path, row["file_name"])
        im = PIL.Image.open(fpath)
        return self.transform(im), self.labels_dict[row["logo"]]

    def __len__(self):
        return self.df.shape[0]
