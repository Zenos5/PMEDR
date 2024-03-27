from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import torchvision
from torchvision import transforms as tvt
import math
import os



class LatexDataset(Dataset):
    def __init__(
        self, data_path, img_path, data_type: str, n_sample: int = None, dataset="100k"
    ):
        super().__init__()
        assert data_type in ["train", "test", "validate"], "Not found data type"
        csv_path = data_path + f"/im2latex_{data_type}.csv"
        df = pd.read_csv(csv_path)
        if n_sample:
            df = df.head(n_sample)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        # print("Begin:",len(self.walker))
        for item in self.walker:
            if os.path.exists(item["image"]) is False:
                print(item["image"], "does not exist")
                self.walker.remove(item)
            # else:
            #     image = torchvision.io.read_image(item["image"])
            #     if (image.shape[1] / 2**4) < 3 or (image.shape[2] / 2**4) < 3:
            #         self.walker.remove(item)
        # print("End:",len(self.walker)
        self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        formula = item["formula"]
        image = torchvision.io.read_image(item["image"])
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]
        return image, formula


class LatexPredictDataset(Dataset):
    def __init__(self, data_path, img_path: str, n_sample: int = None, dataset="100k"):
        super().__init__()
        csv_path = data_path + f"/im2latex_test.csv"
        df = pd.read_csv(csv_path)
        if n_sample:
            df = df.head(n_sample)
        df["image"] = df.image.map(lambda x: img_path + "/" + x)
        self.walker = df.to_dict("records")
        # print("Begin:",len(self.walker))
        for item in self.walker:
            if os.path.isfile(item["image"]) is False:
                print(item["image"], "does not exist")
                self.walker.remove(item)
            # else:
            #     image = torchvision.io.read_image(item["image"])
            #     if (image.shape[1] / 2**4) < 3 or (image.shape[2] / 2**4) < 3:
            #         self.walker.remove(item)
        # print("End:",len(self.walker))
        self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])

    def __len__(self):
        return len(self.walker)

    def __getitem__(self, idx):
        item = self.walker[idx]

        image = torchvision.io.read_image(item["image"])
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]

        return image

class LatexSinglePredictDataset(Dataset):
    def __init__(self, img_file):
        super().__init__()
        self.img_file = img_file
            # else:
            #     image = torchvision.io.read_image(item["image"])
            #     if (image.shape[1] / 2**4) < 3 or (image.shape[2] / 2**4) < 3:
            #         self.walker.remove(item)
        # print("End:",len(self.walker))
        self.transform = tvt.Compose([tvt.Normalize((0.5), (0.5)),])

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        image = torchvision.io.read_image(self.img_file)
        image = image.to(dtype=torch.float)
        image /= image.max()
        image = self.transform(image)  # transform image to [-1, 1]

        return image