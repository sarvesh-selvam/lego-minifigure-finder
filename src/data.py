import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



def _tfms(image_size=224):
    train = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_ = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return {"train": train, "val": eval_, "test": eval_}



class CSVDataset(Dataset):
    def __init__(self, csv_path, img_dir, transforms=None):
        self.df = pd.read_csv(csv_path)
        if not {"filename","label"}.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must have 'filename' and 'label' columns.")
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, str(row["filename"]))
        img = Image.open(path).convert("RGB")
        y = int(row["label"])
        if self.transforms: img = self.transforms(img)
        return img, y



def make_loaders(
    data_dir="data",
    images_subdir="images",
    batch_size=32,
    num_workers=0,         # macOS: start at 0; bump later if stable
    image_size=224,
):
    csvs = {
        "train": os.path.join(data_dir, "train.csv"),
        "val":   os.path.join(data_dir, "val.csv"),
        "test":  os.path.join(data_dir, "test.csv"),
    }
    img_dir = os.path.join(data_dir, images_subdir)
    tf = _tfms(image_size)

    ds = {
        split: CSVDataset(csvs[split], img_dir, transforms=tf["train" if split=="train" else "val"])
        for split in ["train","val","test"]
    }

    loaders = {
        "train": DataLoader(ds["train"], batch_size=batch_size, shuffle=True,
                            num_workers=num_workers),
        "val":   DataLoader(ds["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
        "test":  DataLoader(ds["test"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers),
    }
    return loaders
