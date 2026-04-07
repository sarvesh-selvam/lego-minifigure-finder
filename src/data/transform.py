from torchvision import transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _tfms(image_size=224):
    train = T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomPerspective(distortion_scale=0.3, p=0.4),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        T.RandomGrayscale(p=0.05),
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
