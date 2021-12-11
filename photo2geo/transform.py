from torchvision import transforms

transform_dict = {
    "train": transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.7, 1.2)),
            transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0)),
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
