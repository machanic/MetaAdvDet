from torchvision.transforms import transforms


def get_preprocessor(input_channels=3, input_size=None):
    if input_channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.456]
        std = [0.224]
    normalizer = transforms.Normalize(mean=mean, std=std)
    if input_size is not None:
        preprocess_transform = transforms.Compose([
            transforms.Resize(size=input_size),
            transforms.ToTensor(),
            normalizer
        ])
    else:
        preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            normalizer
        ])
    return preprocess_transform
