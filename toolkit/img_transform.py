from torchvision.transforms import transforms


def get_preprocessor(input_size=(32,32), input_channels=3):
    if input_channels == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.456]
        std = [0.224]
    normalizer = transforms.Normalize(mean=mean, std=std)
    preprocess_transform = transforms.Compose([
        transforms.Resize(size=input_size),
        transforms.ToTensor(),
        normalizer
    ])
    return preprocess_transform
