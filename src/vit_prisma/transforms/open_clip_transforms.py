from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def _convert_to_rgb(image):
    return image.convert('RGB')

def get_clip_val_transforms(
    image_size=224,
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711]
):
    return transforms.Compose([
        transforms.Resize(size=image_size, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.CenterCrop(size=(image_size, image_size)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
