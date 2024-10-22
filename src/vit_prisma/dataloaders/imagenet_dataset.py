import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
 
from torchvision import transforms

import json

def extract_and_parse_text_labels(imagenet_index):
    text_labels = []
    for value in imagenet_index.values():
        if isinstance(value, list) and len(value) > 1:
            # Extract the label, replace underscores with spaces, and capitalize each word
            parsed_label = ' '.join(word.capitalize() for word in value[1].replace('_', ' ').split())
            text_labels.append(parsed_label)
    return text_labels

def get_imagenet_index_to_name(imagenet_path=None):
    ind_to_name = {}

    if imagenet_path:
        json_file_path = os.path.join(imagenet_path, "imagenet_index.json")
        
    
        with open(json_file_path, 'r') as file:
            index_data = json.load(file)
            
            for index, item in index_data.items():
                # Assuming the JSON structure is like {"0": ["n01440764", "tench"], ...}
                # where the first element is the synset ID and the second is the class name
                class_name = item[1]
                ind_to_name[int(index)] = class_name

        return ind_to_name
    else:
        from vit_prisma.dataloaders.imagenet_index import imagenet_index
        return imagenet_index

def get_imagenet_text_labels():
    from vit_prisma.dataloaders.imagenet_index import imagenet_index
    return extract_and_parse_text_labels(imagenet_index)

def get_imagenet_transforms_clip(model_name, size=224):
    if 'dino' in model_name.lower():
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained(model_name)
    elif 'clip' in model_name.lower():
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(model_name).image_processor
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean,
                         std=processor.image_std),
    ])

def load_imagenet(preprocess_transform, dataset_path, dataset_type='imagenet1k-val'):
    if dataset_type == 'imagenet1k-val':
        print(f"Loading dataset type: {dataset_type}") 
        # Imagenet-specific logic
        from vit_prisma.utils.data_utils.imagenet.imagenet_utils import setup_imagenet_paths
        from vit_prisma.dataloaders.imagenet_dataset import get_imagenet_transforms_clip, ImageNetValidationDataset
        imagenet_paths = setup_imagenet_paths(dataset_path)
        val_data = ImageNetValidationDataset(imagenet_paths['val'], 
                                        imagenet_paths['label_strings'], 
                                        imagenet_paths['val_labels'], 
                                        preprocess_transform,
        )
        print(imagenet_paths)
        print(f"Validation data length: {len(val_data)}") 
        return val_data

class ImageNetValidationDataset(torch.utils.data.Dataset):
        def __init__(self, images_dir, imagenet_class_index, validation_labels,  transform=None, return_index=False):
            self.images_dir = images_dir
            self.transform = transform
            self.labels = {}
            self.return_index = return_index

            # load label code to index
            self.label_to_index = {}
    
            with open(imagenet_class_index, 'r') as file:
                # Iterate over each line in the file
                for line_num, line in enumerate(file):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    code = parts[0]
                    self.label_to_index[code] = line_num


            # load image name to label code
            self.image_name_to_label = {}

            # Open the CSV file for reading
            with open(validation_labels, mode='r') as csv_file:
                # Create a CSV reader object
                csv_reader = csv.DictReader(csv_file)
                
                # Iterate over each row in the CSV
                for row in csv_reader:
                    # Split the PredictionString by spaces and take the first element
                    first_prediction = row['PredictionString'].split()[0]
                    # Map the ImageId to the first part of the PredictionString
                    self.image_name_to_label[row['ImageId']] = first_prediction



            self.image_names = list(os.listdir(self.images_dir))

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, idx):


            img_path = os.path.join(self.images_dir, self.image_names[idx])
           # print(img_path)
            image = Image.open(img_path).convert('RGB')

            img_name = os.path.basename(os.path.splitext(self.image_names[idx])[0])

            label_i = self.label_to_index[self.image_name_to_label[img_name]]

            if self.transform:
                image = self.transform(image)

            if self.return_index:
                return image, label_i, idx
            else:
                return image, label_i