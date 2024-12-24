import torch
import torch.nn as nn
from PIL import Image
from typing import List
import yaml
from torchvision import transforms
from clip import load

class ImageProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.batch_size = self.config['models']['image']['batch_size']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load CLIP model
        self.model, self.preprocess = load("ViT-B/32", device=self.device)
        
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract features from images using CLIP
        """
        features = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess images
            processed_images = torch.stack([
                self.preprocess(img) for img in batch_images
            ]).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model.encode_image(processed_images)
                features.append(batch_features.cpu())
                
        return torch.cat(features, dim=0) 