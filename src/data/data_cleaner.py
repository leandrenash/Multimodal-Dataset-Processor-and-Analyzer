import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
import torch
from PIL import Image
import yaml

class MultimodalDataCleaner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def clean_data(self, data: Dict[str, List]) -> Dict[str, List]:
        """
        Clean and preprocess multimodal data
        """
        try:
            cleaned_data = {
                'image': self._clean_images(data.get('image', [])),
                'audio': self._clean_audio(data.get('audio', [])),
                'text': self._clean_text(data.get('text', [])),
                'metadata': data.get('metadata', {})
            }
            
            # Remove None values from lists
            for modality in cleaned_data:
                if isinstance(cleaned_data[modality], list):
                    cleaned_data[modality] = [
                        x for x in cleaned_data[modality] if x is not None
                    ]
                    
            return cleaned_data
        except Exception as e:
            print(f"Error in data cleaning: {e}")
            return None
    
    def _clean_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Clean and preprocess images
        """
        target_size = self.config['preprocessing']['image']['target_size']
        normalized_images = []
        
        for img in images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(target_size)
            
            # Convert to tensor and normalize
            img_tensor = torch.from_numpy(np.array(img)).float()
            if self.config['preprocessing']['image']['normalize']:
                img_tensor = img_tensor / 255.0
                
            normalized_images.append(img_tensor)
            
        return normalized_images
    
    def _clean_audio(self, audio_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Clean and preprocess audio data
        """
        target_sr = self.config['preprocessing']['audio']['sample_rate']
        target_duration = self.config['preprocessing']['audio']['duration']
        
        cleaned_audio = []
        for audio in audio_list:
            # Ensure consistent duration
            target_length = target_sr * target_duration
            if audio.shape[0] > target_length:
                audio = audio[:target_length]
            else:
                # Pad with zeros if too short
                padding = target_length - audio.shape[0]
                audio = torch.nn.functional.pad(audio, (0, padding))
                
            cleaned_audio.append(audio)
            
        return cleaned_audio
    
    def _clean_text(self, texts: List[str]) -> List[str]:
        """
        Clean and preprocess text data
        """
        max_length = self.config['preprocessing']['text']['max_length']
        lowercase = self.config['preprocessing']['text']['lowercase']
        
        cleaned_texts = []
        for text in texts:
            if lowercase:
                text = text.lower()
            
            # Basic cleaning
            text = text.strip()
            text = ' '.join(text.split())  # Remove extra whitespace
            
            # Truncate if necessary
            if len(text) > max_length:
                text = text[:max_length]
                
            cleaned_texts.append(text)
            
        return cleaned_texts 