import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from typing import List, Union
import yaml

class AudioProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        model_name = self.config['models']['audio']['name']
        self.batch_size = self.config['models']['audio']['batch_size']
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def extract_features(self, audio_list: List[Union[torch.Tensor, np.ndarray]]) -> torch.Tensor:
        """
        Extract features from audio using Wav2Vec2
        """
        features = []
        
        for i in range(0, len(audio_list), self.batch_size):
            batch_audio = audio_list[i:i + self.batch_size]
            
            # Convert numpy arrays to torch tensors if needed
            batch_audio = [
                torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio 
                for audio in batch_audio
            ]
            
            # Process audio
            inputs = self.processor(
                batch_audio, 
                sampling_rate=self.config['preprocessing']['audio']['sample_rate'],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_features = outputs.last_hidden_state.mean(dim=1)
                features.append(batch_features.cpu())
                
        return torch.cat(features, dim=0) 