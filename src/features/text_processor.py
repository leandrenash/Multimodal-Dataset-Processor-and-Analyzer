from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union
import yaml

class TextProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        model_name = self.config['models']['text']['name']
        self.batch_size = self.config['models']['text']['batch_size']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def extract_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract features from text using pretrained transformer
        """
        features = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize and move to device
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings as features
                batch_features = outputs.last_hidden_state[:, 0, :]
                features.append(batch_features.cpu())
                
        return torch.cat(features, dim=0) 