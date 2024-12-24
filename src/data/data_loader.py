from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import librosa
import yaml
from typing import Dict, Union, List

class MultimodalDataLoader:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.supported_formats = self.config['data']['supported_formats']
    
    def load_data(self, data_path: Union[str, Path]) -> Dict:
        """
        Load multimodal data from a directory or file
        """
        data_path = Path(data_path)
        data = {
            'image': [],
            'audio': [],
            'text': [],
            'metadata': {}
        }
        
        if data_path.is_file():
            return self._load_single_file(data_path)
        
        # Recursively load all supported files
        for file_path in data_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            file_type = self._get_file_type(file_path)
            if file_type:
                loaded_data = self._load_single_file(file_path)
                data[file_type].append(loaded_data)
                
        return data
    
    def _get_file_type(self, file_path: Path) -> str:
        """
        Determine the type of file based on extension
        """
        suffix = file_path.suffix.lower()
        for data_type, extensions in self.supported_formats.items():
            if suffix in extensions:
                return data_type
        return None
    
    def _load_single_file(self, file_path: Path) -> Union[Image.Image, torch.Tensor, str]:
        """
        Load a single file based on its type
        """
        try:
            file_type = self._get_file_type(file_path)
            
            if file_type == 'image':
                return Image.open(file_path).convert('RGB')
            
            elif file_type == 'audio':
                try:
                    audio, sr = librosa.load(
                        file_path, 
                        sr=self.config['preprocessing']['audio']['sample_rate']
                    )
                    return torch.from_numpy(audio).float()
                except Exception as e:
                    print(f"Error loading audio file {file_path}: {e}")
                    return None
            
            elif file_type == 'text':
                try:
                    if file_path.suffix == '.csv':
                        return pd.read_csv(file_path)
                    elif file_path.suffix == '.json':
                        return pd.read_json(file_path)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return f.read()
                except Exception as e:
                    print(f"Error loading text file {file_path}: {e}")
                    return None
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None 