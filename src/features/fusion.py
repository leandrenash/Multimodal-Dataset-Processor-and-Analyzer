import torch
import torch.nn as nn
from typing import Dict, List, Optional
import yaml

class ModalityFusion:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def late_fusion(self, features: Dict[str, torch.Tensor], 
                   weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        Perform late fusion of features from different modalities
        """
        if weights is None:
            weights = {modality: 1.0 for modality in features.keys()}
            
        # Normalize features for each modality
        normalized_features = {}
        for modality, feat in features.items():
            feat_norm = torch.nn.functional.normalize(feat, p=2, dim=1)
            normalized_features[modality] = feat_norm * weights[modality]
            
        # Concatenate all features
        fused_features = torch.cat(list(normalized_features.values()), dim=1)
        return fused_features
    
    def attention_fusion(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform attention-based fusion of features
        """
        # Convert features to same dimension if needed
        feature_dim = 512  # Common dimension for all modalities
        projected_features = {}
        
        for modality, feat in features.items():
            if feat.size(1) != feature_dim:
                projection = nn.Linear(feat.size(1), feature_dim).to(self.device)
                feat = projection(feat)
            projected_features[modality] = feat
            
        # Stack features for attention
        stacked_features = torch.stack(list(projected_features.values()), dim=1)
        
        # Self-attention
        attention_weights = torch.softmax(
            torch.matmul(stacked_features, stacked_features.transpose(-2, -1)) 
            / torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)),
            dim=-1
        )
        
        # Apply attention and combine
        fused_features = torch.matmul(attention_weights, stacked_features)
        fused_features = fused_features.mean(dim=1)  # Average across modalities
        
        return fused_features 