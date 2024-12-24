import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import yaml

class MultimodalVisualizer:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.reduction_method = self.config['visualization']['reduction_method']
        self.n_components = self.config['visualization']['n_components']
        
    def reduce_dimensions(self, 
                         features: torch.Tensor,
                         method: Optional[str] = None) -> np.ndarray:
        """
        Reduce dimensionality of features using specified method
        """
        try:
            method = method or self.reduction_method
            
            # Convert to CPU if on GPU
            if isinstance(features, torch.Tensor):
                features = features.cpu()
            
            features_np = features.numpy() if isinstance(features, torch.Tensor) else features
            
            # Check for NaN values
            if np.isnan(features_np).any():
                print("Warning: NaN values detected in features")
                features_np = np.nan_to_num(features_np)
            
            if method.lower() == 'umap':
                reducer = umap.UMAP(
                    n_components=self.n_components,
                    random_state=42  # For reproducibility
                )
            elif method.lower() == 'tsne':
                reducer = TSNE(
                    n_components=self.n_components,
                    random_state=42
                )
            elif method.lower() == 'pca':
                reducer = PCA(n_components=self.n_components)
            else:
                raise ValueError(f"Unsupported reduction method: {method}")
            
            return reducer.fit_transform(features_np)
            
        except Exception as e:
            print(f"Error in dimension reduction: {e}")
            return None
    
    def plot_2d_scatter(self,
                       features: Union[torch.Tensor, np.ndarray],
                       labels: Optional[List[str]] = None,
                       title: str = "Feature Space Visualization",
                       interactive: bool = True) -> None:
        """
        Create 2D scatter plot of features
        """
        if isinstance(features, torch.Tensor):
            features = features.numpy()
            
        if features.shape[1] > 2:
            features = self.reduce_dimensions(features)
            
        if interactive:
            fig = px.scatter(
                x=features[:, 0],
                y=features[:, 1],
                color=labels if labels else None,
                title=title,
                labels={'x': 'Component 1', 'y': 'Component 2'}
            )
            fig.show()
        else:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                features[:, 0],
                features[:, 1],
                c=range(len(features)) if labels is None else labels,
                cmap='viridis'
            )
            if labels is not None:
                plt.colorbar(scatter)
            plt.title(title)
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.show()
    
    def plot_modality_comparison(self,
                               features: Dict[str, torch.Tensor],
                               title: str = "Modality Comparison") -> None:
        """
        Compare features from different modalities
        """
        fig = plt.figure(figsize=(15, 5))
        n_modalities = len(features)
        
        for idx, (modality, feat) in enumerate(features.items(), 1):
            reduced_feat = self.reduce_dimensions(feat)
            
            plt.subplot(1, n_modalities, idx)
            plt.scatter(reduced_feat[:, 0], reduced_feat[:, 1])
            plt.title(f"{modality} Features")
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def plot_attention_weights(self,
                             attention_matrix: torch.Tensor,
                             modalities: List[str]) -> None:
        """
        Visualize attention weights between modalities
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            attention_matrix.numpy(),
            xticklabels=modalities,
            yticklabels=modalities,
            annot=True,
            fmt='.2f',
            cmap='viridis'
        )
        plt.title('Cross-Modal Attention Weights')
        plt.show()
    
    def plot_feature_distributions(self,
                                 features: Dict[str, torch.Tensor]) -> None:
        """
        Plot distribution of feature values for each modality
        """
        fig = plt.figure(figsize=(15, 5))
        n_modalities = len(features)
        
        for idx, (modality, feat) in enumerate(features.items(), 1):
            plt.subplot(1, n_modalities, idx)
            
            # Calculate feature statistics
            feat_mean = feat.mean(dim=0).numpy()
            feat_std = feat.std(dim=0).numpy()
            
            # Plot distribution
            plt.hist(feat_mean, bins=30, alpha=0.7)
            plt.title(f"{modality} Feature Distribution")
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            
        plt.tight_layout()
        plt.show()
    
    def create_interactive_plot(self,
                              features: Dict[str, torch.Tensor],
                              labels: Optional[List[str]] = None) -> None:
        """
        Create interactive plot with Plotly for exploring feature spaces
        """
        fig = go.Figure()
        
        for modality, feat in features.items():
            reduced_feat = self.reduce_dimensions(feat)
            
            fig.add_trace(go.Scatter(
                x=reduced_feat[:, 0],
                y=reduced_feat[:, 1],
                mode='markers',
                name=modality,
                text=labels if labels else None,
                hovertemplate=(
                    f"{modality}<br>" +
                    "x: %{x:.2f}<br>" +
                    "y: %{y:.2f}<br>" +
                    "%{text}<extra></extra>" if labels else None
                )
            ))
            
        fig.update_layout(
            title="Interactive Feature Space Visualization",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode='closest'
        )
        
        fig.show() 