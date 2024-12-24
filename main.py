from src.data.data_loader import MultimodalDataLoader
from src.data.data_cleaner import MultimodalDataCleaner
from src.features.audio_processor import AudioProcessor
from src.features.text_processor import TextProcessor
from src.features.image_processor import ImageProcessor
from src.features.fusion import ModalityFusion

def main():
    # Initialize components
    data_loader = MultimodalDataLoader("config/config.yaml")
    data_cleaner = MultimodalDataCleaner("config/config.yaml")
    
    # Load and clean data
    data = data_loader.load_data("path/to/your/data/folder")
    cleaned_data = data_cleaner.clean_data(data)
    
    # Process each modality
    audio_processor = AudioProcessor("config/config.yaml")
    text_processor = TextProcessor("config/config.yaml")
    image_processor = ImageProcessor("config/config.yaml")
    
    # Extract features
    audio_features = audio_processor.extract_features(cleaned_data['audio'])
    text_features = text_processor.extract_features(cleaned_data['text'])
    image_features = image_processor.extract_features(cleaned_data['image'])
    
    # Fuse features
    fusion = ModalityFusion("config/config.yaml")
    fused_features = fusion.late_fusion({
        'audio': audio_features,
        'text': text_features,
        'image': image_features
    })
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 