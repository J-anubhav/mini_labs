import os
import pandas as pd
from PIL import Image
import json
from pathlib import Path
import cv2
import numpy as np

class HandwritingDataProcessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def process_iam_words(self, iam_path):
        """Process IAM handwriting word database"""
        print("Processing IAM Words Dataset...")
        
        # Look for words.txt or similar annotation file
        annotation_files = list(Path(iam_path).glob("**/*words.txt")) + \
                          list(Path(iam_path).glob("**/*annotations*")) + \
                          list(Path(iam_path).glob("**/*.txt"))
        
        if not annotation_files:
            print("No annotation file found, creating from folder structure...")
            return self._process_images_only(iam_path)
            
        # Process with annotations
        data = []
        for ann_file in annotation_files:
            with open(ann_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split()
                if len(parts) >= 9:  # IAM format
                    word_id = parts[0]
                    transcription = parts[-1]
                    
                    # Find corresponding image
                    img_path = self._find_image(iam_path, word_id)
                    if img_path and os.path.exists(img_path):
                        data.append({
                            'image_path': str(img_path),
                            'text': transcription,
                            'dataset': 'iam_words'
                        })
        
        return data
    
    def process_handwritten_names(self, names_path):
        """Process 400k handwritten names dataset"""
        print("Processing Handwritten Names Dataset...")
        
        data = []
        names_path = Path(names_path)
        
        # Look for CSV file with annotations
        csv_files = list(names_path.glob("**/*.csv"))
        
        if csv_files:
            df = pd.read_csv(csv_files[0])
            for _, row in df.iterrows():
                if 'filename' in row and 'text' in row:
                    img_path = names_path / row['filename']
                elif 'image' in row and 'label' in row:
                    img_path = names_path / row['image']
                    row['text'] = row['label']
                else:
                    continue
                    
                if img_path.exists():
                    data.append({
                        'image_path': str(img_path),
                        'text': str(row['text']),
                        'dataset': 'handwritten_names'
                    })
        else:
            # Process images and try to extract text from filenames
            return self._process_images_only(names_path)
            
        return data
    
    def _process_images_only(self, folder_path):
        """Fallback: process images when no annotations available"""
        print(f"Processing images without annotations in {folder_path}")
        
        data = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for img_path in Path(folder_path).rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Try to extract text from filename
                filename = img_path.stem
                
                # Common patterns in dataset filenames
                if '_' in filename:
                    potential_text = filename.split('_')[-1]
                elif '-' in filename:
                    potential_text = filename.split('-')[-1]
                else:
                    potential_text = filename
                
                # Clean up the text
                potential_text = ''.join(c for c in potential_text if c.isalnum() or c.isspace())
                
                if len(potential_text) > 0 and len(potential_text) < 50:  # Reasonable text length
                    data.append({
                        'image_path': str(img_path),
                        'text': potential_text,
                        'dataset': 'extracted_from_filename'
                    })
        
        return data
    
    def _find_image(self, base_path, word_id):
        """Find image file for given word ID"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        for ext in image_extensions:
            # Try different naming patterns
            patterns = [
                f"{word_id}{ext}",
                f"{word_id.replace('-', '_')}{ext}",
                f"*{word_id}*{ext}"
            ]
            
            for pattern in patterns:
                matches = list(Path(base_path).rglob(pattern))
                if matches:
                    return matches[0]
        return None
    
    def create_instruction_dataset(self, data_list):
        """Convert to instruction-following format for Qwen2-VL"""
        instructions = []
        
        for item in data_list:
            instruction = {
                "instruction": "Extract all text from this handwritten image.",
                "input": item['image_path'],
                "output": item['text'],
                "dataset_source": item['dataset']
            }
            instructions.append(instruction)
        
        return instructions
    
    def validate_images(self, data_list):
        """Validate that all images can be opened"""
        valid_data = []
        
        for item in data_list:
            try:
                img = Image.open(item['image_path'])
                img.verify()  # Verify image integrity
                
                # Check image dimensions (minimum size)
                img = Image.open(item['image_path'])
                if img.width < 10 or img.height < 10:
                    continue
                    
                valid_data.append(item)
            except Exception as e:
                print(f"Invalid image {item['image_path']}: {e}")
                continue
                
        print(f"Valid images: {len(valid_data)} / {len(data_list)}")
        return valid_data
    
    def split_dataset(self, data_list, train_ratio=0.8, val_ratio=0.1):
        """Split dataset into train/validation/test"""
        import random
        random.shuffle(data_list)
        
        total = len(data_list)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = data_list[:train_end]
        val_data = data_list[train_end:val_end]
        test_data = data_list[val_end:]
        
        return train_data, val_data, test_data
    
    def process_all_datasets(self):
        """Main processing function"""
        all_data = []
        
        # Process IAM words
        iam_path = self.dataset_path / "iam_words"
        if iam_path.exists():
            iam_data = self.process_iam_words(iam_path)
            all_data.extend(iam_data)
        
        # Process handwritten names
        names_path = self.dataset_path / "handwritten_names"
        if names_path.exists():
            names_data = self.process_handwritten_names(names_path)
            all_data.extend(names_data)
        
        print(f"Total samples collected: {len(all_data)}")
        
        # Validate images
        valid_data = self.validate_images(all_data)
        
        # Create instruction format
        instruction_data = self.create_instruction_dataset(valid_data)
        
        # Split dataset
        train_data, val_data, test_data = self.split_dataset(instruction_data)
        
        # Save splits
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            output_file = self.output_path / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {len(split_data)} samples to {output_file}")
        
        return splits

# Usage
if __name__ == "__main__":
    processor = HandwritingDataProcessor(
        dataset_path="datasets",
        output_path="processed_data"
    )
    
    splits = processor.process_all_datasets()
    
    print("\nDataset Statistics:")
    for split_name, split_data in splits.items():
        print(f"{split_name}: {len(split_data)} samples")