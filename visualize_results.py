#!/usr/bin/env python3
"""
Quick visualization of generated SEM images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def visualize_results(dataset_dir="demo_output/dataset"):
    """Visualize the generated SEM images and masks"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Dataset directory {dataset_dir} not found!")
        return
    
    # Load metadata
    metadata_file = dataset_path / "generation_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"📊 Dataset Info:")
        print(f"  Method: {metadata['generation_method']}")
        print(f"  Image size: {metadata['image_size']}")
        print(f"  Total images: {metadata['total_images']}")
        print()
    
    # Find all image files
    image_files = sorted(list(dataset_path.glob("sem_synthetic_*.png")))
    mask_files = sorted(list(dataset_path.glob("mask_*.png")))
    
    if not image_files:
        print("No generated images found!")
        return
    
    # Show first few results
    num_to_show = min(3, len(image_files))
    
    fig, axes = plt.subplots(2, num_to_show, figsize=(num_to_show * 4, 8))
    if num_to_show == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_to_show):
        # Load images
        img = cv2.imread(str(image_files[i]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_files[i]), cv2.IMREAD_GRAYSCALE)
        
        # Show synthetic SEM image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f'Synthetic SEM {i+1}')
        axes[0, i].axis('off')
        
        # Show mask
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Material Mask {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Generated SEM Images and Material Masks", y=1.02, fontsize=16)
    
    # Save the visualization
    output_path = dataset_path / "visualization_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📸 Visualization saved to: {output_path}")
    
    plt.show()
    
    # Print statistics
    print(f"📈 Statistics:")
    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        material_ratio = np.sum(mask > 127) / mask.size
        print(f"  Image {i+1}: Material ratio = {material_ratio:.2%}")

def compare_with_original():
    """Compare generated images with original SEM samples"""
    # Load original sample
    sample_files = ['1_030.tif', '1-50-1.bmp', '3-4-02.tif']
    available_samples = [f for f in sample_files if Path(f).exists()]
    
    if not available_samples:
        print("No original samples found for comparison")
        return
    
    # Load first sample
    original = cv2.imread(available_samples[0], cv2.IMREAD_GRAYSCALE)
    
    # Load generated
    generated_file = Path("demo_output/dataset/sem_synthetic_000.png")
    if generated_file.exists():
        generated = cv2.imread(str(generated_file), cv2.IMREAD_GRAYSCALE)
        
        # Resize for comparison
        h, w = generated.shape
        original_resized = cv2.resize(original, (w, h))
        
        # Show comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_resized, cmap='gray')
        axes[0].set_title('Original SEM Sample')
        axes[0].axis('off')
        
        axes[1].imshow(generated, cmap='gray')
        axes[1].set_title('Generated Synthetic SEM')
        axes[1].axis('off')
        
        # Show histogram comparison
        axes[2].hist(original_resized.flatten(), bins=50, alpha=0.7, label='Original', density=True)
        axes[2].hist(generated.flatten(), bins=50, alpha=0.7, label='Generated', density=True)
        axes[2].set_xlabel('Pixel Intensity')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Intensity Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.suptitle("Original vs Generated SEM Images", y=1.02, fontsize=16)
        plt.savefig("demo_output/comparison.png", dpi=150, bbox_inches='tight')
        print("📊 Comparison saved to: demo_output/comparison.png")
        plt.show()

if __name__ == "__main__":
    print("🔬 SEM Generation Results Visualization")
    print("=" * 50)
    
    # Visualize results
    visualize_results()
    
    print("\n" + "=" * 50)
    
    # Compare with original
    compare_with_original()
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Check the demo_output directory for all generated files.") 