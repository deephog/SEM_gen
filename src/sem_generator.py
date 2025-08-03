"""
Main SEM Image Generator
Integrates all modules to create synthetic SEM images with two materials
"""

import numpy as np
import cv2
from pathlib import Path
import json
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union

from .texture_synthesizer import TextureSynthesizer
from .quilting import ImageQuilting
from .mask_generator import MaskGenerator


class SEMGenerator:
    """Main class for generating synthetic SEM images"""
    
    def __init__(self, output_dir="generated_sem", device=None, seed=None):
        """
        Initialize SEM generator
        
        Args:
            output_dir: Directory to save generated images
            device: Device for neural synthesis (cuda/cpu)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.neural_synthesizer = TextureSynthesizer(device=device)
        self.quilting_synthesizer = ImageQuilting()
        self.mask_generator = MaskGenerator(seed=seed)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.generation_log = []
    
    def generate_material_textures(self, patch_dirs: List[str], output_size: Tuple[int, int] = (512, 512),
                                 method: str = 'mixed', save_intermediate: bool = False) -> List[np.ndarray]:
        """
        Generate textures for different materials
        
        Args:
            patch_dirs: List of directories containing patches for each material
            output_size: Size of generated textures
            method: Generation method ('neural', 'quilting', 'mixed')
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of generated texture images
        """
        textures = []
        
        for i, patch_dir in enumerate(patch_dirs):
            print(f"Generating texture for material {i+1} from {patch_dir}...")
            
            # Create material-specific output directory
            material_dir = self.output_dir / f"material_{i+1}"
            material_dir.mkdir(exist_ok=True)
            
            if method == 'neural':
                texture = self._generate_neural_texture(
                    patch_dir, output_size, save_intermediate, material_dir
                )
            elif method == 'quilting':
                texture = self._generate_quilting_texture(
                    patch_dir, output_size, save_intermediate, material_dir
                )
            elif method == 'mixed':
                # Use neural synthesis as base, enhance with quilting details
                texture = self._generate_mixed_texture(
                    patch_dir, output_size, save_intermediate, material_dir
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            textures.append(texture)
            
            # Save individual texture
            texture_path = material_dir / f"texture_material_{i+1}.png"
            cv2.imwrite(str(texture_path), texture)
        
        return textures
    
    def _generate_neural_texture(self, patch_dir: str, output_size: Tuple[int, int],
                               save_intermediate: bool, material_dir: Path) -> np.ndarray:
        """Generate texture using neural synthesis"""
        progress_dir = material_dir / "neural_progress" if save_intermediate else None
        
        texture = self.neural_synthesizer.generate_from_patches(
            patch_dir,
            output_size=output_size,
            num_iterations=500,
            save_progress=save_intermediate,
            output_dir=progress_dir
        )
        
        return texture
    
    def _generate_quilting_texture(self, patch_dir: str, output_size: Tuple[int, int],
                                 save_intermediate: bool, material_dir: Path) -> np.ndarray:
        """Generate texture using image quilting"""
        progress_dir = material_dir / "quilting_progress" if save_intermediate else None
        
        texture = self.quilting_synthesizer.generate_from_patches(
            patch_dir,
            output_size=output_size,
            save_progress=save_intermediate,
            output_dir=progress_dir
        )
        
        return texture
    
    def _generate_mixed_texture(self, patch_dir: str, output_size: Tuple[int, int],
                              save_intermediate: bool, material_dir: Path) -> np.ndarray:
        """Generate texture using mixed approach"""
        print("  Using mixed approach: neural + quilting...")
        
        # Start with neural synthesis for overall texture
        neural_texture = self._generate_neural_texture(
            patch_dir, output_size, save_intermediate, material_dir
        )
        
        # Enhance with quilting details at smaller scale
        quilting_small = ImageQuilting(patch_size=32, overlap=8)
        quilting_texture = quilting_small.generate_from_patches(
            patch_dir, output_size=output_size
        )
        
        # Blend the two textures
        mixed_texture = self._blend_textures(neural_texture, quilting_texture, alpha=0.7)
        
        if save_intermediate:
            cv2.imwrite(str(material_dir / "neural_base.png"), neural_texture)
            cv2.imwrite(str(material_dir / "quilting_detail.png"), quilting_texture)
            cv2.imwrite(str(material_dir / "mixed_result.png"), mixed_texture)
        
        return mixed_texture
    
    def _blend_textures(self, texture1: np.ndarray, texture2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Blend two textures"""
        # Normalize textures
        t1 = texture1.astype(np.float32) / 255.0
        t2 = texture2.astype(np.float32) / 255.0
        
        # Blend
        blended = alpha * t1 + (1 - alpha) * t2
        
        # Convert back to uint8
        result = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
        
        return result
    
    def generate_masks(self, count: int, image_size: Tuple[int, int], 
                      mask_types: Optional[List[str]] = None) -> List[np.ndarray]:
        """Generate multiple masks for material distribution"""
        print(f"Generating {count} masks of size {image_size}...")
        
        masks_dir = self.output_dir / "masks"
        masks = self.mask_generator.generate_multiple_masks(
            image_size, count=count, mask_types=mask_types, output_dir=masks_dir
        )
        
        return masks
    
    def combine_materials_with_mask(self, material1: np.ndarray, material2: np.ndarray,
                                  mask: np.ndarray, blend_border: bool = True,
                                  border_width: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine two material textures using a mask
        
        Args:
            material1: First material texture
            material2: Second material texture  
            mask: Binary mask (255 = material2, 0 = material1)
            blend_border: Whether to blend at boundaries
            border_width: Width of blending border
            
        Returns:
            Tuple of (combined_image, processed_mask)
        """
        # Ensure same size
        h, w = mask.shape
        mat1 = cv2.resize(material1, (w, h))
        mat2 = cv2.resize(material2, (w, h))
        
        # Create normalized mask
        mask_norm = mask.astype(np.float32) / 255.0
        
        if blend_border and border_width > 0:
            # Create smooth transition at borders
            kernel = np.ones((border_width, border_width), np.uint8)
            
            # Dilate and erode to find border region
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            border_region = mask_dilated - mask_eroded
            
            # Apply Gaussian blur to the mask in border regions
            border_mask = border_region.astype(np.float32) / 255.0
            blurred_mask = cv2.GaussianBlur(mask_norm, (border_width*2+1, border_width*2+1), 0)
            
            # Use blurred mask only in border regions
            final_mask = mask_norm * (1 - border_mask) + blurred_mask * border_mask
        else:
            final_mask = mask_norm
        
        # Combine materials
        combined = mat1.astype(np.float32) * (1 - final_mask[:, :, np.newaxis] if len(mat1.shape) == 3 else 1 - final_mask) + \
                  mat2.astype(np.float32) * (final_mask[:, :, np.newaxis] if len(mat2.shape) == 3 else final_mask)
        
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        # Convert final mask back to binary for output
        output_mask = (final_mask > 0.5).astype(np.uint8) * 255
        
        return combined, output_mask
    
    def add_sem_effects(self, image: np.ndarray, noise_level: float = 0.02,
                       brightness_variation: float = 0.1, contrast_boost: float = 1.2) -> np.ndarray:
        """Add SEM-specific effects to make image more realistic"""
        img = image.astype(np.float32) / 255.0
        
        # Add noise (typical in SEM images)
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, img.shape)
            img = img + noise
        
        # Add brightness variation
        if brightness_variation > 0:
            brightness_map = np.random.normal(1.0, brightness_variation, img.shape)
            img = img * brightness_map
        
        # Enhance contrast
        img = np.clip((img - 0.5) * contrast_boost + 0.5, 0, 1)
        
        # Convert back to uint8
        result = (img * 255).astype(np.uint8)
        
        return result
    
    def generate_sem_dataset(self, material_patch_dirs: List[str], count: int = 10,
                           image_size: Tuple[int, int] = (512, 512),
                           generation_method: str = 'mixed',
                           mask_types: Optional[List[str]] = None,
                           add_effects: bool = True) -> None:
        """
        Generate a complete dataset of synthetic SEM images
        
        Args:
            material_patch_dirs: Directories containing patches for each material
            count: Number of synthetic images to generate
            image_size: Size of generated images
            generation_method: Method for texture generation
            mask_types: Types of masks to use
            add_effects: Whether to add SEM-specific effects
        """
        print(f"Generating SEM dataset with {count} images...")
        print(f"Materials: {len(material_patch_dirs)}")
        print(f"Image size: {image_size}")
        print(f"Method: {generation_method}")
        
        # Generate material textures
        print("\n1. Generating material textures...")
        material_textures = self.generate_material_textures(
            material_patch_dirs, image_size, generation_method, save_intermediate=True
        )
        
        # Generate masks
        print("\n2. Generating masks...")
        masks = self.generate_masks(count, image_size, mask_types)
        
        # Generate synthetic images
        print("\n3. Combining materials to create synthetic SEM images...")
        dataset_dir = self.output_dir / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        generation_metadata = {
            'generation_method': generation_method,
            'image_size': image_size,
            'material_dirs': material_patch_dirs,
            'total_images': count,
            'images': []
        }
        
        for i in range(count):
            print(f"  Generating image {i+1}/{count}...")
            
            # Select materials (cycle if more images than material combinations)
            if len(material_textures) >= 2:
                mat1 = material_textures[0]
                mat2 = material_textures[1]
            else:
                # If only one material, use it for both but with different processing
                mat1 = material_textures[0]
                mat2 = self._vary_texture(material_textures[0])
            
            # Use corresponding mask
            mask = masks[i]
            
            # Combine materials
            combined_image, processed_mask = self.combine_materials_with_mask(
                mat1, mat2, mask, blend_border=True, border_width=3
            )
            
            # Add SEM effects
            if add_effects:
                combined_image = self.add_sem_effects(
                    combined_image,
                    noise_level=random.uniform(0.01, 0.03),
                    brightness_variation=random.uniform(0.05, 0.15),
                    contrast_boost=random.uniform(1.1, 1.4)
                )
            
            # Save files
            image_filename = f"sem_synthetic_{i:03d}.png"
            mask_filename = f"mask_{i:03d}.png"
            
            image_path = dataset_dir / image_filename
            mask_path = dataset_dir / mask_filename
            
            cv2.imwrite(str(image_path), combined_image)
            cv2.imwrite(str(mask_path), processed_mask)
            
            # Record metadata
            generation_metadata['images'].append({
                'id': i,
                'image_file': image_filename,
                'mask_file': mask_filename,
                'materials_used': [0, 1],
                'mask_type': 'generated'
            })
        
        # Save metadata
        metadata_path = dataset_dir / "generation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(generation_metadata, f, indent=2)
        
        print(f"\n✅ Dataset generation complete!")
        print(f"📁 Location: {dataset_dir}")
        print(f"📊 Images: {count}")
        print(f"📋 Metadata: {metadata_path}")
    
    def _vary_texture(self, texture: np.ndarray, variation_strength: float = 0.3) -> np.ndarray:
        """Create a variation of existing texture"""
        varied = texture.astype(np.float32)
        
        # Add some random brightness/contrast variation
        brightness_factor = 1.0 + random.uniform(-variation_strength, variation_strength)
        contrast_factor = 1.0 + random.uniform(-variation_strength/2, variation_strength/2)
        
        varied = varied * brightness_factor
        varied = (varied - 127.5) * contrast_factor + 127.5
        
        # Add slight rotation
        angle = random.uniform(-15, 15)
        if abs(angle) > 1:
            center = (texture.shape[1]//2, texture.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            varied = cv2.warpAffine(varied, rotation_matrix, (texture.shape[1], texture.shape[0]))
        
        return np.clip(varied, 0, 255).astype(np.uint8)
    
    def visualize_generation_result(self, image_path: str, mask_path: str, 
                                  title: str = "Generated SEM Image") -> None:
        """Visualize a generated SEM image with its mask"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original synthetic image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Synthetic SEM Image')
        axes[0].axis('off')
        
        # Material mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Material Distribution Mask')
        axes[1].axis('off')
        
        # Overlay
        overlay = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay_blend = cv2.addWeighted(
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7,
            overlay, 0.3, 0
        )
        axes[2].imshow(cv2.cvtColor(overlay_blend, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay Visualization')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    generator = SEMGenerator(output_dir="generated_sem_dataset", seed=42)
    
    # Example with two material types
    material_dirs = [
        "patches/material_1",  # Adjust paths as needed
        "patches/material_2"
    ]
    
    # Check if directories exist
    if all(Path(d).exists() for d in material_dirs):
        # Generate dataset
        generator.generate_sem_dataset(
            material_patch_dirs=material_dirs,
            count=20,
            image_size=(512, 512),
            generation_method='mixed',
            mask_types=['perlin', 'voronoi', 'cellular'],
            add_effects=True
        )
        
        # Visualize a sample result
        dataset_dir = Path("generated_sem_dataset/dataset")
        if dataset_dir.exists():
            sample_image = dataset_dir / "sem_synthetic_000.png"
            sample_mask = dataset_dir / "mask_000.png"
            
            if sample_image.exists() and sample_mask.exists():
                generator.visualize_generation_result(
                    str(sample_image), str(sample_mask),
                    "Sample Generated SEM Image"
                )
    else:
        print("Please create patch directories with your annotated patches first!")
        print("Use the patch_annotator.py to create patches from your reference images.") 