"""
Mask Generator for Material Distribution
Generates various types of masks for defining material boundaries
"""

import numpy as np
import cv2
from pathlib import Path
import random
from scipy import ndimage
import matplotlib.pyplot as plt


class MaskGenerator:
    """Generate various types of masks for material distribution"""
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_perlin_noise(self, shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
        """Generate Perlin noise for natural-looking boundaries"""
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)
        
        def lerp(t, a, b):
            return a + t * (b - a)
        
        def grad(hash, x, y):
            h = hash & 15
            u = x if h < 8 else y
            v = y if h < 4 else (x if h == 12 or h == 14 else 0)
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        
        # Simplified Perlin noise implementation
        height, width = shape
        noise = np.zeros((height, width))
        
        # Generate multiple octaves
        for octave in range(octaves):
            freq = 2 ** octave
            amp = persistence ** octave
            
            # Generate noise for this octave
            octave_noise = np.random.rand(height // freq + 2, width // freq + 2)
            octave_noise = cv2.resize(octave_noise, (width, height), interpolation=cv2.INTER_LINEAR)
            
            noise += amp * octave_noise
        
        # Normalize
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        return noise
    
    def generate_voronoi_mask(self, shape, num_seeds=10, material_ratio=0.5):
        """Generate Voronoi diagram based mask"""
        height, width = shape
        
        # Generate random seed points
        seeds = []
        for _ in range(num_seeds):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            material = random.choice([0, 1])
            seeds.append((x, y, material))
        
        # Create distance map for each seed
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                min_dist = float('inf')
                closest_material = 0
                
                for seed_x, seed_y, material in seeds:
                    dist = np.sqrt((x - seed_x)**2 + (y - seed_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_material = material
                
                mask[y, x] = closest_material * 255
        
        return mask
    
    def generate_perlin_mask(self, shape, threshold=0.5, scale=100, octaves=4):
        """Generate mask using Perlin noise"""
        noise = self.generate_perlin_noise(shape, scale, octaves)
        mask = (noise > threshold).astype(np.uint8) * 255
        return mask
    
    def generate_cellular_automata_mask(self, shape, initial_density=0.45, iterations=5):
        """Generate mask using cellular automata"""
        height, width = shape
        
        # Initialize with random noise
        mask = np.random.rand(height, width) < initial_density
        
        # Apply cellular automata rules
        for _ in range(iterations):
            new_mask = np.zeros_like(mask)
            
            for y in range(height):
                for x in range(width):
                    # Count neighbors
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < height and 0 <= nx < width:
                                if mask[ny, nx]:
                                    neighbors += 1
                    
                    # Apply rules
                    if mask[y, x]:
                        new_mask[y, x] = neighbors >= 3
                    else:
                        new_mask[y, x] = neighbors >= 5
            
            mask = new_mask
        
        return (mask * 255).astype(np.uint8)
    
    def generate_fractal_mask(self, shape, fractal_type='mandelbrot', max_iter=100):
        """Generate fractal-based mask"""
        height, width = shape
        mask = np.zeros((height, width))
        
        if fractal_type == 'mandelbrot':
            # Mandelbrot set
            xmin, xmax = -2.5, 1.5
            ymin, ymax = -2.0, 2.0
            
            for y in range(height):
                for x in range(width):
                    # Map pixel to complex plane
                    real = xmin + (xmax - xmin) * x / width
                    imag = ymin + (ymax - ymin) * y / height
                    c = complex(real, imag)
                    
                    # Iterate
                    z = 0
                    for i in range(max_iter):
                        if abs(z) > 2:
                            break
                        z = z*z + c
                    
                    mask[y, x] = i
        
        elif fractal_type == 'julia':
            # Julia set
            c = complex(-0.7, 0.27015)
            xmin, xmax = -2.0, 2.0
            ymin, ymax = -2.0, 2.0
            
            for y in range(height):
                for x in range(width):
                    real = xmin + (xmax - xmin) * x / width
                    imag = ymin + (ymax - ymin) * y / height
                    z = complex(real, imag)
                    
                    for i in range(max_iter):
                        if abs(z) > 2:
                            break
                        z = z*z + c
                    
                    mask[y, x] = i
        
        # Threshold and convert to binary
        threshold = np.percentile(mask, 50)  # Use median as threshold
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        return binary_mask
    
    def generate_layered_mask(self, shape, num_layers=3, layer_thickness=20):
        """Generate layered/striped mask pattern"""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Create horizontal layers with some randomness
        current_y = 0
        layer_id = 0
        
        while current_y < height:
            # Random layer thickness
            thickness = layer_thickness + random.randint(-5, 5)
            thickness = max(5, min(thickness, height - current_y))
            
            # Fill layer
            material = layer_id % 2
            mask[current_y:current_y + thickness, :] = material * 255
            
            current_y += thickness
            layer_id += 1
        
        # Add some noise to make it more natural
        noise = np.random.rand(height, width) * 0.1
        mask = mask.astype(np.float32) / 255.0
        mask += noise
        mask = np.clip(mask, 0, 1)
        mask = (mask > 0.5).astype(np.uint8) * 255
        
        return mask
    
    def generate_blob_mask(self, shape, num_blobs=5, blob_size_range=(50, 150)):
        """Generate mask with blob-like regions"""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for _ in range(num_blobs):
            # Random blob center
            center_x = random.randint(0, width - 1)
            center_y = random.randint(0, height - 1)
            
            # Random blob size
            blob_size = random.randint(*blob_size_range)
            
            # Random blob shape (ellipse)
            angle = random.uniform(0, 360)
            axes = (random.randint(blob_size//2, blob_size), 
                   random.randint(blob_size//2, blob_size))
            
            # Draw filled ellipse
            cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)
        
        return mask
    
    def generate_network_mask(self, shape, num_nodes=20, connection_prob=0.3, line_thickness=5):
        """Generate network/crack-like mask pattern"""
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Generate random nodes
        nodes = []
        for _ in range(num_nodes):
            x = random.randint(line_thickness, width - line_thickness)
            y = random.randint(line_thickness, height - line_thickness)
            nodes.append((x, y))
        
        # Connect nodes with probability
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if random.random() < connection_prob:
                    pt1 = nodes[i]
                    pt2 = nodes[j]
                    cv2.line(mask, pt1, pt2, 255, line_thickness)
        
        return mask
    
    def smooth_mask(self, mask, kernel_size=5, iterations=1):
        """Smooth mask boundaries"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        for _ in range(iterations):
            # Morphological operations to smooth
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Gaussian blur
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            
            # Re-threshold
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def adjust_material_ratio(self, mask, target_ratio=0.5):
        """Adjust mask to achieve target material ratio"""
        current_ratio = np.sum(mask > 127) / mask.size
        
        if current_ratio < target_ratio:
            # Need more white pixels
            threshold = 127 - int((target_ratio - current_ratio) * 127)
        else:
            # Need fewer white pixels
            threshold = 127 + int((current_ratio - target_ratio) * 127)
        
        threshold = max(0, min(255, threshold))
        _, adjusted_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        
        return adjusted_mask
    
    def generate_mask(self, shape, mask_type='perlin', **kwargs):
        """Generate mask of specified type"""
        if mask_type == 'perlin':
            return self.generate_perlin_mask(shape, **kwargs)
        elif mask_type == 'voronoi':
            return self.generate_voronoi_mask(shape, **kwargs)
        elif mask_type == 'cellular':
            return self.generate_cellular_automata_mask(shape, **kwargs)
        elif mask_type == 'fractal':
            return self.generate_fractal_mask(shape, **kwargs)
        elif mask_type == 'layered':
            return self.generate_layered_mask(shape, **kwargs)
        elif mask_type == 'blob':
            return self.generate_blob_mask(shape, **kwargs)
        elif mask_type == 'network':
            return self.generate_network_mask(shape, **kwargs)
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
    
    def generate_multiple_masks(self, shape, count=10, mask_types=None, output_dir=None):
        """Generate multiple masks of different types"""
        if mask_types is None:
            mask_types = ['perlin', 'voronoi', 'cellular', 'fractal', 'layered', 'blob', 'network']
        
        masks = []
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        for i in range(count):
            mask_type = random.choice(mask_types)
            mask = self.generate_mask(shape, mask_type)
            
            # Apply smoothing and ratio adjustment
            mask = self.smooth_mask(mask, kernel_size=3)
            mask = self.adjust_material_ratio(mask, target_ratio=0.4 + random.random() * 0.2)
            
            masks.append(mask)
            
            if output_dir:
                filename = output_dir / f"mask_{i:03d}_{mask_type}.png"
                cv2.imwrite(str(filename), mask)
        
        return masks
    
    def visualize_mask(self, mask, title="Generated Mask"):
        """Visualize a single mask"""
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.show()
    
    def visualize_multiple_masks(self, masks, titles=None, save_path=None):
        """Visualize multiple masks in a grid"""
        n_masks = len(masks)
        cols = min(4, n_masks)
        rows = (n_masks + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, mask in enumerate(masks):
            if i < len(axes):
                axes[i].imshow(mask, cmap='gray')
                if titles and i < len(titles):
                    axes[i].set_title(titles[i])
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(masks), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    generator = MaskGenerator(seed=42)
    
    # Generate various types of masks
    shape = (512, 512)
    
    # Generate single masks
    perlin_mask = generator.generate_mask(shape, 'perlin', threshold=0.5, scale=80)
    voronoi_mask = generator.generate_mask(shape, 'voronoi', num_seeds=15)
    fractal_mask = generator.generate_mask(shape, 'fractal', fractal_type='julia')
    
    # Visualize
    generator.visualize_multiple_masks(
        [perlin_mask, voronoi_mask, fractal_mask],
        ['Perlin Noise', 'Voronoi', 'Julia Fractal'],
        'sample_masks.png'
    )
    
    # Generate multiple masks
    masks = generator.generate_multiple_masks(
        shape, count=20, output_dir="generated_masks"
    )
    
    print(f"Generated {len(masks)} masks and saved to 'generated_masks/' directory") 