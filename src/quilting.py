"""
Image Quilting Implementation
Based on Efros & Freeman "Image Quilting for Texture Synthesis and Transfer"
"""

import numpy as np
import cv2
from pathlib import Path
import random
from scipy.ndimage import label
import matplotlib.pyplot as plt


class ImageQuilting:
    """Image Quilting texture synthesis algorithm"""
    
    def __init__(self, patch_size=64, overlap=16, tolerance=0.1):
        """
        Initialize Image Quilting
        
        Args:
            patch_size: Size of patches to use for quilting
            overlap: Overlap between adjacent patches
            tolerance: Tolerance for patch selection (0.1 = top 10%)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.tolerance = tolerance
        
    def load_patches(self, patch_dir):
        """Load patches from directory"""
        patch_dir = Path(patch_dir)
        patches = []
        
        for patch_file in patch_dir.glob("*.png"):
            patch = cv2.imread(str(patch_file), cv2.IMREAD_GRAYSCALE)
            if patch is not None:
                # Ensure patch is large enough
                h, w = patch.shape
                min_size = max(32, self.patch_size)  # Ensure it's at least as large as patch_size
                
                if h < min_size or w < min_size:
                    # Resize patch if too small
                    scale = max(min_size / h, min_size / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    print(f"  Quilting: Resized patch from {w}x{h} to {new_w}x{new_h}")
                
                patches.append(patch.astype(np.float32) / 255.0)
                
        return patches
    
    def extract_patches_from_images(self, images, patch_size=None):
        """Extract overlapping patches from input images"""
        if patch_size is None:
            patch_size = self.patch_size
            
        all_patches = []
        
        for img in images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            img = img.astype(np.float32) / 255.0
            
            h, w = img.shape
            
            # Extract patches with stride
            stride = patch_size // 2
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    all_patches.append(patch)
                    
        return all_patches
    
    def compute_patch_error(self, patch1, patch2, mask=None):
        """Compute SSD error between two patches"""
        diff = patch1 - patch2
        if mask is not None:
            diff = diff * mask
        return np.sum(diff ** 2)
    
    def find_best_patches(self, target_region, patches, num_candidates=None):
        """Find best matching patches for target region"""
        if num_candidates is None:
            num_candidates = max(1, int(len(patches) * self.tolerance))
        
        errors = []
        valid_patches = []
        
        target_h, target_w = target_region.shape
        
        for patch in patches:
            patch_h, patch_w = patch.shape
            
            # Check if patch is large enough
            if patch_h >= target_h and patch_w >= target_w:
                # Extract region of same size
                patch_region = patch[:target_h, :target_w]
                error = self.compute_patch_error(target_region, patch_region)
                errors.append(error)
                valid_patches.append(patch_region)
        
        if not errors:
            return []
            
        # Sort by error and return top candidates
        sorted_indices = np.argsort(errors)
        top_indices = sorted_indices[:num_candidates]
        
        return [valid_patches[i] for i in top_indices]
    
    def compute_overlap_error(self, patch, existing_image, y, x, direction='both'):
        """Compute error in overlap region"""
        patch_h, patch_w = patch.shape
        img_h, img_w = existing_image.shape
        
        total_error = 0
        
        if direction in ['both', 'left'] and x > 0:
            # Left overlap
            overlap_w = min(self.overlap, x, patch_w)
            if overlap_w > 0:
                patch_region = patch[:, :overlap_w]
                img_region = existing_image[y:y+patch_h, x:x+overlap_w]
                
                # Handle size mismatch
                min_h = min(patch_region.shape[0], img_region.shape[0])
                min_w = min(patch_region.shape[1], img_region.shape[1])
                
                if min_h > 0 and min_w > 0:
                    patch_region = patch_region[:min_h, :min_w]
                    img_region = img_region[:min_h, :min_w]
                    total_error += self.compute_patch_error(patch_region, img_region)
        
        if direction in ['both', 'top'] and y > 0:
            # Top overlap
            overlap_h = min(self.overlap, y, patch_h)
            if overlap_h > 0:
                patch_region = patch[:overlap_h, :]
                img_region = existing_image[y:y+overlap_h, x:x+patch_w]
                
                # Handle size mismatch
                min_h = min(patch_region.shape[0], img_region.shape[0])
                min_w = min(patch_region.shape[1], img_region.shape[1])
                
                if min_h > 0 and min_w > 0:
                    patch_region = patch_region[:min_h, :min_w]
                    img_region = img_region[:min_h, :min_w]
                    total_error += self.compute_patch_error(patch_region, img_region)
        
        return total_error
    
    def find_min_cut_path(self, error_surface):
        """Find minimum error path for seam cutting"""
        h, w = error_surface.shape
        
        # Dynamic programming for minimum path
        dp = np.copy(error_surface)
        
        # Fill DP table
        for i in range(1, h):
            for j in range(w):
                # Consider three possible previous positions
                candidates = []
                if j > 0:
                    candidates.append(dp[i-1, j-1])
                candidates.append(dp[i-1, j])
                if j < w-1:
                    candidates.append(dp[i-1, j+1])
                
                dp[i, j] += min(candidates)
        
        # Backtrack to find path
        path = []
        j = np.argmin(dp[-1, :])
        
        for i in range(h-1, -1, -1):
            path.append((i, j))
            
            if i > 0:
                # Find previous j
                candidates = []
                if j > 0:
                    candidates.append((dp[i-1, j-1], j-1))
                candidates.append((dp[i-1, j], j))
                if j < w-1:
                    candidates.append((dp[i-1, j+1], j+1))
                
                _, j = min(candidates)
        
        path.reverse()
        return path
    
    def blend_patches(self, existing_image, new_patch, y, x):
        """Blend new patch with existing image using min-cut"""
        patch_h, patch_w = new_patch.shape
        img_h, img_w = existing_image.shape
        
        # Create copy of existing image
        result = existing_image.copy()
        
        # Determine actual patch size that fits
        actual_h = min(patch_h, img_h - y)
        actual_w = min(patch_w, img_w - x)
        
        if actual_h <= 0 or actual_w <= 0:
            return result
        
        # Crop patch to fit
        patch_cropped = new_patch[:actual_h, :actual_w]
        
        # Simple overlap blending
        if y > 0 and x > 0:
            # Both top and left overlap
            overlap_h = min(self.overlap, y)
            overlap_w = min(self.overlap, x)
            
            # Vertical seam (left overlap)
            if overlap_w > 0:
                for i in range(actual_h):
                    for j in range(overlap_w):
                        alpha = j / overlap_w
                        result[y+i, x+j] = alpha * patch_cropped[i, j] + (1-alpha) * result[y+i, x+j]
            
            # Horizontal seam (top overlap)
            if overlap_h > 0:
                for i in range(overlap_h):
                    for j in range(actual_w):
                        if j >= overlap_w:  # Avoid double blending
                            alpha = i / overlap_h
                            result[y+i, x+j] = alpha * patch_cropped[i, j] + (1-alpha) * result[y+i, x+j]
            
            # No overlap region
            result[y+overlap_h:y+actual_h, x+overlap_w:x+actual_w] = \
                patch_cropped[overlap_h:actual_h, overlap_w:actual_w]
                
        elif y > 0:
            # Only top overlap
            overlap_h = min(self.overlap, y)
            if overlap_h > 0:
                for i in range(overlap_h):
                    alpha = i / overlap_h
                    result[y+i, x:x+actual_w] = alpha * patch_cropped[i, :] + (1-alpha) * result[y+i, x:x+actual_w]
                result[y+overlap_h:y+actual_h, x:x+actual_w] = patch_cropped[overlap_h:actual_h, :]
            else:
                result[y:y+actual_h, x:x+actual_w] = patch_cropped
                
        elif x > 0:
            # Only left overlap
            overlap_w = min(self.overlap, x)
            if overlap_w > 0:
                for j in range(overlap_w):
                    alpha = j / overlap_w
                    result[y:y+actual_h, x+j] = alpha * patch_cropped[:, j] + (1-alpha) * result[y:y+actual_h, x+j]
                result[y:y+actual_h, x+overlap_w:x+actual_w] = patch_cropped[:, overlap_w:actual_w]
            else:
                result[y:y+actual_h, x:x+actual_w] = patch_cropped
        else:
            # No overlap
            result[y:y+actual_h, x:x+actual_w] = patch_cropped
        
        return result
    
    def synthesize_texture(self, patches, output_size, save_progress=False, output_dir=None):
        """Main texture synthesis using image quilting"""
        output_h, output_w = output_size
        
        # Initialize output image
        result = np.zeros((output_h, output_w), dtype=np.float32)
        
        if output_dir and save_progress:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Calculate number of patches needed
        step_size = self.patch_size - self.overlap
        num_patches_h = (output_h - self.overlap + step_size - 1) // step_size
        num_patches_w = (output_w - self.overlap + step_size - 1) // step_size
        
        print(f"Synthesizing {num_patches_h}x{num_patches_w} patches...")
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = i * step_size
                x = j * step_size
                
                print(f"Processing patch ({i}, {j})...")
                
                if i == 0 and j == 0:
                    # First patch - random selection
                    patch = random.choice(patches)
                    result = self.blend_patches(result, patch, y, x)
                else:
                    # Find best matching patches
                    best_patches = []
                    min_error = float('inf')
                    
                    for patch in patches:
                        error = self.compute_overlap_error(patch, result, y, x)
                        if error < min_error:
                            min_error = error
                            best_patches = [patch]
                        elif error == min_error:
                            best_patches.append(patch)
                    
                    # Select randomly from best patches
                    if best_patches:
                        selected_patch = random.choice(best_patches)
                        result = self.blend_patches(result, selected_patch, y, x)
                
                # Save progress if requested
                if save_progress and output_dir and (i * num_patches_w + j) % 10 == 0:
                    progress_img = (result * 255).astype(np.uint8)
                    filename = output_dir / f"quilting_progress_{i:02d}_{j:02d}.png"
                    cv2.imwrite(str(filename), progress_img)
        
        return result
    
    def generate_from_patches(self, patch_dir, output_size=(512, 512), 
                            save_progress=False, output_dir=None):
        """Complete pipeline: load patches -> synthesize texture"""
        
        # Load patches
        patches = self.load_patches(patch_dir)
        if not patches:
            raise ValueError(f"No patches found in {patch_dir}")
        
        print(f"Loaded {len(patches)} patches from {patch_dir}")
        
        # Also extract more patches from the loaded patches
        extra_patches = self.extract_patches_from_images(patches, self.patch_size)
        all_patches = patches + extra_patches
        
        print(f"Total patches for synthesis: {len(all_patches)}")
        
        # Synthesize texture
        result = self.synthesize_texture(
            all_patches, output_size, 
            save_progress=save_progress, output_dir=output_dir
        )
        
        # Convert back to uint8
        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        
        return result_uint8


if __name__ == "__main__":
    # Example usage
    quilter = ImageQuilting(patch_size=64, overlap=16, tolerance=0.1)
    
    # Test with patch directory
    patch_dir = "patches/sample_image"  # Adjust path as needed
    if Path(patch_dir).exists():
        result = quilter.generate_from_patches(
            patch_dir, 
            output_size=(512, 512),
            save_progress=True,
            output_dir="quilting_progress"
        )
        
        # Save final result
        cv2.imwrite("quilted_texture.png", result)
        print("Image quilting complete!") 