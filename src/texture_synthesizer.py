"""
Neural Texture Synthesis Implementation
Based on Gatys et al. "Texture Synthesis Using Convolutional Neural Networks"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


class VGGFeatureExtractor(nn.Module):
    """Extract features using pre-trained VGG19"""
    
    def __init__(self, feature_layers=None):
        super(VGGFeatureExtractor, self).__init__()
        
        if feature_layers is None:
            # Default layers for texture synthesis
            self.feature_layers = {
                '0': 'conv1_1',
                '5': 'conv2_1', 
                '10': 'conv3_1',
                '19': 'conv4_1',
                '28': 'conv5_1'
            }
        else:
            self.feature_layers = feature_layers
            
        self.vgg = models.vgg19(pretrained=True).features
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        """Extract features from multiple layers"""
        features = {}
        
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.feature_layers:
                features[self.feature_layers[name]] = x
                
        return features


class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation"""
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        features = x.view(batch_size * channels, height * width)
        gram = torch.mm(features, features.t())
        return gram.div(batch_size * channels * height * width)


class TextureSynthesizer:
    """Neural texture synthesis using VGG features"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize feature extractor
        self.feature_extractor = VGGFeatureExtractor().to(self.device)
        self.gram_matrix = GramMatrix()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                               std=[1/0.229, 1/0.224, 1/0.225])
        ])
    
    def load_patches(self, patch_dir):
        """Load patches from directory"""
        patch_dir = Path(patch_dir)
        patches = []
        
        for patch_file in patch_dir.glob("*.png"):
            patch = cv2.imread(str(patch_file), cv2.IMREAD_GRAYSCALE)
            if patch is not None:
                patches.append(patch)
                
        return patches
    
    def preprocess_patch(self, patch):
        """Convert grayscale patch to RGB and preprocess"""
        # Convert grayscale to RGB
        if len(patch.shape) == 2:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        else:
            patch_rgb = patch
            
        # Ensure minimum size for VGG processing (VGG needs at least 32x32)
        h, w = patch_rgb.shape[:2]
        min_size = 64  # Safe minimum size
        
        if h < min_size or w < min_size:
            # Resize to minimum size while maintaining aspect ratio
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            patch_rgb = cv2.resize(patch_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"  Resized patch from {w}x{h} to {new_w}x{new_h}")
        
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(patch_rgb)
        tensor = self.transform(pil_image).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def extract_texture_features(self, patches):
        """Extract texture features from patches"""
        all_features = {}
        
        for i, patch in enumerate(patches):
            patch_tensor = self.preprocess_patch(patch)
            features = self.feature_extractor(patch_tensor)
            
            # Compute gram matrices
            gram_features = {}
            for layer_name, feature_map in features.items():
                gram = self.gram_matrix(feature_map)
                gram_features[layer_name] = gram
            
            all_features[f'patch_{i}'] = gram_features
            
        return all_features
    
    def average_gram_matrices(self, patch_features):
        """Average gram matrices across all patches"""
        layer_names = list(patch_features[list(patch_features.keys())[0]].keys())
        averaged_grams = {}
        
        for layer_name in layer_names:
            grams = [features[layer_name] for features in patch_features.values()]
            averaged_grams[layer_name] = torch.stack(grams).mean(dim=0)
            
        return averaged_grams
    
    def synthesize_texture(self, target_size, patch_features, num_iterations=1000, 
                          learning_rate=0.1, save_progress=False, output_dir=None):
        """Synthesize texture using neural texture synthesis"""
        
        # Average the gram matrices from all patches
        target_grams = self.average_gram_matrices(patch_features)
        
        # Initialize generated image with noise
        generated = torch.randn(1, 3, target_size[0], target_size[1], 
                              device=self.device, requires_grad=True)
        
        # Optimizer
        optimizer = optim.LBFGS([generated], lr=learning_rate)
        
        # Loss weights for different layers
        layer_weights = {
            'conv1_1': 1.0,
            'conv2_1': 1.0,
            'conv3_1': 1.0,
            'conv4_1': 1.0,
            'conv5_1': 1.0
        }
        
        iteration = [0]  # Use list to modify in closure
        
        def closure():
            optimizer.zero_grad()
            
            # Extract features from generated image
            generated_features = self.feature_extractor(generated)
            
            total_loss = 0
            
            # Compute texture loss for each layer
            for layer_name, target_gram in target_grams.items():
                generated_gram = self.gram_matrix(generated_features[layer_name])
                layer_loss = nn.MSELoss()(generated_gram, target_gram)
                total_loss += layer_weights.get(layer_name, 1.0) * layer_loss
            
            total_loss.backward()
            
            iteration[0] += 1
            
            if iteration[0] % 100 == 0:
                print(f"Iteration {iteration[0]}: Loss = {total_loss.item():.4f}")
                
                # Save progress if requested
                if save_progress and output_dir:
                    self.save_intermediate_result(generated, iteration[0], output_dir)
            
            return total_loss
        
        # Optimization loop
        for i in range(num_iterations // 50):  # LBFGS takes multiple steps per call
            optimizer.step(closure)
            
        return generated
    
    def save_intermediate_result(self, generated_tensor, iteration, output_dir):
        """Save intermediate result during synthesis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Convert tensor to image
        with torch.no_grad():
            image = generated_tensor.clone().squeeze(0)
            image = self.inverse_transform(image)
            image = torch.clamp(image, 0, 1)
            
            # Convert to numpy
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            
            # Convert to grayscale for SEM-like appearance
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            # Save
            filename = output_dir / f"synthesis_iter_{iteration:04d}.png"
            cv2.imwrite(str(filename), image_gray)
    
    def postprocess_result(self, generated_tensor):
        """Convert result tensor to numpy image"""
        with torch.no_grad():
            # Clamp and denormalize
            image = generated_tensor.clone().squeeze(0)
            image = self.inverse_transform(image)
            image = torch.clamp(image, 0, 1)
            
            # Convert to numpy
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np * 255).astype(np.uint8)
            
            # Convert to grayscale for SEM-like appearance
            image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            return image_gray
    
    def generate_from_patches(self, patch_dir, output_size=(512, 512), 
                            num_iterations=1000, save_progress=False, output_dir=None):
        """Complete pipeline: load patches -> extract features -> synthesize"""
        
        # Load patches
        patches = self.load_patches(patch_dir)
        if not patches:
            raise ValueError(f"No patches found in {patch_dir}")
        
        print(f"Loaded {len(patches)} patches from {patch_dir}")
        
        # Extract features
        print("Extracting texture features...")
        patch_features = self.extract_texture_features(patches)
        
        # Synthesize texture
        print(f"Synthesizing texture of size {output_size}...")
        generated_tensor = self.synthesize_texture(
            output_size, patch_features, num_iterations, 
            save_progress=save_progress, output_dir=output_dir
        )
        
        # Postprocess
        result_image = self.postprocess_result(generated_tensor)
        
        return result_image


if __name__ == "__main__":
    # Example usage
    synthesizer = TextureSynthesizer()
    
    # Test with patch directory
    patch_dir = "patches/sample_image"  # Adjust path as needed
    if Path(patch_dir).exists():
        result = synthesizer.generate_from_patches(
            patch_dir, 
            output_size=(512, 512),
            num_iterations=500,
            save_progress=True,
            output_dir="synthesis_progress"
        )
        
        # Save final result
        cv2.imwrite("synthesized_texture.png", result)
        print("Texture synthesis complete!") 