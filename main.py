#!/usr/bin/env python3
"""
SEM Synthetic Image Generator - Main Entry Point
Provides command-line interface for the complete workflow
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.patch_annotator import PatchAnnotator
from src.sem_generator import SEMGenerator


def run_annotation_tool():
    """Launch the patch annotation tool"""
    print("🖼️ Launching Patch Annotation Tool...")
    print("Instructions:")
    print("1. Click 'Load Image' to select your SEM reference image")
    print("2. Drag to select patches of interest")
    print("3. Click 'Save Patches' when done")
    print("4. Close the window to continue\n")
    
    annotator = PatchAnnotator(save_dir="patches")
    annotator.run()


def run_generation_pipeline(args):
    """Run the complete generation pipeline"""
    print("🔬 Starting SEM Generation Pipeline...")
    
    # Check if patch directories exist
    patch_dirs = []
    for i in range(1, 3):  # Support up to 2 materials for now
        patch_dir = Path(f"patches") / f"*"  # Will be more specific based on annotation
        if patch_dir.parent.exists():
            # Find subdirectories (created by annotation tool)
            subdirs = [d for d in patch_dir.parent.iterdir() if d.is_dir()]
            patch_dirs.extend([str(d) for d in subdirs[:2]])  # Take first 2
            break
    
    if not patch_dirs:
        print("❌ No patch directories found!")
        print("Please run the annotation tool first: python main.py --annotate")
        return
    
    print(f"📁 Using patch directories: {patch_dirs}")
    
    # Initialize generator
    generator = SEMGenerator(
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    # Generate dataset
    generator.generate_sem_dataset(
        material_patch_dirs=patch_dirs,
        count=args.count,
        image_size=(args.size, args.size),
        generation_method=args.method,
        mask_types=args.mask_types,
        add_effects=not args.no_effects
    )
    
    print(f"\n✅ Generation complete! Check {args.output_dir}/dataset/")


def run_quick_demo():
    """Run a quick demo with existing sample images"""
    print("🚀 Running Quick Demo...")
    
    # Check if sample images exist
    sample_files = ['1_030.tif', '1-50-1.bmp', '3-4-02.tif']
    available_samples = [f for f in sample_files if Path(f).exists()]
    
    if not available_samples:
        print("❌ No sample images found in current directory!")
        print("Please place some SEM images in the current directory or run annotation tool.")
        return
    
    print(f"📁 Found sample images: {available_samples}")
    print("This demo will create patches automatically from the first image...")
    
    # Create demo patches directory
    demo_dir = Path("demo_patches")
    demo_dir.mkdir(exist_ok=True)
    
    # Simple patch extraction from first image
    import cv2
    import numpy as np
    
    img = cv2.imread(available_samples[0], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        h, w = img.shape
        patch_size = 64
        
        # Extract regular grid of patches
        patch_id = 0
        for y in range(0, h - patch_size, patch_size * 2):
            for x in range(0, w - patch_size, patch_size * 2):
                patch = img[y:y+patch_size, x:x+patch_size]
                patch_path = demo_dir / f"patch_{patch_id:03d}.png"
                cv2.imwrite(str(patch_path), patch)
                patch_id += 1
                if patch_id >= 20:  # Limit number of patches
                    break
            if patch_id >= 20:
                break
        
        print(f"📦 Created {patch_id} demo patches in {demo_dir}")
        
        # Run generation with demo patches
        generator = SEMGenerator(output_dir="demo_output", seed=42)
        
        generator.generate_sem_dataset(
            material_patch_dirs=[str(demo_dir)],  # Use same patches for both materials
            count=5,
            image_size=(256, 256),  # Smaller for demo
            generation_method='quilting',  # Faster method
            mask_types=['perlin', 'voronoi'],
            add_effects=True
        )
        
        print("✅ Demo complete! Check demo_output/dataset/")
    else:
        print("❌ Could not read sample image")


def main():
    parser = argparse.ArgumentParser(
        description="SEM Synthetic Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch annotation tool to create patches
  python main.py --annotate
  
  # Generate 20 synthetic images using mixed method
  python main.py --generate --count 20 --method mixed
  
  # Quick demo with automatic patch extraction
  python main.py --demo
  
  # Generate with specific parameters
  python main.py --generate --count 50 --size 1024 --method neural --device cuda
        """
    )
    
    # Main actions
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--annotate', action='store_true',
                            help='Launch patch annotation tool')
    action_group.add_argument('--generate', action='store_true',
                            help='Generate synthetic SEM images')
    action_group.add_argument('--demo', action='store_true',
                            help='Run quick demo with automatic patches')
    
    # Generation parameters
    parser.add_argument('--count', type=int, default=10,
                       help='Number of images to generate (default: 10)')
    parser.add_argument('--size', type=int, default=512,
                       help='Output image size (default: 512)')
    parser.add_argument('--method', choices=['neural', 'quilting', 'mixed'], 
                       default='mixed',
                       help='Generation method (default: mixed)')
    parser.add_argument('--output-dir', default='generated_sem',
                       help='Output directory (default: generated_sem)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device for neural synthesis (auto-detect if not specified)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--mask-types', nargs='+', 
                       choices=['perlin', 'voronoi', 'cellular', 'fractal', 'layered', 'blob', 'network'],
                       default=['perlin', 'voronoi', 'cellular'],
                       help='Types of masks to generate')
    parser.add_argument('--no-effects', action='store_true',
                       help='Disable SEM-specific effects')
    
    args = parser.parse_args()
    
    print("🔬 SEM Synthetic Image Generator")
    print("=" * 50)
    
    if args.annotate:
        run_annotation_tool()
    elif args.generate:
        run_generation_pipeline(args)
    elif args.demo:
        run_quick_demo()


if __name__ == "__main__":
    main() 