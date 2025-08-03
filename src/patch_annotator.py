"""
Interactive Patch Annotation Tool for SEM Images
Allows users to select and save patches from reference images
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import json
from pathlib import Path


class PatchAnnotator:
    def __init__(self, save_dir="patches"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.image = None
        self.image_path = None
        self.patches = []
        self.current_patch = None
        self.selecting = False
        self.start_x = 0
        self.start_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize the user interface"""
        self.root = tk.Tk()
        self.root.title("SEM Patch Annotator")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load image button
        load_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Patch size controls
        ttk.Label(control_frame, text="Patch Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.patch_size_var = tk.IntVar(value=64)
        patch_size_spin = ttk.Spinbox(control_frame, from_=32, to=256, width=10, 
                                     textvariable=self.patch_size_var, increment=16)
        patch_size_spin.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear patches button
        clear_btn = ttk.Button(control_frame, text="Clear Patches", command=self.clear_patches)
        clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Save patches button
        save_btn = ttk.Button(control_frame, text="Save Patches", command=self.save_patches)
        save_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Image display frame
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(display_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(fill=tk.X)
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)
        
        # Info panel
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.info_label = ttk.Label(info_frame, text="Load an image to start annotating patches")
        self.info_label.pack()
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select SEM Image",
            filetypes=[("Image files", "*.tif *.tiff *.bmp *.png *.jpg *.jpeg")]
        )
        
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if self.image is None:
                messagebox.showerror("Error", "Could not load image")
                return
                
            self.display_image()
            self.info_label.config(text=f"Loaded: {os.path.basename(file_path)} | "
                                       f"Size: {self.image.shape[1]}x{self.image.shape[0]} | "
                                       f"Patches: {len(self.patches)}")
    
    def display_image(self):
        """Display the image on canvas"""
        if self.image is None:
            return
            
        # Convert to PIL Image for display
        pil_image = Image.fromarray(self.image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Redraw existing patches
        self.draw_patches()
    
    def start_selection(self, event):
        """Start patch selection"""
        if self.image is None:
            return
            
        self.selecting = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        
        # Delete previous selection rectangle if exists
        self.canvas.delete("selection")
    
    def update_selection(self, event):
        """Update selection rectangle during dragging"""
        if not self.selecting:
            return
            
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        # Delete previous selection rectangle
        self.canvas.delete("selection")
        
        # Draw new selection rectangle
        self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline="red", width=2, tags="selection"
        )
    
    def end_selection(self, event):
        """End patch selection and save patch"""
        if not self.selecting or self.image is None:
            return
            
        self.selecting = False
        
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Calculate patch coordinates
        x1 = int(min(self.start_x, end_x))
        y1 = int(min(self.start_y, end_y))
        x2 = int(max(self.start_x, end_x))
        y2 = int(max(self.start_y, end_y))
        
        # Check if selection is large enough
        if x2 - x1 < 16 or y2 - y1 < 16:
            self.canvas.delete("selection")
            return
        
        # Extract patch from image
        patch = self.image[y1:y2, x1:x2]
        
        # Store patch info
        patch_info = {
            'coordinates': (x1, y1, x2, y2),
            'patch': patch,
            'id': len(self.patches)
        }
        
        self.patches.append(patch_info)
        
        # Update display
        self.canvas.delete("selection")
        self.draw_patch_rectangle(x1, y1, x2, y2, len(self.patches) - 1)
        
        # Update info
        self.info_label.config(text=f"Loaded: {os.path.basename(self.image_path)} | "
                                   f"Size: {self.image.shape[1]}x{self.image.shape[0]} | "
                                   f"Patches: {len(self.patches)}")
    
    def draw_patches(self):
        """Redraw all patch rectangles"""
        for i, patch_info in enumerate(self.patches):
            x1, y1, x2, y2 = patch_info['coordinates']
            self.draw_patch_rectangle(x1, y1, x2, y2, i)
    
    def draw_patch_rectangle(self, x1, y1, x2, y2, patch_id):
        """Draw a single patch rectangle with ID"""
        # Draw rectangle
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="blue", width=2, tags=f"patch_{patch_id}"
        )
        
        # Draw patch ID
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        self.canvas.create_text(
            center_x, center_y, text=str(patch_id),
            fill="yellow", font=("Arial", 12, "bold"),
            tags=f"patch_{patch_id}"
        )
    
    def clear_patches(self):
        """Clear all patches"""
        if messagebox.askyesno("Clear Patches", "Are you sure you want to clear all patches?"):
            self.patches = []
            self.canvas.delete("patch_*")  # Remove all patch rectangles
            self.info_label.config(text=f"Loaded: {os.path.basename(self.image_path) if self.image_path else 'None'} | "
                                       f"Size: {self.image.shape[1]}x{self.image.shape[0]} if self.image is not None else 'N/A' | "
                                       f"Patches: 0")
    
    def save_patches(self):
        """Save all patches to disk"""
        if not self.patches:
            messagebox.showwarning("No Patches", "No patches to save")
            return
        
        if not self.image_path:
            messagebox.showerror("Error", "No source image loaded")
            return
        
        # Create subdirectory for this image
        image_name = Path(self.image_path).stem
        patch_dir = self.save_dir / image_name
        patch_dir.mkdir(exist_ok=True)
        
        # Save patches
        patch_metadata = {
            'source_image': self.image_path,
            'image_shape': self.image.shape,
            'patches': []
        }
        
        for i, patch_info in enumerate(self.patches):
            # Save patch image
            patch_filename = f"patch_{i:03d}.png"
            patch_path = patch_dir / patch_filename
            cv2.imwrite(str(patch_path), patch_info['patch'])
            
            # Store metadata
            x1, y1, x2, y2 = patch_info['coordinates']
            patch_metadata['patches'].append({
                'id': i,
                'filename': patch_filename,
                'coordinates': [x1, y1, x2, y2],
                'size': [x2-x1, y2-y1]
            })
        
        # Save metadata
        metadata_path = patch_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(patch_metadata, f, indent=2)
        
        messagebox.showinfo("Success", f"Saved {len(self.patches)} patches to {patch_dir}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    annotator = PatchAnnotator()
    annotator.run() 