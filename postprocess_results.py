"""
Post-Processing Module
Mask refinement, polygonization, and visualization
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import label, center_of_mass
import json
from typing import List, Tuple


# ============================================================================
# MASK REFINEMENT
# ============================================================================

class MaskRefiner:
    """Refine predicted masks through morphological operations."""
    
    @staticmethod
    def remove_small_objects(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """
        Remove small isolated detections.
        
        Args:
            mask: Binary mask (H, W)
            min_size: Minimum object size in pixels
        
        Returns:
            Refined mask
        """
        labeled, num_features = label(mask)
        
        refined = np.zeros_like(mask)
        for feature_id in range(1, num_features + 1):
            feature_mask = (labeled == feature_id)
            if feature_mask.sum() >= min_size:
                refined[feature_mask] = 1
        
        return refined
    
    @staticmethod
    def morphological_closing(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply morphological closing to fill small holes.
        
        Args:
            mask: Binary mask (H, W)
            kernel_size: Kernel size for morphological operation
        
        Returns:
            Closed mask
        """
        from scipy.ndimage import binary_closing
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        return binary_closing(mask, structure=kernel).astype(np.uint8)
    
    @staticmethod
    def morphological_opening(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply morphological opening to remove noise.
        
        Args:
            mask: Binary mask (H, W)
            kernel_size: Kernel size for morphological operation
        
        Returns:
            Opened mask
        """
        from scipy.ndimage import binary_opening
        kernel = np.ones((kernel_size, kernel_size), dtype=bool)
        return binary_opening(mask, structure=kernel).astype(np.uint8)
    
    @staticmethod
    def refine_mask(mask: np.ndarray, min_size: int = 50, 
                   close_kernel: int = 5, open_kernel: int = 3) -> np.ndarray:
        """
        Apply full refinement pipeline.
        
        Args:
            mask: Binary mask
            min_size: Minimum object size
            close_kernel: Kernel size for closing
            open_kernel: Kernel size for opening
        
        Returns:
            Refined mask
        """
        # Opening (remove noise)
        mask = MaskRefiner.morphological_opening(mask, open_kernel)
        
        # Closing (fill holes)
        mask = MaskRefiner.morphological_closing(mask, close_kernel)
        
        # Remove small objects
        mask = MaskRefiner.remove_small_objects(mask, min_size)
        
        return mask


# ============================================================================
# POLYGONIZATION & GeoJSON
# ============================================================================

class Polygonizer:
    """Convert masks to polygons and GeoJSON."""
    
    @staticmethod
    def get_contours(mask: np.ndarray) -> List[np.ndarray]:
        """
        Extract contours from binary mask.
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            List of contour arrays
        """
        from scipy import ndimage
        
        labeled, num_features = label(mask)
        contours = []
        
        for feature_id in range(1, num_features + 1):
            feature_mask = (labeled == feature_id).astype(np.uint8)
            # Simple edge extraction
            edges = np.gradient(feature_mask.astype(float))
            contour = np.where(edges[0]**2 + edges[1]**2 > 0)
            if len(contour[0]) > 0:
                contours.append(np.column_stack(contour))
        
        return contours
    
    @staticmethod
    def get_bounding_boxes(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes for detected objects.
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            List of (x_min, y_min, x_max, y_max) tuples
        """
        labeled, num_features = label(mask)
        bboxes = []
        
        for feature_id in range(1, num_features + 1):
            feature_mask = (labeled == feature_id)
            coords = np.where(feature_mask)
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            bboxes.append((x_min, y_min, x_max, y_max))
        
        return bboxes
    
    @staticmethod
    def get_centroids(mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        Get centroid coordinates of detected objects.
        
        Args:
            mask: Binary mask (H, W)
        
        Returns:
            List of (y, x) centroid coordinates
        """
        labeled, num_features = label(mask)
        centroids = []
        
        for feature_id in range(1, num_features + 1):
            feature_mask = (labeled == feature_id)
            coords = np.where(feature_mask)
            
            y_centroid = coords[0].mean()
            x_centroid = coords[1].mean()
            
            centroids.append((y_centroid, x_centroid))
        
        return centroids
    
    @staticmethod
    def mask_to_geojson(mask: np.ndarray, pixel_to_latlon=None, 
                       properties: dict = None) -> dict:
        """
        Convert mask to GeoJSON format.
        
        Args:
            mask: Binary mask (H, W)
            pixel_to_latlon: Function to convert (y, x) pixel to (lat, lon)
            properties: Additional properties for features
        
        Returns:
            GeoJSON FeatureCollection
        """
        labeled, num_features = label(mask)
        features = []
        
        for feature_id in range(1, num_features + 1):
            feature_mask = (labeled == feature_id)
            coords = np.where(feature_mask)
            
            # Get bounding box
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # Get centroid
            y_center = coords[0].mean()
            x_center = coords[1].mean()
            
            # Convert to lat/lon if function provided
            if pixel_to_latlon:
                lat_center, lon_center = pixel_to_latlon(y_center, x_center)
                lat_min, lon_min = pixel_to_latlon(y_max, x_min)
                lat_max, lon_max = pixel_to_latlon(y_min, x_max)
            else:
                lat_center, lon_center = y_center, x_center
                lat_min, lon_min = y_max, x_min
                lat_max, lon_max = y_min, x_max
            
            feature = {
                'type': 'Feature',
                'properties': {
                    'id': feature_id,
                    'area_pixels': feature_mask.sum(),
                    'centroid_lat': lat_center,
                    'centroid_lon': lon_center,
                    **(properties or {})
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon_center, lat_center]
                }
            }
            features.append(feature)
        
        return {
            'type': 'FeatureCollection',
            'features': features
        }


# ============================================================================
# OVERLAY & VISUALIZATION
# ============================================================================

class VisualizationHelper:
    """Helper functions for visualization and overlay."""
    
    @staticmethod
    def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, 
                             alpha: float = 0.5, color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Overlay segmentation mask on image.
        
        Args:
            image: RGB image (H, W, 3), values in [0, 255]
            mask: Binary mask (H, W)
            alpha: Transparency of mask overlay
            color: Color of mask overlay (R, G, B)
        
        Returns:
            Overlay image
        """
        overlay = image.copy().astype(float)
        
        mask_indices = mask > 0
        for c in range(3):
            overlay[mask_indices, c] = (
                (1 - alpha) * overlay[mask_indices, c] + 
                alpha * color[c]
            )
        
        return np.clip(overlay, 0, 255).astype(np.uint8)
    
    @staticmethod
    def create_rgb_from_bands(band_r: np.ndarray, band_g: np.ndarray, 
                             band_b: np.ndarray) -> np.ndarray:
        """
        Create RGB composite from individual bands.
        
        Args:
            band_r: Red channel (H, W), values in [0, 1]
            band_g: Green channel (H, W), values in [0, 1]
            band_b: Blue channel (H, W), values in [0, 1]
        
        Returns:
            RGB image (H, W, 3), values in [0, 255]
        """
        rgb = np.stack([band_r, band_g, band_b], axis=2)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return rgb
    
    @staticmethod
    def create_heatmap_from_predictions(predictions: np.ndarray) -> np.ndarray:
        """
        Create heatmap visualization from model predictions.
        
        Args:
            predictions: Prediction probabilities (H, W), values in [0, 1]
        
        Returns:
            Heatmap RGB image (H, W, 3)
        """
        import matplotlib.cm as cm
        
        # Normalize to [0, 1]
        normalized = np.clip(predictions, 0, 1)
        
        # Apply colormap
        colormap = cm.get_cmap('hot')
        heatmap = colormap(normalized)
        
        # Convert to 8-bit
        heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
        
        return heatmap_rgb


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_results(predictions: np.ndarray, image: np.ndarray, 
                  output_dir: str = './results'):
    """
    Export segmentation results in multiple formats.
    
    Args:
        predictions: Model predictions (H, W), values in [0, 1]
        image: Original image (H, W, 3) or (6, H, W)
        output_dir: Directory to save results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert predictions to binary mask
    mask = (predictions > 0.5).astype(np.uint8)
    
    # Refine mask
    refined_mask = MaskRefiner.refine_mask(mask)
    
    # Save as numpy
    np.save(os.path.join(output_dir, 'prediction_mask.npy'), refined_mask)
    
    # Save as GeoJSON
    geojson = Polygonizer.mask_to_geojson(refined_mask)
    with open(os.path.join(output_dir, 'detections.geojson'), 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    
    # Create visualization
    if image.shape[0] == 3 or image.shape[2] == 3:
        # Convert to uint8 if needed
        if image.max() <= 1:
            image_display = (image * 255).astype(np.uint8)
        else:
            image_display = image.astype(np.uint8)
        
        if image_display.shape[0] == 3:
            image_display = np.transpose(image_display, (1, 2, 0))
        
        # Create overlay
        overlay = VisualizationHelper.overlay_mask_on_image(
            image_display, refined_mask, alpha=0.4, color=(255, 0, 0)
        )
        
        # Save overlay (requires PIL)
        try:
            from PIL import Image as PILImage
            PILImage.fromarray(overlay).save(os.path.join(output_dir, 'overlay.png'))
        except ImportError:
            pass
    
    print(f"✓ Results exported to {output_dir}")
