import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, label
from skimage.measure import regionprops
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.ndimage import (label, distance_transform_edt)
from skimage import filters, morphology, measure
import numpy as np
from skimage.draw import polygon

class AxialAnchor:
    def remove_noise(self, img_slice):
        thresh = filters.threshold_otsu(img_slice)
        rough_mask = img_slice > thresh

        cleaned = morphology.binary_closing(rough_mask, morphology.disk(3))
        labels = measure.label(cleaned)
        if labels.max() > 0:
            brain_blob = (labels == (np.argmax(np.bincount(labels.flat)[1:]) + 1))
        else:
            brain_blob = cleaned

        final_brain_mask = ndimage.binary_fill_holes(brain_blob)
        final_brain_mask = morphology.binary_closing(final_brain_mask, morphology.disk(8))
        
        return np.where(img_slice, final_brain_mask, 0)

    def is_brain_like(self, contour, min_area=1000):
        y, x = contour[:, 0], contour[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        if area < min_area:
            return False

        try:
            hull = ConvexHull(contour)
            solidity = area / hull.volume
        except:
            return "Invalid"

        if solidity > 0.98:
            return True
        else:
            return False

    def get_brain_contour(self, img_slice):
        binary_mask = self.remove_noise(img_slice)
        
        if binary_mask.sum() < 100:
            return None

        contours = measure.find_contours(binary_mask.astype(float), 0.5)
        
        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=len)
    
        brain_mask = np.zeros_like(img_slice, dtype=bool)
        rr, cc = polygon(largest_contour[:, 0], largest_contour[:, 1], img_slice.shape)
        brain_mask[rr, cc] = True
        
        return brain_mask, largest_contour

    def estimate_artifact_region(self, img_slice, brain_mask):
        # 1. Get the "Dents" (Air spaces inside the Hull but outside the Brain)
        hull = morphology.convex_hull_image(brain_mask)
        dents_mask = hull & ~brain_mask
        
        # 2. Cleanup: Remove tiny noise/pixels
        dents_mask = morphology.remove_small_objects(dents_mask, min_size=20)
        
        # 3. Label each space and check criteria
        labeled_dents = measure.label(dents_mask)
        regions = measure.regionprops(labeled_dents)
        
        best_mask = np.zeros_like(img_slice, dtype=bool)
        max_score = 0
        
        # Get the Hull boundary (perimeter) to check for overlap
        # Erosion trick: pixels on the edge of the hull
        hull_boundary = hull ^ morphology.binary_erosion(hull, morphology.disk(1))

        for reg in regions:
            # A. Check for Hull Boundary Overlap
            # If the dent touches the outer hull, it's a "boundary artifact"
            touches_boundary = np.logical_and(reg.image, hull_boundary[reg.slice]).any()
            
            if touches_boundary:
                # B. Check for Compactness (Roundness)
                # circularity = 4pi * Area / Perimeter^2
                perimeter = reg.perimeter
                area = reg.area
                circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
                
                # C. Score by Size + Circularity
                # We want compact (high circularity) and medium sized
                score = area * circularity
                
                if score > max_score:
                    max_score = score
                    best_mask = (labeled_dents == reg.label)

        return max_score, best_mask

    def find_anchor(self, img_data, target_slices, axis=2, verbose=False):
        candidate_slices = []
        
        for z in target_slices:
            img_slice = np.take(img_data, z, axis=axis)
            
            if img_slice.size == 0 or img_slice.max() == img_slice.min():
                continue
            
            result = self.get_brain_contour(img_slice)
            if result is None:
                continue

            brain_mask, contour = result
            
            if not self.is_brain_like(contour):
                continue
            
            # Detect artifacts (simple - no predefined search zone)
            artifact_score, artifact_mask = self.estimate_artifact_region(
                img_slice, 
                brain_mask
            )
            
            if artifact_mask.sum() > 0:
                n_clusters = label(artifact_mask)[1]
            else:
                n_clusters = 0
            
            # Store candidate
            if artifact_score > 0:
                candidate_slices.append({
                    'slice_index': z,
                    'artifact_score': artifact_score,
                    'artifact_mask': artifact_mask,
                    'brain_mask': brain_mask,
                    'contour': contour,
                    'n_clusters': n_clusters
                })
                
                if verbose:
                    print(f"{z:<8} {n_clusters:<10} {'Yes':<15} {artifact_score:<15.3f}")
            else:
                if verbose and z % 20 == 0:
                    print(f"{z:<8} {0:<10} {'No':<15} {0.0:<15.3f}")
        
        if len(candidate_slices) == 0:
            raise ValueError("No suitable anchor slice found!")
        
        # Find slice with strongest artifact
        best_candidate = max(candidate_slices, key=lambda x: x['artifact_score'])
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"✓ ANCHOR SLICE: {best_candidate['slice_index']}")
            print("=" * 60)
            print(f"Artifact score:     {best_candidate['artifact_score']:.3f}")
            print(f"Number of clusters: {best_candidate['n_clusters']}")
            print(f"Total candidates:   {len(candidate_slices)}")
        
        return best_candidate['slice_index'], best_candidate

    def refine_anchor_mask(self, mask, sigma=1.0):
        """
        Refines and smoothens the binary anchor mask.
        """
        if mask is None or mask.sum() == 0:
            return mask

        # 1. Fill any tiny internal pinholes
        refined = ndimage.binary_fill_holes(mask)

        # 2. Morphological Opening
        # This removes tiny 'hairs' or thin protrusions (spikes) 
        # without changing the overall size much.
        refined = morphology.binary_opening(refined, morphology.disk(2))

        # 3. Morphological Closing
        # This smoothens the outer boundary and bridges small gaps.
        refined = morphology.binary_closing(refined, morphology.disk(2))

        # 4. Gaussian Smoothing (Anti-aliasing)
        # Turning the mask into a float, blurring, and re-thresholding 
        # creates much smoother 'sub-pixel' looking curves.
        smoothed_float = filters.gaussian(refined.astype(float), sigma=sigma)
        refined = smoothed_float > 0.5  # Re-threshold at 50%

        return refined

    def visualize_anchor_detection(self, img_slice, brain_mask, artifact_mask, 
                                     slice_index, artifact_score):

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1
        axes[0, 0].imshow(img_slice, cmap='gray')
        axes[0, 0].set_title(f'Slice {slice_index}\nOriginal')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_slice, cmap='gray')
        axes[0, 1].contour(brain_mask, colors='yellow', linewidths=2)
        axes[0, 1].set_title('Brain Boundary')
        axes[0, 1].axis('off')
        
        # Show distance from boundary
        distance_map = distance_transform_edt(brain_mask)
        axes[0, 2].imshow(distance_map, cmap='viridis')
        axes[0, 2].contour(brain_mask, colors='white', linewidths=1, alpha=0.5)
        axes[0, 2].set_title('Distance from Boundary\n(brighter = deeper)')
        axes[0, 2].axis('off')
        
        # Row 2
        # Show zero/near-zero regions
        max_int = img_slice.max()
        threshold = max_int * 0.02
        near_zero = img_slice < threshold
        
        axes[1, 0].imshow(img_slice, cmap='gray')
        axes[1, 0].imshow(near_zero, cmap='Reds', alpha=0.5)
        axes[1, 0].set_title(f'Near-Zero Regions\n(< {threshold:.1f})')
        axes[1, 0].axis('off')
        
        # Show detected artifact
        axes[1, 1].imshow(img_slice, cmap='gray')
        axes[1, 1].imshow(artifact_mask, cmap='hot', alpha=0.7)
        axes[1, 1].contour(artifact_mask, colors='red', linewidths=2)
        axes[1, 1].set_title(f'Detected Artifact\nScore: {artifact_score:.3f}')
        axes[1, 1].axis('off')
        
        # Properties text
        if artifact_mask.sum() > 0:
            labeled_artifacts = label(artifact_mask)[0]
            props = regionprops(labeled_artifacts)
            
            properties_text = "Detected Artifacts:\n" + "─" * 25 + "\n"
            
            for i, region in enumerate(props, 1):
                area = region.area
                perimeter = region.perimeter
                circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Distance from boundary
                dist = distance_map[region.coords[:, 0], region.coords[:, 1]].mean()
                
                properties_text += (
                    f"\nCluster {i}:\n"
                    f"  Size: {area} pixels\n"
                    f"  Circularity: {circ:.3f}\n"
                    f"  Distance from edge: {dist:.1f}px\n"
                    f"  Mean intensity: {img_slice[artifact_mask].mean():.1f}\n"
                )
        else:
            properties_text = "No artifacts detected"
        
        axes[1, 2].text(0.1, 0.5, properties_text, 
                    fontsize=9, family='monospace',
                    verticalalignment='center')
        axes[1, 2].axis('off')
        plt.tight_layout()