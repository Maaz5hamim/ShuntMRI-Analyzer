
import numpy as np
from skimage import measure, filters, morphology
from collections import deque
from scipy import ndimage
from scipy.ndimage import label, center_of_mass, binary_dilation, binary_fill_holes, generate_binary_structure, binary_closing, binary_opening
from skimage.measure import regionprops

class Tracker():
    def __init__(self, search_radius):
        self.search_radius = search_radius

    def _get_brain_mask(self, img_data, brain_mask_volume, z, axis):
        if brain_mask_volume is not None:
            return np.take(brain_mask_volume, z, axis=axis)
        img_slice = np.take(img_data, z, axis=axis)
        result = self._get_brain_contour(img_slice)
        return result[0] if result is not None else None

    def _get_brain_contour(self, img_slice):
        binary_mask = self.remove_noise(img_slice)
        
        if binary_mask.sum() < 100:
            return None

        contours = measure.find_contours(binary_mask.astype(float), 0.5)
        
        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=len)
    
        from skimage.draw import polygon
        brain_mask = np.zeros_like(img_slice, dtype=bool)
        rr, cc = polygon(largest_contour[:, 0], largest_contour[:, 1], img_slice.shape)
        brain_mask[rr, cc] = True
        
        return brain_mask, largest_contour

    def _set_slice(self, volume, z, axis, mask):
        if axis == 2:
            volume[:, :, z] = mask
        elif axis == 1:
            volume[:, z, :] = mask
        elif axis == 0:
            volume[z, :, :] = mask

    def track_artifact_3d(self, img_data, anchor_slice, anchor_mask, brain_mask_volume=None, axis=2):
        artifact_3d = np.zeros_like(img_data, dtype=bool)
        self._set_slice(artifact_3d, anchor_slice, axis, anchor_mask)

        backward_end = self.track_backward(img_data, anchor_slice, anchor_mask, artifact_3d, brain_mask_volume, axis)
        forward_end  = self.track_forward( img_data, anchor_slice, anchor_mask, artifact_3d, brain_mask_volume, axis)

        # Bidirectional refinement: re-check every tracked slice using both neighbours
        artifact_3d = self.refine_bidirectional(
            img_data, artifact_3d, backward_end, forward_end,
            anchor_slice, brain_mask_volume, axis)

        # Drop isolated spiky voxels — keep only the single largest 3-D connected blob
        artifact_3d = self.keep_largest_component(artifact_3d)

        tracking_info = {
            'anchor_slice':    anchor_slice,
            'start_slice':     backward_end,
            'end_slice':       forward_end,
            'total_slices':    forward_end - backward_end + 1,
            'artifact_volume': artifact_3d.sum()
        }
        return artifact_3d, tracking_info

    def refine_bidirectional(self, img_data, artifact_3d, start_slice, end_slice,
        anchor_slice, brain_mask_volume, axis):
        refined = artifact_3d.copy()

        for z in range(start_slice, end_slice + 1):
            if z == anchor_slice:
                continue  # keep anchor fixed

            img_slice  = np.take(img_data, z, axis=axis)
            brain_mask = self._get_brain_mask(img_data, brain_mask_volume, z, axis)
            if brain_mask is None:
                continue

            prev_mask = np.take(refined, z - 1, axis=axis) if z > start_slice else None
            next_mask = np.take(refined, z + 1, axis=axis) if z < end_slice   else None

            neighbors = [m for m in (prev_mask, next_mask) if m is not None and m.any()]
            if not neighbors:
                continue
            combined_ref = np.logical_or.reduce(neighbors) if len(neighbors) > 1 else neighbors[0]
            expected_centroid = center_of_mass(combined_ref)
            if np.any(np.isnan(expected_centroid)):
                continue

            new_mask, confidence = self.find_artifact_in_slice(
                img_slice, brain_mask, prev_mask, expected_centroid,
                search_radius=self.search_radius, next_mask=next_mask)

            if new_mask is not None and new_mask.sum() >= 5:
                self._set_slice(refined, z, axis, new_mask)

        return refined

    def track_backward(self, img_data, anchor_slice, anchor_mask, artifact_3d, brain_mask_volume, axis=2):
        # Rolling window of the last 3 masks: their union is the search reference.
        # A single drifted detection no longer corrupts the whole subsequent chain.
        mask_buffer   = deque([anchor_mask.copy()], maxlen=3)
        current_slice = anchor_slice

        for z in range(anchor_slice - 1, -1, -1):
            img_slice  = np.take(img_data, z, axis=axis)
            brain_mask = self._get_brain_mask(img_data, brain_mask_volume, z, axis)
            if brain_mask is None:
                break

            smooth_ref      = np.logical_or.reduce(list(mask_buffer))
            smooth_centroid = center_of_mass(smooth_ref)

            artifact_mask, confidence = self.find_artifact_in_slice(
                img_slice, brain_mask, smooth_ref, smooth_centroid, search_radius=self.search_radius)

            if artifact_mask is None or artifact_mask.sum() < 5:
                break
            # Validate drift/size against the single most-recent mask, not the smoothed union
            if not self.validate_tracked_artifact(artifact_mask, mask_buffer[-1], img_slice, brain_mask):
                break

            self._set_slice(artifact_3d, z, axis, artifact_mask)
            mask_buffer.append(artifact_mask)
            current_slice = z

        return current_slice

    def track_forward(self, img_data, anchor_slice, anchor_mask, artifact_3d, brain_mask_volume, axis=2):
        n_slices      = img_data.shape[axis]
        mask_buffer   = deque([anchor_mask.copy()], maxlen=3)
        current_slice = anchor_slice

        for z in range(anchor_slice + 1, n_slices):
            img_slice  = np.take(img_data, z, axis=axis)
            brain_mask = self._get_brain_mask(img_data, brain_mask_volume, z, axis)
            if brain_mask is None:
                break

            smooth_ref      = np.logical_or.reduce(list(mask_buffer))
            smooth_centroid = center_of_mass(smooth_ref)

            artifact_mask, confidence = self.find_artifact_in_slice(
                img_slice, brain_mask, smooth_ref, smooth_centroid, search_radius=self.search_radius)

            if artifact_mask is None or artifact_mask.sum() < 5:
                break
            if not self.validate_tracked_artifact(artifact_mask, mask_buffer[-1], img_slice, brain_mask):
                break

            self._set_slice(artifact_3d, z, axis, artifact_mask)
            mask_buffer.append(artifact_mask)
            current_slice = z

        return current_slice

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
    
    def keep_largest_component(self, artifact_3d):
        """Discard everything except the largest 3-D connected component."""
        from scipy.ndimage import label as nd_label
        labeled, n = nd_label(artifact_3d)
        if n == 0:
            return artifact_3d
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background label
        return labeled == sizes.argmax()



        if brain_mask.sum() == 0:
            return None, 0.0

        hull_mask  = morphology.convex_hull_image(brain_mask)
        dents_mask = hull_mask & ~brain_mask
        dents_mask = morphology.remove_small_objects(dents_mask, min_size=20)

        if dents_mask.sum() == 0:
            return None, 0.0

        spatial_mask = self.create_search_region(brain_mask.shape, expected_centroid, search_radius)

        neighbor_masks = [m for m in (prev_mask, next_mask) if m is not None and m.sum() > 0]
        if neighbor_masks:
            predicted_zone = np.zeros_like(brain_mask, dtype=bool)
            for nm in neighbor_masks:
                predicted_zone |= binary_dilation(nm, iterations=4)
            candidate_zones = dents_mask & spatial_mask #& predicted_zone
            if candidate_zones.sum() == 0:
                candidate_zones = dents_mask & spatial_mask
        else:
            candidate_zones = dents_mask & spatial_mask

        if candidate_zones.sum() == 0:
            return None, 0.0

        labeled, n_components = label(candidate_zones)
        brain_mean = img_slice[brain_mask].mean() if brain_mask.any() else 1.0
        best_cluster, best_score = None, -1.0

        for comp_id in range(1, n_components + 1):
            component = (labeled == comp_id)
            size = component.sum()
            if size < 10 or size > 2000:
                continue

            mean_intensity = img_slice[component].mean()

            # Global gate: at least 20% darker than brain mean
            if mean_intensity > brain_mean * 0.80:
                continue

            # Local contrast gate: darker than immediately adjacent brain tissue
            adjacent_brain = binary_dilation(component, iterations=3) & brain_mask
            if adjacent_brain.any():
                if mean_intensity > img_slice[adjacent_brain].mean() * 0.85:
                    continue

            try:
                props = regionprops(component.astype(int))[0]
            except IndexError:
                continue

            # IoU averaged across all available neighbours
            if neighbor_masks:
                iou_vals = []
                for nm in neighbor_masks:
                    inter = np.logical_and(component, nm).sum()
                    union = np.logical_or(component, nm).sum()
                    iou_vals.append(inter / (union + 1e-6))
                iou_score = float(np.mean(iou_vals))
            else:
                iou_score = 1.0

            dist          = np.linalg.norm(np.array(props.centroid) - np.array(expected_centroid))
            dist_score    = max(0.0, 1.0 - dist / search_radius)
            darkness_score = 1.0 - np.clip(mean_intensity / (brain_mean + 1e-6), 0, 1)

            confidence = (
                0.50 * iou_score                       +
                0.20 * dist_score                      +
                0.15 * min(props.solidity, 0.95)       +
                0.15 * darkness_score
            )

            if confidence > best_score and confidence > 0.25:
                best_score   = confidence
                best_cluster = component

        return best_cluster, best_score

    def find_artifact_in_slice(self, img_slice, brain_mask, prev_mask, expected_centroid,
        search_radius, next_mask=None):

        if brain_mask.sum() == 0:
            return None, 0.0

        hull_mask  = morphology.convex_hull_image(brain_mask)
        dents_mask = hull_mask & ~brain_mask

        spatial_mask = self.create_search_region(img_slice.shape, expected_centroid, search_radius)
        target_zone = dents_mask & spatial_mask
        
        if target_zone.sum() == 0:
            return None, 0.0
        
        zone_values = img_slice[target_zone]
        brain_mean = img_slice[brain_mask].mean()
        
        threshold = filters.threshold_otsu(zone_values)
        artifact_pixels_mask = (img_slice <= threshold) & target_zone

        if artifact_pixels_mask.sum() < 3: # Too few pixels to be an artifact
            return None, 0.0
        
        refined_artifact = morphology.binary_opening(artifact_pixels_mask, morphology.disk(1))

        from scipy.ndimage import label
        labeled_regions, num_features = label(refined_artifact)
        
        if num_features == 0:
            return None, 0.0
        elif num_features > 1:
            # If there are multiple dark spots, pick the one closest to the anchor centroid
            best_comp = None
            min_dist = float('inf')
            
            for i in range(1, num_features + 1):
                region_mask = (labeled_regions == i)
                region_com = center_of_mass(region_mask)
                d = np.linalg.norm(np.array(region_com) - np.array(expected_centroid))
                
                # Tie-break: You could also factor in darkness here
                if d < min_dist:
                    min_dist = d
                    best_comp = region_mask
            
            refined_artifact = best_comp
        
        # 5. Score based on Darkest Average
        # A real artifact will have a very low mean intensity
        actual_mean = img_slice[refined_artifact].mean()
        darkness_score = 1.0 - (actual_mean / brain_mean)
        
        # Distance penalty: how far is this new dark cluster from our anchor?
        current_centroid = center_of_mass(refined_artifact)
        dist = np.linalg.norm(np.array(current_centroid) - np.array(expected_centroid))
        dist_score = max(0.0, 1.0 - dist / search_radius)

        # Final confidence: mostly based on how dark it is and how close it is
        confidence = (0.6 * darkness_score) + (0.4 * dist_score)

        return refined_artifact, confidence

    def validate_tracked_artifact(self, current_mask, previous_mask, img_slice, brain_mask, verbose=False):
        if previous_mask is None:
            return True

        drift = np.linalg.norm(
            np.array(center_of_mass(current_mask)) - np.array(center_of_mass(previous_mask)))
        if drift > 5:
            if verbose: print(f"  ✗ centroid drift {drift:.1f}px")
            return False

        size_ratio = current_mask.sum() / (previous_mask.sum() + 1e-6)
        if not (0.3 <= size_ratio <= 2.0):
            if verbose: print(f"  ✗ size ratio {size_ratio:.2f}")
            return False

        return True

    def create_search_region(self, shape, center, radius):
        Y, X = np.ogrid[:shape[0], :shape[1]]
        cy, cx = center
        return (np.sqrt((Y - cy)**2 + (X - cx)**2) <= radius)

    def combine_planes(self, *artifact_volumes):
        # Union of all planes
        combined = np.zeros_like(artifact_volumes[0], dtype=bool)
        for artifact in artifact_volumes:
            combined |= artifact
        
        # Keep only largest component
        return self.keep_largest_component(combined)

    def track_artifact(self, img_data, axial_anchor_idx, axial_anchor_mask, verbose=False):
        if verbose == True:
            print("=" * 80)
            print("MULTI-PLANAR CATHETER ARTIFACT TRACKING")
            print("=" * 80)
        
            # STEP 1: Axial tracking (primary)
            print(f"\n{'='*80}")
            print(f"STEP 1: AXIAL TRACKING (Axis=2, Z slices)")
            print(f"{'='*80}")
            print(f"Axial anchor: Z slice = {axial_anchor_idx}")
            print(f"Seed pixels: {axial_anchor_mask.sum()}")

        artifact_3d_axial, info_axial = self.track_artifact_3d(
            img_data,
            anchor_slice=axial_anchor_idx,
            anchor_mask=axial_anchor_mask,
            brain_mask_volume=None,
            axis=2
        )

        info_axial['plane'] = 'axial'

        if verbose == True:
            print(f"Axial slices tracked: {info_axial['start_slice']} → {info_axial['end_slice']}")
            print(f"Axial volume: {info_axial['artifact_volume']} voxels")

        # Find coronal anchor from axial result
        coronal_counts = artifact_3d_axial.sum(axis=(0, 2))
        coronal_idx = int(coronal_counts.argmax())
        coronal_mask = artifact_3d_axial[:, coronal_idx, :].astype(bool)

        if verbose == True:
            print(f"\n{'='*80}")
            print(f"CORONAL TRACKING (Axis=1, Y slices)")
            print(f"{'='*80}")
            print(f"Coronal anchor: Y slice = {coronal_idx}")
            print(f"Seed pixels: {coronal_mask.sum()}")

        artifact_3d_coronal, info_coronal = self.track_artifact_3d(
            img_data,
            anchor_slice=coronal_idx,
            anchor_mask=coronal_mask,
            brain_mask_volume=None,
            axis=1
        )

        info_coronal['plane'] = 'coronal'

        if verbose == True:
            print(f"Coronal slices tracked: {info_coronal['start_slice']} → {info_coronal['end_slice']}")
            print(f"Coronal volume: {info_coronal['artifact_volume']} voxels")

        artifact_axial_coronal = self.combine_planes(artifact_3d_axial, artifact_3d_coronal)

        if verbose == True:
            print(f"\nAxial + Coronal combined: {artifact_axial_coronal.sum()} voxels")

        sagittal_counts = artifact_axial_coronal.sum(axis=(1, 2))   # shape: (X,)
        sagittal_idx  = int(sagittal_counts.argmax())
        sagittal_mask = artifact_axial_coronal[sagittal_idx, :, :].astype(bool)   # shape: (Y, Z)

        if verbose == True:
            print(f"\n{'='*80}")
            print(f"SAGITTAL TRACKING (Axis=0, X slices)")
            print(f"{'='*80}")
            print(f"Sagittal anchor: X slice = {sagittal_idx}")
            print(f"Seed pixels: {sagittal_mask.sum()}")

        artifact_3d_sagittal, info_sagittal = self.track_artifact_3d(
            img_data,
            anchor_slice=sagittal_idx,
            anchor_mask=sagittal_mask,
            brain_mask_volume=None,
            axis=0
        )
        
        info_sagittal['plane'] = 'sagittal'

        if verbose == True:
            print(f"\n{'='*80}")
            print("STEP 4: FINAL COMBINATION")
            print(f"{'='*80}")
        
        artifact_combined = self.combine_planes(
            artifact_3d_axial, artifact_3d_coronal, artifact_3d_sagittal
        )

        print(f"Final combined volume: {artifact_combined.sum()} voxels")
        # print(f"Improvement over axial-only: {artifact_combined.sum() / info_axial['artifact_volume']:.2f}x")

        tracking_results = {
            'axial': (artifact_3d_axial, info_axial),
            'coronal': (artifact_3d_coronal, info_coronal),
            'sagittal': (artifact_3d_sagittal, info_sagittal),
            'combined': artifact_combined
        }

        return artifact_combined, tracking_results
    
    def compensate_volume_loss(self, original, smoothed, max_iterations=3):
        original_vol = original.sum()
        current_vol = smoothed.sum()
        
        if current_vol >= original_vol * 0.95:  # Within 5% is good enough
            return smoothed
            
        compensated = smoothed.copy()
        struct = generate_binary_structure(3, 1)
        
        for i in range(max_iterations):
            # Try dilating
            test = binary_dilation(compensated, structure=struct, iterations=1)
            test_vol = test.sum()
            
            # Stop if we overshoot
            if test_vol > original_vol * 1.05:
                break
            
            compensated = test
            
            # Stop if close enough
            if test_vol >= original_vol * 0.95:
                break
        
        return compensated

    def smooth_artifact(self, artifact_3d, preserve_volume=True):
        # Step 1: Remove small objects
        smoothed = morphology.remove_small_objects(artifact_3d, min_size=20)
        
        # Step 2: Fill holes in each 2D slice (faster than 3D)
        for z in range(smoothed.shape[2]):
            smoothed[:, :, z] = binary_fill_holes(smoothed[:, :, z])
        
        # Step 3: Morphological closing (fills gaps, smooths surfaces)
        struct = generate_binary_structure(3, 2)  # 18-connectivity (more aggressive)
        smoothed = binary_closing(smoothed, structure=struct, iterations=2)
        
        # Step 4: Morphological opening (removes thin protrusions)
        smoothed = binary_opening(smoothed, structure=struct, iterations=1)
        
        # Step 5: Remove remaining small holes
        smoothed = morphology.remove_small_holes(smoothed, area_threshold=50)

        if preserve_volume:
            smoothed = self.compensate_volume_loss(artifact_3d, smoothed)

        smoothed = self.keep_largest_component(smoothed)
        
        return smoothed
    


