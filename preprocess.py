from scipy.ndimage import gaussian_filter
import SimpleITK as sitk
import numpy as np
import ants
import nibabel as nib
import os
from scipy import ndimage

class Preprocess:
    def __init__(self, template_path):
        self.template_path = template_path 

    def load_subject(self, filepath):
        img = nib.load(filepath)
        data = img.get_fdata()
        return data, img

    def normalize_minmax(self, ants_img, percentile_range=(1, 99)):
        # Convert to numpy to filter out zeroes for the quantile calculation
        data = ants_img.numpy()
        non_zero_data = data[data > 0]
        
        if non_zero_data.size == 0:
            return ants_img # Return as is if image is empty

        vmin = np.percentile(non_zero_data, percentile_range[0])
        vmax = np.percentile(non_zero_data, percentile_range[1])

        normalized_data = (data - vmin) / (vmax - vmin + 1e-8)
        
        normalized_data = np.clip(normalized_data, 0, 1)
        
        return ants_img.new_image_like(normalized_data)

    def enhance_unsharp_mask(self, ants_img, sigma=2.0, amount=1.0):
        blurred = ants.smooth_image(ants_img, sigma)

        details = ants_img - blurred

        enhanced = ants_img + (details * amount)

        return enhanced

    def bias_correction_n4itk(self, data, reference_img, mask=None, shrink_factor=4, n_iterations=[50, 50, 50, 50]):
        sitk_image = sitk.GetImageFromArray(data.transpose(2, 1, 0))
        spacing = reference_img.header.get_zooms()[:3]
        spacing = tuple(float(s) for s in spacing)  # Convert to proper format
        sitk_image.SetSpacing(spacing)

        if mask is None:
            sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1)
        else:
            sitk_mask = sitk.GetImageFromArray(mask.transpose(2, 1, 0).astype(np.uint8))
            sitk_mask.SetSpacing(reference_img.header.get_zooms()[:3])

        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(n_iterations)
        corrector.SetConvergenceThreshold(0.001)

        corrected_image = corrector.Execute(sitk_image, sitk_mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)

        corrected = sitk.GetArrayFromImage(corrected_image).transpose(2, 1, 0)
        bias_field = sitk.GetArrayFromImage(log_bias_field).transpose(2, 1, 0)

        return corrected, np.exp(bias_field)

    def register_to_template(self, subject_path, mask_path, template_path, output_dir, registration_type):
        
        template = ants.image_read(template_path)
        subject = ants.image_read(subject_path)

        subject_corr = ants.n4_bias_field_correction(subject)

        registration = ants.registration(
            fixed=template,
            moving=subject_corr,
            type_of_transform=registration_type,
            outprefix=str(output_dir / 'reg_')
        )

        # 2. Apply to the Brain Image
        reg_brain = ants.apply_transforms(
            fixed=template,
            moving=subject,
            transformlist=registration['fwdtransforms'],
            interpolator='lanczosWindowedSinc'
        )

        # 3. Apply to the Mask (using the SAME transform)

        reg_mask = None
        
        if mask_path is not None:
            mask = ants.image_read(mask_path)
            mask = ants.copy_image_info(subject, mask)
            reg_mask = ants.apply_transforms(
                fixed=template,
                moving=mask,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'
            )

        return reg_brain, reg_mask, template

    def preprocess_subject(self, subject_path, mask_path, output_dir):
        reg_brain, reg_mask, template = self.register_to_template(
            subject_path, mask_path, self.template_path, output_dir, registration_type='SyN')

        processed_reg_brain = self.normalize_minmax(reg_brain)

        processed_reg_brain = self.enhance_unsharp_mask(processed_reg_brain)

        # reg_mask = ants.threshold_image(reg_mask, 0.5, 1.0)

        return processed_reg_brain, reg_mask, template

