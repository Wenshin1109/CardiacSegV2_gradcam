import nibabel as nib
import numpy as np
from monai.transforms import Resize

def resample_image(img, target_size=(128, 128, 128)):
    resample_transform = Resize(spatial_size=target_size, mode="trilinear")
    return resample_transform(img[np.newaxis, np.newaxis, ...])[0, 0]

def normalize_intensity(img, a_min=-42, a_max=423):
    img = np.clip(img, a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img

def process_and_save(input_path, output_path):
    try:
        print(f"Loading image: {input_path}")
        img = nib.load(input_path).get_fdata()
        
        print("Performing intensity normalization and resampling...")
        normalized_img = normalize_intensity(img)
        resampled_img = resample_image(normalized_img)
        
        print(f"Processed image shape: {resampled_img.shape}")
        
        # Save as a new NIfTI file
        processed_img_nii = nib.Nifti1Image(resampled_img, affine=np.eye(4))
        nib.save(processed_img_nii, output_path)
        print(f"Image saved to: {output_path}")
    except Exception as e:
        print(f"Error during processing: {e}")
