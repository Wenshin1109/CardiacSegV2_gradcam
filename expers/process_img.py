import nibabel as nib
from monai.transforms import Resize
import numpy as np
import torch

def resample_image(img, target_size=(128, 128, 128)):
    """
    Resample a 3D image using MONAI Resize.
    Ensure input image is 3D, expand to 4D for processing,
    and remove extra dimensions afterward.
    """
    print(f"Image shape before processing: {img.shape}")

    # check image shape
    if len(img.shape) == 3:
        print("Expanding image to 4D...")
        img_4d = img[np.newaxis, ...]  # (B, D, H, W)
    else:
        raise ValueError(f"Invalid input image shape: {img.shape}. Expected 3D.")

    print(f"Image shape after expansion: {img_4d.shape}")

    # determine target spatial size
    spatial_size = target_size
    print(f"Using target spatial size: {spatial_size}")

    # resample image
    resample_transform = Resize(spatial_size=spatial_size, mode="trilinear")
    resampled_img = resample_transform(img_4d)
    resampled_img = np.asarray(resampled_img[0], dtype=np.float32)


    print(f"Resampled image shape: {resampled_img.shape}")

    
    return resampled_img






def normalize_intensity(img, a_min=-42, a_max=423):
    """
    Normalize image intensity values to range [0, 1] based on min and max.
    """
    img = np.clip(img, a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img


def process_and_save(input_path, output_path):
    """
    Load, normalize, resample, and save the image.
    Automatically adjust target size if needed.
    """
    try:
        print(f"Loading image: {input_path}")
        img = nib.load(input_path).get_fdata()

        print(f"Original image shape: {img.shape}")

        print("Performing intensity normalization and resampling...")
        normalized_img = normalize_intensity(img)
        resampled_img = resample_image(normalized_img)

        print(f"Processed image shape: {resampled_img.shape}")

        # save processed image
        processed_img_nii = nib.Nifti1Image(resampled_img, affine=np.eye(4))
        nib.save(processed_img_nii, output_path)
        print(f"Image saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
    except ValueError as e:
        print(f"Image shape validation failed: {e}")
    except Exception as e:
        print(f"Unexpected error during processing: {e}")
