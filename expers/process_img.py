import nibabel as nib
import numpy as np
from monai.transforms import Resize

def resample_image(img, target_size=(128, 128, 128)):
    """
    Resample a 3D image to a fixed target size using MONAI Resize.
    Ensure the input image is 3D, expand to 4D before processing,
    and remove extra dimensions afterward.
    """
    # check image shape
    if len(img.shape) == 4:
        print(f"Warning: Unexpected 4D image shape detected {img.shape}. Removing batch dimension.")
        img = img[0]  # 移remove batch dimension
    
    if len(img.shape) != 3:
        raise ValueError(f"Invalid image shape: {img.shape}, expected 3D.")

    # expand to 4D
    img_4d = img[np.newaxis, np.newaxis, ...]

    # run resampling
    resample_transform = Resize(spatial_size=target_size, mode="trilinear")
    resampled_img = resample_transform(img_4d)

    # remove extra dimensions
    return resampled_img[0, 0]


def normalize_intensity(img, a_min=-42, a_max=423):
    """
    Normalize image intensity values to range [0, 1] based on min and max.
    """
    img = np.clip(img, a_min, a_max)
    img = (img - a_min) / (a_max - a_min)
    return img


def process_and_save(input_path, output_path):
    """
    Load an image, perform normalization and resampling, and save it.
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
    except Exception as e:
        print(f"Error during processing: {e}")
