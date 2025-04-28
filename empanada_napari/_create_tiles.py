from magicgui import magicgui, widgets
import napari
from typing import Optional
import os
import numpy as np
import json
import tifffile as tif
from tqdm import tqdm

def chop_up_2d_im_into_patches(image_path, mask_path, PATCH_SIZE, save_directory):
    """
    Chop up a large BigTIFF image (and optionally a corresponding mask) into
    PATCH_SIZE x PATCH_SIZE patches. Uses memory mapping to avoid loading
    the whole image into memory. If the image does not evenly divide by PATCH_SIZE,
    the patches at the border are padded with zeros. Metadata (including
    original/padded shapes and mask availability) is saved in a dummy TIFF file.
    """
    # Open image as a memory-mapped array using tif.memmap
    try:
        im = tif.memmap(image_path)
    except Exception:
        print("Error reading {} tif as memmap, trying to load full image to memory".format(image_path))
        im = tif.imread(image_path)

    print('Original image data type: {}'.format(im.dtype))
    print('Original image shape: {}'.format(im.shape))
    original_shape = im.shape

    # check if the image is 2D
    if len(original_shape) != 2:
        raise ValueError("Input image must be 2D (height x width).")

    # Compute padded shape.
    pad_h = (PATCH_SIZE - (original_shape[0] % PATCH_SIZE)) % PATCH_SIZE
    pad_w = (PATCH_SIZE - (original_shape[1] % PATCH_SIZE)) % PATCH_SIZE
    padded_shape = (original_shape[0] + pad_h, original_shape[1] + pad_w)
    print("Padded shape will be: {}".format(padded_shape))

    # If a mask path is provided, open the mask as a memory-mapped array.
    if mask_path is not None and os.path.exists(mask_path):
        try:
            mask = tif.memmap(mask_path)
        except Exception:
            print("Error reading {} tif as memmap, trying to load full image to memory".format(mask_path))
            mask = tif.imread(mask_path)
        assert im.shape == mask.shape, "Image and mask must have the same shape."
    else:
        mask = None

    # Create output directories.
    im_dir = os.path.join(save_directory, 'im')
    os.makedirs(im_dir, exist_ok=True)

    if mask is not None:
        msk_dir = os.path.join(save_directory, 'msk')
        os.makedirs(msk_dir, exist_ok=True)

    # Loop over grid positions and extract patches.
    for i in tqdm(range(0, padded_shape[0], PATCH_SIZE), desc="Processing rows"):
        for j in range(0, padded_shape[1], PATCH_SIZE):
            # Determine end indices (do not exceed original dimensions)
            end_i = min(i + PATCH_SIZE, original_shape[0])
            end_j = min(j + PATCH_SIZE, original_shape[1])

            # Extract image patch.
            im_patch = im[i:end_i, j:end_j]

            # Pad patch if necessary.
            if im_patch.shape[0] < PATCH_SIZE or im_patch.shape[1] < PATCH_SIZE:
                pad_rows = PATCH_SIZE - im_patch.shape[0]
                pad_cols = PATCH_SIZE - im_patch.shape[1]
                im_patch = np.pad(im_patch, ((0, pad_rows), (0, pad_cols)), mode='constant')

            im_patch_filename = os.path.join(im_dir, f'{i}_{j}.tif')
            tif.imwrite(im_patch_filename, im_patch, bigtiff=True)

            # If mask exists, process it similarly.
            if mask is not None:
                msk_patch = mask[i:end_i, j:end_j]
                if msk_patch.shape[0] < PATCH_SIZE or msk_patch.shape[1] < PATCH_SIZE:
                    pad_rows = PATCH_SIZE - msk_patch.shape[0]
                    pad_cols = PATCH_SIZE - msk_patch.shape[1]
                    msk_patch = np.pad(msk_patch, ((0, pad_rows), (0, pad_cols)), mode='constant')
                msk_patch_filename = os.path.join(msk_dir, f'{i}_{j}.tif')
                tif.imwrite(msk_patch_filename, msk_patch, bigtiff=True)

    # Save metadata to a dummy TIFF file (includes a flag for mask availability).
    metadata = {
        "original_image_shape": original_shape,
        "padded_image_shape": padded_shape,
        "mask_available": mask is not None
    }
    metadata_filename = os.path.join(save_directory, "metadata.tif")
    dummy = np.zeros((1, 1), dtype=np.uint8)
    tif.imwrite(metadata_filename, dummy, description=json.dumps(metadata), bigtiff=True)
    print("Metadata saved to {}".format(metadata_filename))



def put_patches_back_together(patch_directory, save_directory, original_image_shape=None,
                              padded_image_shape=None):
    """
    Merge image (and optionally mask) patches from the patch_directory back together.
    If the original and padded shapes are not provided via command line, they are read from the metadata TIFF.
    If a mask folder exists, the mask is merged; otherwise, only the image is processed.
    The merged images are saved with the metadata embedded.
    """
    # Load metadata if shapes not provided.
    metadata_filename = os.path.join(patch_directory, "metadata.tif")
    if original_image_shape is None or padded_image_shape is None:
        if os.path.exists(metadata_filename):
            with tif.TiffFile(metadata_filename) as tif_meta:
                description = tif_meta.pages[0].description
            metadata = json.loads(description)
            original_image_shape = tuple(metadata["original_image_shape"])
            padded_image_shape = tuple(metadata["padded_image_shape"])
            mask_available = metadata.get("mask_available", False)
            print("Loaded metadata: original shape {}, padded shape {}, mask_available: {}".format(
                original_image_shape, padded_image_shape, mask_available))
        else:
            raise ValueError("Metadata file not found and image shapes not provided.")
    else:
        # If shapes are provided, try to determine if mask is available by checking for 'msk' folder.
        mask_available = os.path.exists(os.path.join(patch_directory, 'msk'))

    im_dir = os.path.join(patch_directory, 'im')
    im_files = sorted([f for f in os.listdir(im_dir) if f.lower().endswith(('.tif', '.tiff'))])

    # Merge image patches.
    merged_image = np.zeros(padded_image_shape, dtype=np.uint8)
    for file in tqdm(im_files, desc="Merging image patches"):
        coords = file.split('.')[0].split('_')
        i_coord, j_coord = int(coords[0]), int(coords[1])
        im_patch = tif.imread(os.path.join(im_dir, file))
        patch_shape = im_patch.shape
        merged_image[i_coord:i_coord + patch_shape[0], j_coord:j_coord + patch_shape[1]] = im_patch

    merged_image_unpadded = merged_image[:original_image_shape[0], :original_image_shape[1]]
    image_out_path = os.path.join(save_directory, 'im.tif')

    # Save merged image with metadata.
    metadata = {
        "original_image_shape": original_image_shape,
        "padded_image_shape": padded_image_shape,
        "mask_available": mask_available
    }
    tif.imwrite(image_out_path, merged_image_unpadded, description=json.dumps(metadata), bigtiff=True)
    print("Merged image saved to {}".format(image_out_path))

    # If a mask is available, merge mask patches.
    if mask_available:
        msk_dir = os.path.join(patch_directory, 'msk')
        msk_files = sorted([f for f in os.listdir(msk_dir) if f.lower().endswith(('.tif', '.tiff'))])
        merged_mask = np.zeros(padded_image_shape, dtype=np.uint16)
        for file in tqdm(msk_files, desc="Merging mask patches"):
            coords = file.split('.')[0].split('_')
            i_coord, j_coord = int(coords[0]), int(coords[1])
            msk_patch = tif.imread(os.path.join(msk_dir, file))
            patch_shape = msk_patch.shape
            merged_mask[i_coord:i_coord + patch_shape[0], j_coord:j_coord + patch_shape[1]] = msk_patch

        merged_mask_unpadded = merged_mask[:original_image_shape[0], :original_image_shape[1]]
        mask_out_path = os.path.join(save_directory, 'msk.tif')
        tif.imwrite(mask_out_path, merged_mask_unpadded, description=json.dumps(metadata), bigtiff=True)
        print("Merged mask saved to {}".format(mask_out_path))
    else:
        print("No mask data available; skipping mask merging.")
    return


def create_tiles():
    """
    Creates square tiles from a given big tif file, optional mask file and tile size
    """

    gui_params = dict(
        image_path=dict(widget_type='FileEdit', label='Path to tif image'),
        mask_path=dict(widget_type='FileEdit', label='Path to tif mask (optional)'),
        patch_size=dict(widget_type='FloatSpinBox', min=1, max=300000, step=1, value=512, label="Tile size (px)",
                        tooltip='Tile size (px)'),
        save_directory=dict(widget_type='FileEdit', mode='d', label='Directory to save tiles', value=os.getcwd(),
                            tooltip='Directory to save tiles and metadata (if applicable)'),
        call_button="Create Tiles",
    )

    @magicgui(
        **gui_params,
    )
    def create_tiles_widget(
            viewer: napari.viewer.Viewer,
            image_path: str,
            mask_path: Optional[str] = None,
            patch_size: Optional[float] = 512,
            save_directory: Optional[str] = os.getcwd()
    ):
        """
        GUI widget to create tiles from a large image and optional mask.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Call the function to chop up the image and mask into patches
        chop_up_2d_im_into_patches(image_path, mask_path, int(patch_size), save_directory)

        print(f"Tiles created and saved to {save_directory}")

    return create_tiles_widget


def merge_tiles():
    """
    Merges tiles from a given directory back into a single image and optional mask
    """

    gui_params = dict(
        patch_directory=dict(widget_type='FileEdit', mode='d', label='Directory with patches (same as from "Create Tiles" option)',
                             value=os.getcwd(),
                             tooltip='Directory containing image patches (and optional mask patches)'),
        save_directory=dict(widget_type='FileEdit', mode='d', label='Directory to save merged image', value=os.getcwd(),
                            tooltip='Directory to save merged image and mask (if applicable)')
    )

    @magicgui(
        call_button='Merge Tiles',
        **gui_params,
        layout='vertical'
    )
    def merge_tiles_widget(
            viewer: napari.viewer.Viewer,
            patch_directory: str,
            save_directory: str,
    ):
        """
        GUI widget to merge tiles back into a single image and optional mask.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Call the function to put patches back together
        put_patches_back_together(patch_directory, save_directory)

        print(f"Merged image saved to {save_directory}")

    return merge_tiles_widget
