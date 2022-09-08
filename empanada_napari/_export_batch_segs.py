import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui

def export_batch_segs():
    import os
    import math
    import dask.array as da
    import numpy as np
    from skimage import io

    def _get_impaths_from_dask(dask_array):
        # delayed keys
        keys = [l for l in dask_array.dask.layers if 'imread' in l]
        # absolute image paths
        return [dask_array.dask[k][1] for k in keys]

    @magicgui(
        call_button='Export labels',
        layout='vertical',
        save_dir=dict(widget_type='FileEdit', value='', label='Save directory', mode='d', tooltip='directory in which to save segmentations'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        labels_layer: Labels,
        save_dir
    ):
        image = image_layer.data
        mask = labels_layer.data

        assert len(image) == len(mask), \
        f"Image and labels layer must have the same number of images, got {len(image)} and {len(mask)}"

        if image.ndim == 3:
            if isinstance(image, da.Array):
                imnames = [
                    '.'.join(os.path.basename(imp).split('.')[:-1]) + '.tiff' 
                    for imp in _get_impaths_from_dask(image)
                ]
            else:
                zpad = math.ceil(math.log(len(image), 10))
                imnames = [str(n).zfill(zpad) + '.tiff' for n in range(len(image))]

            for i in range(len(mask)):
                imname = imnames[i]
                if isinstance(image, da.Array):
                    h, w = image[i].compute().shape
                else:
                    h, w = image[i].shape

                seg = np.squeeze(mask[i, :h, :w]).astype(np.int32)
                io.imsave(os.path.join(save_dir, imname), seg, check_contrast=False)

        else:
            imname = image_layer.name + '.tiff'
            io.imsave(os.path.join(save_dir, imname), mask.astype(np.int32), check_contrast=False)

        print('Segmentations exported!')

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def export_batch_segs_widget():
    return export_batch_segs, {'name': 'Export 2D segmentations'}
