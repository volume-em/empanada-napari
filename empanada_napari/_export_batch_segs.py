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

    save_ops = {
        '2D images': '2D images',
        '3D image': '3D image',
    }

    def _get_impaths_from_dask(dask_array):
        # delayed keys
        keys = [l for l in dask_array.dask.layers if 'imread' in l]
        # absolute image paths
        return [dask_array.dask[k][1] for k in keys]

    @magicgui(
        call_button='Export labels',
        layout='vertical',
        dataset_name=dict(widget_type='LineEdit', value='', label='Folder name',
                          tooltip='Name to use for the dataset folder, creates a directory in the Save Directory with this name, it appends if directory already exists.'),
        save_dir=dict(widget_type='FileEdit', value='', label='Save directory', mode='d', tooltip='Directory in which to save segmentations'),
        export_type=dict(widget_type='RadioButtons', choices=list(save_ops.keys()), value=list(save_ops.keys())[0], label='Export type', tooltip='Exports segmentations as individual 2D images or a single 3d image.'),
        grayscale=dict(widget_type='CheckBox', value=False, label='Export grayscale image', tooltip='Convert to grayscale'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        labels_layer: Labels,
        export_type: str,
        dataset_name: str,
        save_dir: str,
        grayscale: bool,
    ):
        assert dataset_name, "Must provide a dataset name!"
        assert len(viewer.dims.order) <= 3, "Please use 'Save training patches' instead."

        outdir = os.path.join(save_dir, dataset_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
            print('Created directory', outdir)
        else:
            print('Adding images to existing directory', outdir)

        if grayscale:
            os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'masks'), exist_ok=True)

        export_option = save_ops[export_type]
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
                imnames = [image_layer.name + '_' + str(n).zfill(zpad) + '.tiff' for n in range(len(image))]
                # imnames = [str(n).zfill(zpad) + '.tiff' for n in range(len(image))]

            if export_option == '3D image':
                # Creates a 3D image or 2D stack of images from the label image layer
                i, h, w = image.shape
                seg_stack = np.squeeze(mask[:i, :h, :w]).astype(np.int32)
                imname = image_layer.name + '.tiff'

                if grayscale:
                    img_stack = np.squeeze(image[:i, :h, :w])
                    io.imsave(os.path.join(outdir, f'images/{imname}'), img_stack, check_contrast=False)
                io.imsave(os.path.join(outdir, f'masks/{imname}'), seg_stack, check_contrast=False)

            else:
                for i in range(len(image)):
                    imname = imnames[i]
                    if isinstance(image, da.Array):
                        h, w = image[i].compute().shape
                    else:
                        h, w = image[i].shape

                    img = np.squeeze(image[i, :h, :w])
                    seg = np.squeeze(mask[i, :h, :w]).astype(np.int32)

                    if grayscale:
                        io.imsave(os.path.join(outdir, f'images/{imname}'), img, check_contrast=False)
                    io.imsave(os.path.join(outdir, f'masks/{imname}'), seg, check_contrast=False)

        else:
            imname = image_layer.name + '.tiff'
            if grayscale:
                #imname = image_layer.name + '.tiff'
                io.imsave(os.path.join(outdir, f'images/{imname}'), image, check_contrast=False)
            io.imsave(os.path.join(outdir, f'masks/{imname}'), mask.astype(np.int32), check_contrast=False)


        print('Segmentations exported!')

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def export_batch_segs_widget():
    return export_batch_segs, {'name': 'Export Segmentations'}
