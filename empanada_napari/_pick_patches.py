import os
import napari
import string
import random
import numpy as np
import dask.array as da
from magicgui import magicgui
from napari_plugin_engine import napari_hook_implementation
from empanada.array_utils import take

def pick_patches():
    from napari.qt.threading import thread_worker

    def _pad_flipbook(flipbook, size):
        assert flipbook.ndim == 3

        h, w = flipbook.shape[1:]
        assert size[0] >= h and size[1] >= w

        ph, pw = size[0] - h, size[1] - w

        flipbook = np.pad(flipbook, ((0, 0), (0, ph), (0, pw)))
        assert flipbook.shape[1] == size[0]
        assert flipbook.shape[2] == size[1]

        return flipbook

    @thread_worker
    def _random_crops(volume, patch_size, num_patches, points, isotropic):
        flipbooks = []
        locs = []
        for _ in range(num_patches):
            if isotropic:
                axes = [0, 1, 2]
                axis = random.choice(axes)

                # set height and width axes
                del axes[axes.index(axis)]
                ha, wa = axes
            else:
                axis = 0
                ha, wa = 1, 2

            if points and points is not None:
                patch_ctr = points.pop(0)
                plane = patch_ctr[axis]
                plane = max(2, plane)
                plane = min(volume.shape[axis] - 3, plane)
                fb_slice = slice(plane - 2, plane + 3)

                ys = int(patch_ctr[ha] - patch_size / 2)
                ys = min(ys, volume.shape[ha] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[wa] - patch_size / 2)
                xs = min(xs, volume.shape[wa] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, volume.shape[ha])
                xe = min(xs + patch_size, volume.shape[wa])
            else:
                # pick a plane from sample of every 3
                plane = np.random.randint(2, volume.shape[axis] // 3) * 3
                fb_slice = slice(plane - 2, plane + 3)

                ys = np.random.choice(np.arange(0, max(1, volume.shape[ha] - patch_size), patch_size))
                xs = np.random.choice(np.arange(0, max(1, volume.shape[wa] - patch_size), patch_size))
                ye = min(ys + patch_size, volume.shape[ha])
                xe = min(xs + patch_size, volume.shape[wa])

            flipbook = take(volume, fb_slice, axis)
            flipbook = take(flipbook, slice(ys, ye), ha)
            flipbook = take(flipbook, slice(xs, xe), wa)

            if type(flipbook) == da.core.Array:
                flipbook = flipbook.compute()

            if axis == 1:
                flipbook = flipbook.transpose(1, 0, 2)
            elif axis == 2:
                flipbook = flipbook.transpose(2, 0, 1)

            flipbook = _pad_flipbook(flipbook, (patch_size, patch_size))

            flipbooks.append(flipbook)
            locs.append((axis, fb_slice.start, fb_slice.stop, ys, ye, xs, xe))

        return np.stack(flipbooks, axis=0), locs

    gui_params = dict(
        num_patches=dict(widget_type='SpinBox', value=8, min=2, max=32, step=1, label='Number of patches for annotation'),
        patch_size=dict(widget_type='SpinBox', value=256, min=224, max=512, step=16, label='Patch size in pixels'),
        isotropic=dict(widget_type='CheckBox', text='Take xy,xz,yz', value=False, tooltip='If volume has isotropic voxels, pick patches from all planes.'),
    )

    @magicgui(
        call_button='Pick patches',
        layout='vertical',
        **gui_params
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: napari.layers.Image,
        points_layer: napari.layers.Points,
        num_patches,
        patch_size,
        isotropic
    ):

        volume = image_layer.data
        name = image_layer.name
        assert volume.ndim == 3, "Must be 3D data!"

        if points_layer is not None:
            local_points = []
            for pt in points_layer.data:
                local_points.append(tuple([int(c) for c in image_layer.world_to_data(pt)]))
        else:
            local_points = None

        def _show_flipbooks(*args):
            flipbooks, locs = args[0]

            metadata = {
                'prefix': f'{name}',
                'suffices': [f'-LOC-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}_{l[5]}-{l[6]}' for l in locs]
            }

            viewer.add_image(flipbooks, name=f'{name}_flipbooks', metadata=metadata, visible=True)

        crop_worker = _random_crops(volume, patch_size, num_patches, local_points, isotropic)
        crop_worker.returned.connect(_show_flipbooks)
        crop_worker.start()

        # remove all points that
        # were chosen as patches
        if points_layer is not None:
            points_layer.data = points_layer.data[num_patches:]

    return widget

def store_dataset():
    from skimage import io

    def _random_suffix(size=10):
        # printing letters
        letters = string.ascii_letters
        digits = string.digits

        pstr = []
        for _ in range(size):
            if random.random() < 0.5:
                pstr.append(random.choice(letters))
            else:
                pstr.append(random.choice(digits))

        return ''.join(pstr)

    gui_params = dict(
        save_dir=dict(widget_type='FileEdit', value='./', label='Save directory', mode='d', tooltip='Directory in which to create a new dataset.'),
        dataset_name=dict(widget_type='LineEdit', value='', label='Dataset name', tooltip='Name to use for the dataset, creates a directory in the Save Directory with this name')
    )

    @magicgui(
        call_button='Save flipbooks',
        layout='vertical',
        **gui_params
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: napari.layers.Image,
        labels_layer: napari.layers.Labels,
        save_dir,
        dataset_name
    ):

        assert dataset_name, "Must provide a dataset name!"

        flipbooks = image_layer.data
        flipbook_labels = labels_layer.data
        assert flipbooks.ndim == 4, "Flipbooks are expected to be 4D. Did you pick the right image layer?"
        assert flipbook_labels.ndim == 4, "Labels are expected to be 4D. Did you pick the right labels layer?"
        assert flipbooks.shape[1] == 5, "Flipbooks must be 5 slices long!"

        # labels can span entire image space,
        # crop them to only span the given flipbook space
        if not np.allclose(flipbooks.shape, flipbook_labels.shape):
            assert all([sfl >= sf for sfl,sf in zip(flipbook_labels.shape, flipbooks.shape)])
            a,b,c,d = flipbooks.shape
            flipbook_labels = flipbook_labels[:a, :b, :c, :d]
            print(f'Cropped labels to match flipbook image size.')

        if image_layer.metadata:
            has_metadata = True
            prefix = image_layer.metadata['prefix']
            suffices = image_layer.metadata['suffices']
        else:
            has_metadata = False
            prefix = 'unknown'
            suffices = ['-' + _random_suffix() for _ in range(len(flipbooks))]

        # get middle images of flipbooks
        images = flipbooks[:, 2]
        masks = flipbook_labels[:, 2]

        outdir = os.path.join(save_dir, dataset_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(os.path.join(outdir, f'{prefix}/images'), exist_ok=True)
            os.makedirs(os.path.join(outdir, f'{prefix}/masks'), exist_ok=True)
            print('Created directory', outdir)

        for sfx,img,msk in zip(suffices, images, masks):
            fname = f'{prefix}{sfx}.tiff'

            # if we have metadata, use it to crop image and mask
            if has_metadata:
                hrange, wrange = sfx.split('_')[-2:]
                hmin, hmax = hrange.split('-')
                wmin, wmax = wrange.split('-')
                h, w = int(hmax) - int(hmin), int(wmax) - int(wmin)

                img = img[:h, :w]
                msk = msk[:h, :w]

            io.imsave(os.path.join(outdir, f'{prefix}/images/{fname}'), img, check_contrast=False)

            if msk.max() <= 255:
                dtype = np.uint8
            else:
                dtype = np.uint32

            io.imsave(os.path.join(outdir, f'{prefix}/masks/{fname}'), msk.astype(dtype), check_contrast=False)

        print('Finished saving.')

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def pick_patches_widget():
    return pick_patches, {'name': 'Pick training patches'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def store_dataset_widget():
    return store_dataset, {'name': 'Store training dataset'}
