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
    
    def _pad_patch(patch, size):
        h, w = patch.shape
        assert size[0] >= h and size[1] >= w

        ph, pw = size[0] - h, size[1] - w

        patch = np.pad(patch, ((0, ph), (0, pw)))
        assert patch .shape[0] == size[0]
        assert patch.shape[1] == size[1]

        return patch

    def _pad_label_patch(label_patch, size):
        h, w = label_patch.shape
        assert size[0] >= h and size[1] >= w

        ph, pw = size[0] - h, size[1] - w

        label_patch = np.pad(label_patch, ((0, ph), (0, pw)))
        assert label_patch.shape[0] == size[0]
        assert label_patch.shape[1] == size[1]

        return label_patch

    def _pad_flipbook(flipbook, size):
        assert flipbook.ndim == 3

        h, w = flipbook.shape[1:]
        assert size[0] >= h and size[1] >= w

        ph, pw = size[0] - h, size[1] - w

        flipbook = np.pad(flipbook, ((0, 0), (0, ph), (0, pw)))
        assert flipbook.shape[1] == size[0]
        assert flipbook.shape[2] == size[1]

        return flipbook

    def _pad_label_flipbook(label_flipbook, size):
        assert label_flipbook.ndim == 3

        h, w = label_flipbook.shape[1:]
        assert size[0] >= h and size[1] >= w

        ph, pw = size[0] - h, size[1] - w

        label_flipbook = np.pad(label_flipbook, ((0, 0), (0, ph), (0, pw)))
        assert label_flipbook.shape[1] == size[0]
        assert label_flipbook.shape[2] == size[1]

        return label_flipbook

    @thread_worker
    def _pick_patches(image, patch_size, num_patches, points):
        patches = []
        locs = []
        for _ in range(num_patches):
            plane = None
            if points and points is not None:
                patch_ctr = points.pop(0)                
                if len(patch_ctr) == 2:
                    ys = int(patch_ctr[0] - patch_size / 2)
                    ys = min(ys, image.shape[0] - patch_size)
                    ys = max(ys, 0)
                    xs = int(patch_ctr[1] - patch_size / 2)
                    xs = min(xs, image.shape[1] - patch_size)
                    xs = max(xs, 0)

                    ye = min(ys + patch_size, image.shape[0])
                    xe = min(xs + patch_size, image.shape[1])
                    patch = image[ys:ye, xs:xe]
                else:
                    plane = patch_ctr[0]
                    ys = int(patch_ctr[1] - patch_size / 2)
                    ys = min(ys, image.shape[1] - patch_size)
                    ys = max(ys, 0)
                    xs = int(patch_ctr[2] - patch_size / 2)
                    xs = min(xs, image.shape[2] - patch_size)
                    xs = max(xs, 0)

                    ye = min(ys + patch_size, image.shape[1])
                    xe = min(xs + patch_size, image.shape[2])
                    patch = image[plane, ys:ye, xs:xe]
            else:
                if image.ndim == 2:
                    ys = np.random.choice(np.arange(0, max(1, image.shape[0] - patch_size), patch_size))
                    xs = np.random.choice(np.arange(0, max(1, image.shape[1] - patch_size), patch_size))
                    ye = min(ys + patch_size, image.shape[0])
                    xe = min(xs + patch_size, image.shape[1])
                    patch = image[ys:ye, xs:xe]
                else:
                    plane = np.random.randint(0, image.shape[0]) 
                    ys = np.random.choice(np.arange(0, max(1, image.shape[1] - patch_size), patch_size))
                    xs = np.random.choice(np.arange(0, max(1, image.shape[2] - patch_size), patch_size))
                    ye = min(ys + patch_size, image.shape[1])
                    xe = min(xs + patch_size, image.shape[2])
                    patch = image[plane, ys:ye, xs:xe]

            if type(patch) == da.core.Array:
                patch = patch.compute()

            patch = _pad_patch(patch, (patch_size, patch_size))

            patches.append(patch)
            if plane is None:
                locs.append((ys, ye, xs, xe))
            else:
                locs.append((plane, ys, ye, xs, xe))

        return np.stack(patches, axis=0), locs

    @thread_worker
    def _pick_paired_patches(image, label, patch_size, num_patches, points):
        patches = []
        label_patches = []
        locs = []
        for _ in range(num_patches):
            plane = None
            if points and points is not None:
                patch_ctr = points.pop(0)
                if len(patch_ctr) == 2:
                    ys = int(patch_ctr[0] - patch_size / 2)
                    ys = min(ys, image.shape[0] - patch_size)
                    ys = max(ys, 0)
                    xs = int(patch_ctr[1] - patch_size / 2)
                    xs = min(xs, image.shape[1] - patch_size)
                    xs = max(xs, 0)

                    ye = min(ys + patch_size, image.shape[0])
                    xe = min(xs + patch_size, image.shape[1])
                    patch = image[ys:ye, xs:xe]

                    label_patch = label[ys:ye, xs:xe]
                else:
                    plane = patch_ctr[0]
                    ys = int(patch_ctr[1] - patch_size / 2)
                    ys = min(ys, image.shape[1] - patch_size)
                    ys = max(ys, 0)
                    xs = int(patch_ctr[2] - patch_size / 2)
                    xs = min(xs, image.shape[2] - patch_size)
                    xs = max(xs, 0)

                    ye = min(ys + patch_size, image.shape[1])
                    xe = min(xs + patch_size, image.shape[2])
                    patch = image[plane, ys:ye, xs:xe]

                    label_patch = label[plane, ys:ye, xs:xe]
            else:
                if image.ndim == 2:
                    ys = np.random.choice(np.arange(0, max(1, image.shape[0] - patch_size), patch_size))
                    xs = np.random.choice(np.arange(0, max(1, image.shape[1] - patch_size), patch_size))
                    ye = min(ys + patch_size, image.shape[0])
                    xe = min(xs + patch_size, image.shape[1])
                    patch = image[ys:ye, xs:xe]

                    label_patch = label[ys:ye, xs:xe]
                else:
                    plane = np.random.randint(0, image.shape[0])
                    ys = np.random.choice(np.arange(0, max(1, image.shape[1] - patch_size), patch_size))
                    xs = np.random.choice(np.arange(0, max(1, image.shape[2] - patch_size), patch_size))
                    ye = min(ys + patch_size, image.shape[1])
                    xe = min(xs + patch_size, image.shape[2])
                    patch = image[plane, ys:ye, xs:xe]

                    label_patch = label[plane, ys:ye, xs:xe]

            if type(patch) and type(label_patch) == da.core.Array:
                patch = patch.compute()
                label_patch = label_patch.compute()

            patch = _pad_patch(patch, (patch_size, patch_size))
            label_patch = _pad_label_patch(label_patch, (patch_size, patch_size))

            patches.append(patch)
            label_patches.append(label_patch)
            if plane is None:
                locs.append((ys, ye, xs, xe))
            else:
                locs.append((plane, ys, ye, xs, xe))

        return np.stack(patches, axis=0), np.stack(label_patches, axis=0), locs

    @thread_worker
    def _pick_flipbooks(image, patch_size, num_patches, points, isotropic):
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
                plane = min(image.shape[axis] - 3, plane)
                fb_slice = slice(plane - 2, plane + 3)

                ys = int(patch_ctr[ha] - patch_size / 2)
                ys = min(ys, image.shape[ha] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[wa] - patch_size / 2)
                xs = min(xs, image.shape[wa] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, image.shape[ha])
                xe = min(xs + patch_size, image.shape[wa])
            else:
                # pick a plane from sample of every 3
                plane = np.random.randint(2, image.shape[axis] // 3) * 3
                fb_slice = slice(plane - 2, plane + 3)

                ys = np.random.choice(np.arange(0, max(1, image.shape[ha] - patch_size), patch_size))
                xs = np.random.choice(np.arange(0, max(1, image.shape[wa] - patch_size), patch_size))
                ye = min(ys + patch_size, image.shape[ha])
                xe = min(xs + patch_size, image.shape[wa])

            flipbook = take(image, fb_slice, axis)
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

    @thread_worker
    def _pick_paired_flipbooks(image, label, patch_size, num_patches, points, isotropic):
        flipbooks = []
        label_flipbooks = []
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
                img_plane = min(image.shape[axis] - 3, plane)
                label_plane = min(label.shape[axis] - 3, plane)
                img_fb_slice = slice(img_plane - 2, img_plane + 3)
                label_fb_slice = slice(label_plane - 2, label_plane + 3)

                ys = int(patch_ctr[ha] - patch_size / 2)
                ys = min(ys, image.shape[ha] - patch_size)
                ys = max(ys, 0)
                xs = int(patch_ctr[wa] - patch_size / 2)
                xs = min(xs, image.shape[wa] - patch_size)
                xs = max(xs, 0)

                ye = min(ys + patch_size, image.shape[ha])
                xe = min(xs + patch_size, image.shape[wa])
            else:
                # pick a plane from sample of every 3
                plane = np.random.randint(2, image.shape[axis] // 3) * 3
                img_fb_slice = slice(plane - 2, plane + 3)
                label_fb_slice = slice(plane - 2, plane + 3)

                ys = np.random.choice(np.arange(0, max(1, image.shape[ha] - patch_size), patch_size))
                xs = np.random.choice(np.arange(0, max(1, image.shape[wa] - patch_size), patch_size))
                ye = min(ys + patch_size, image.shape[ha])
                xe = min(xs + patch_size, image.shape[wa])

            flipbook = take(image, img_fb_slice, axis)
            flipbook = take(flipbook, slice(ys, ye), ha)
            flipbook = take(flipbook, slice(xs, xe), wa)

            label_flipbook = take(label, label_fb_slice, axis)
            label_flipbook = take(label_flipbook, slice(ys, ye), ha)
            label_flipbook = take(label_flipbook, slice(xs, xe), wa)

            if type(flipbook) and type(label_flipbook) == da.core.Array:
                flipbook = flipbook.compute()
                label_flipbook = label_flipbook.compute()

            if axis == 1:
                flipbook = flipbook.transpose(1, 0, 2)
                label_flipbook = label_flipbook.transpose(1, 0, 2)
            elif axis == 2:
                flipbook = flipbook.transpose(2, 0, 1)
                label_flipbook = label_flipbook.transpose(2, 0, 1)

            flipbook = _pad_flipbook(flipbook, (patch_size, patch_size))
            label_flipbook = _pad_label_flipbook(label_flipbook, (patch_size, patch_size))

            flipbooks.append(flipbook)
            label_flipbooks.append(label_flipbook)
            locs.append((axis, img_fb_slice.start, img_fb_slice.stop, ys, ye, xs, xe))

        return np.stack(flipbooks, axis=0), np.stack(label_flipbooks, axis=0), locs

    gui_params = dict(
        num_patches=dict(widget_type='SpinBox', value=16, min=1, max=32, step=1, label='Number of patches for annotation'),
        patch_size=dict(widget_type='SpinBox', value=256, min=224, max=1024, step=16, label='Patch size in pixels'),
        pyramid_level=dict(widget_type='ComboBox', choices=list(range(9)), value=0, label='Multiscale image level', tooltip='If image layer is a multiscale image, pick the resolution level for patches. Assumes 2x scaling between levels.'),
        pick_points_only=dict(widget_type='CheckBox', text='Pick all points', value=False, tooltip='Overwrites the number of patches and creates patches for all points.'),
        isotropic=dict(widget_type='CheckBox', text='Pick from xy, xz, or yz', value=False, tooltip='If a 3D image with isotropic voxels, pick patches from all planes.'),
        is_2d_stack=dict(widget_type='CheckBox', text='Image is 2D stack', value=False, tooltip='Check if image layer is a stack of 2D images.'),

        label_option_header=dict(widget_type='Label', label=f'<h3 text-align="center">Paired labeled data (optional)</h3>', tooltip='Useful for ready to go Ground Truth labels.'),
        label_option=dict(widget_type='CheckBox', text='Pick paired patches from images and GT labels', value=False, tooltip='Whether to pick training patches from paired image and label layers.'),
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
        pyramid_level,
        pick_points_only,
        isotropic,
        is_2d_stack,

        label_option_header,
        label_option: bool,
        label_layer: napari.layers.Labels,
    ):
        name = image_layer.name
        if points_layer is not None:
            local_points = []
            for pt in points_layer.data:
                local_points.append(tuple([int(c * image_layer.scale[i]) for i,c in enumerate(image_layer.world_to_data(pt))]))

            layer_idx = [l.name for l in viewer.layers].index(image_layer.name)
            
            assert all(t == 0 for t in image_layer.translate), \
            f"Cannot pick from points for this image! In console enter viewer.layers[{layer_idx}].translate = (0, 0, 0), then retry."

        else:
            local_points = None

        if pick_points_only:
            num_patches = len(local_points)
        
        def _show_patches(*args):
            patches, locs = args[0]

            if len(locs[0]) == 5:
                suffices = [f's{pyramid_level}-LOC-2d-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}' for l in locs]
            else:
                suffices = [f's{pyramid_level}-LOC-2d_{l[0]}-{l[1]}_{l[2]}-{l[3]}' for l in locs]

            metadata = {
                'prefix': f'{name}',
                'suffices': suffices 
            }

            viewer.add_image(patches, name=f'{name}_patches', metadata=metadata, visible=True)
            viewer.add_labels(np.zeros(patches.shape, dtype=np.int32), name=f'{name}_patches_labels', metadata=metadata, visible=True)
            viewer.dims.current_step = 0

        def _show_paired_patches(*args):
            patches, label_patches, locs = args[0]

            if len(locs[0]) == 5:
                suffices = [f's{pyramid_level}-LOC-2d-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}' for l in locs]
            else:
                suffices = [f's{pyramid_level}-LOC-2d_{l[0]}-{l[1]}_{l[2]}-{l[3]}' for l in locs]

            metadata = {
                'prefix': f'{name}',
                'suffices': suffices
            }

            viewer.add_image(patches, name=f'{name}_patches', metadata=metadata, visible=True)
            viewer.add_labels(label_patches, name=f'{name}_patches_labels', metadata=metadata, visible=True)
            viewer.dims.current_step = 0

        def _show_flipbooks(*args):
            flipbooks, locs = args[0]

            metadata = {
                'prefix': f'{name}',
                'suffices': [f's{pyramid_level}-LOC-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}_{l[5]}-{l[6]}' for l in locs]
            }

            if len(viewer.dims.order) == 3:
                viewer.dims.order = (0, 1, 2)
            elif len(viewer.dims.order) == 4:
                viewer.dims.order = (0, 1, 2, 3)

            scale = (1, 1, 1, 1)
            viewer.add_image(flipbooks, name=f'{name}_flipbooks', metadata=metadata, visible=True, scale=scale)
            viewer.add_labels(np.zeros(flipbooks.shape, dtype=np.int32), name=f'{name}_flipbooks_labels', metadata=metadata, scale=scale, visible=True)
            viewer.dims.current_step = (0, 2, 0, 0)

        def _show_paired_flipbooks(*args):
            flipbooks, label_flipbooks, locs = args[0]

            metadata = {
                'prefix': f'{name}',
                'suffices': [f's{pyramid_level}-LOC-{l[0]}_{l[1]}-{l[2]}_{l[3]}-{l[4]}_{l[5]}-{l[6]}' for l in locs]
            }

            if len(viewer.dims.order) == 3:
                viewer.dims.order = (0, 1, 2)
            elif len(viewer.dims.order) == 4:
                viewer.dims.order = (0, 1, 2, 3)

            scale = (1, 1, 1, 1)
            viewer.add_image(flipbooks, name=f'{name}_flipbooks', metadata=metadata, visible=True, scale=scale)
            viewer.add_labels(label_flipbooks, name=f'{name}_flipbooks_labels', metadata=metadata, scale=scale, visible=True)
            viewer.dims.current_step = (0, 2, 0, 0)

        if image_layer.multiscale:
            image = image_layer.data[pyramid_level]
            if label_option:
                label = label_layer.data[pyramid_level]
                assert label.shape == image.shape
            # scale the points by pyramid level
            if local_points is not None:
                local_points = [tuple([int(c / (2 ** pyramid_level)) for c in pt]) for pt in local_points]
        else:
            image = image_layer.data
            if label_option:
                label = label_layer.data
                assert label.shape == image.shape

        ndim = image.ndim 
        assert ndim in [2, 3], "Must be 2D or 3D data!"

        if ndim == 2 or is_2d_stack:
            if label_option:
                crop_worker = _pick_paired_patches(image, label, patch_size, num_patches, local_points)
                crop_worker.returned.connect(_show_paired_patches)
            else:
                crop_worker = _pick_patches(image, patch_size, num_patches, local_points)
                crop_worker.returned.connect(_show_patches)
        else:
            if label_option:
                crop_worker = _pick_paired_flipbooks(image, label, patch_size, num_patches, local_points, isotropic)
                crop_worker.returned.connect(_show_paired_flipbooks)
            else:
                crop_worker = _pick_flipbooks(image, patch_size, num_patches, local_points, isotropic)
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
        dataset_name=dict(widget_type='LineEdit', value='', label='Dataset name', tooltip='Name to use for the dataset, creates a directory in the Save Directory with this name, it appends if directory already exists.'),
    )
    @magicgui(
        call_button='Save patches',
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

        patches = image_layer.data
        patch_labels = labels_layer.data
        assert patches.shape == patch_labels.shape, "Patch and label shapes must match!"

        if image_layer.metadata:
            has_metadata = True
            prefix = image_layer.metadata['prefix']
            suffices = image_layer.metadata['suffices']
        else:
            has_metadata = False
            prefix = 'unknown'
            suffices = ['-' + _random_suffix() for _ in range(len(patches))]

        if patches.ndim == 4:
            # get middle images of flipbooks
            images = patches[:, 2]
            masks = patch_labels[:, 2]
        else:
            images = patches
            masks = patch_labels

        outdir = os.path.join(save_dir, dataset_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
            print('Created directory', outdir)
        else:
            print('Adding images to existing directory', outdir)

        os.makedirs(os.path.join(outdir, f'{prefix}/images'), exist_ok=True)
        os.makedirs(os.path.join(outdir, f'{prefix}/masks'), exist_ok=True)

        for sfx,img,msk in zip(suffices, images, masks):
            fname = f'{prefix}{sfx}.tiff'

            # if we have metadata, use it to crop image and mask
            # and remove excess padding
            if has_metadata:
                hrange, wrange = sfx.split('_')[-2:]
                hmin, hmax = hrange.split('-')
                wmin, wmax = wrange.split('-')
                h, w = int(hmax) - int(hmin), int(wmax) - int(wmin)

                img = img[:h, :w]
                msk = msk[:h, :w]

            io.imsave(os.path.join(outdir, f'{prefix}/images/{fname}'), img, check_contrast=False)
            io.imsave(os.path.join(outdir, f'{prefix}/masks/{fname}'), msk.astype(np.int32), check_contrast=False)

        print('Finished saving.')

    return widget


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def pick_patches_widget():
    return pick_patches, {'name': 'Pick finetune/training patches'}


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def store_dataset_widget():
    return store_dataset, {'name': 'Save finetune/training patches'}
