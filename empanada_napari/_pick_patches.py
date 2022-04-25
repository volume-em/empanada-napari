import napari
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
    def _random_crops(volume, patch_size, num_patches, isotropic):
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
        num_patches=dict(widget_type='SpinBox', value=16, min=1, max=512, step=1, label='Number of patches for annotation'),
        patch_size=dict(widget_type='SpinBox', value=224, min=128, max=512, step=16, label='Number of patches for annotation'),
        filter_pct=dict(widget_type='FloatSpinBox', value=0.9, min=0.0, max=1., step=0.1, label='Percent of uninformative patches to filter out'),
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
        num_patches,
        patch_size,
        filter_pct,
        isotropic
    ):

        volume = image_layer.data
        assert volume.ndim == 3, "Must be 3D data!"

        def _show_flipbooks(*args):
            flipbooks, locs = args[0]
            viewer.add_image(flipbooks, name=f'flipbooks', visible=True)

        crop_worker = _random_crops(volume, patch_size, num_patches, isotropic)
        crop_worker.returned.connect(_show_flipbooks)
        crop_worker.start()

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def pick_patches_widget():
    return pick_patches, {'name': 'Pick training patches'}
