from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory, magicgui
from empanada.array_utils import merge_boxes, crop_and_binarize
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import napari
import numpy as np
import dask.array as da

def delete_labels():
    @magicgui(
        call_button='Delete labels',
        layout='vertical',
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,

    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = 'ADD'
            return

        labels = labels_layer.data
        world_points = points_layer.data

        # get points as indices in local coordinates
        local_points = []
        for pt in world_points:
            local_points.append(tuple([int(c) for c in labels_layer.world_to_data(pt)]))

        if type(labels) == da.core.Array:
            label_ids = [labels[pt].compute() for pt in local_points]
        else:
            label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))

        # replace labels with minimum of the selected labels
        for l in label_ids:
            labels[labels == l] = 0

        labels_layer.data = labels
        points_layer.data = []

        print(f'Removed labels {label_ids}')

    return widget

def merge_labels():
    @magicgui(
        call_button='Merge labels',
        layout='vertical',
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,

    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = 'ADD'
            return

        labels = labels_layer.data
        world_points = points_layer.data

        # get points as indices in local coordinates
        local_points = []
        for pt in world_points:
            local_points.append(tuple([int(c) for c in labels_layer.world_to_data(pt)]))

        if type(labels) == da.core.Array:
            label_ids = [labels[pt].compute() for pt in local_points]
        else:
            label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))

        # get merged label value
        # prefer the currently selected label
        if labels_layer.selected_label in label_ids:
            new_label_id = labels_layer.selected_label
        else:
            new_label_id = min(label_ids)

        # replace labels with minimum of the selected labels
        for l in label_ids:
            if l != new_label_id:
                labels[labels == l] = new_label_id

        labels_layer.data = labels
        points_layer.data = []

        print(f'Merged labels {label_ids} to {new_label_id}')

    return widget

def split_labels():

    def _box_to_slice(shed_box):
        n = len(shed_box)
        n_dim = n//2

        slices = []
        for i in range(n_dim):
            slices.append(slice(shed_box[i], shed_box[i+n_dim]))

        return tuple(slices)

    def _translate_point_in_box(point, shed_box):
        n_dim = len(shed_box) // 2
        return tuple([int(point[i] - shed_box[i]) for i in range(n_dim)])

    @magicgui(
        call_button='Split labels',
        layout='vertical',
        min_distance=dict(widget_type='Slider', label='Minimum Distance', min=1, max=100, value=10, tooltip='Min Distance between Markers'),
        points_as_markers=dict(widget_type='CheckBox', text='Use points as markers', value=False, tooltip='Whether to use the placed points as markers for watershed. If checked, Min. Distance is ignored.'),
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        min_distance: int,
        points_as_markers: bool
    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = 'ADD'
            return

        labels = labels_layer.data
        world_points = points_layer.data

        # get points as indices in local coordinates
        local_points = []
        for pt in world_points:
            local_points.append(tuple([int(c) for c in labels_layer.world_to_data(pt)]))

        if type(labels) == da.core.Array:
            raise Exception(f'Split operation is not supported on Dask Array labels!')

        label_ids = np.array([labels[pt].item() for pt in local_points])
        local_points = np.stack(local_points, axis=0)

        # drop any label_ids equal to 0; in case point
        # was placed on the background
        background_pts = label_ids == 0
        local_points = local_points[~background_pts]
        label_ids = label_ids[~background_pts]

        label_id = label_ids[0]

        if len(np.unique(label_ids)) > 1:
            print('Split operation only supports 1 label at a time.')

            # drop points corresponding to extraneous labels
            local_points = local_points[label_ids == label_id]

        shed_box = [rp.bbox for rp in regionprops(labels) if rp.label == label_id][0]
        binary = crop_and_binarize(labels, shed_box, label_id)

        if points_as_markers:
            energy = binary
            markers = np.zeros(binary.shape, dtype=bool)
            for local_pt in local_points:
                markers[_translate_point_in_box(local_pt, shed_box)] = True

        else:
            distance = ndi.distance_transform_edt(binary)
            energy = -distance

            # handle irritating quirk of peak_local_max
            # when any dimension has length 1
            if any([s == 1 for s in distance.shape]):
                coords = peak_local_max(np.squeeze(distance), min_distance=min_distance)
                markers = np.zeros(np.squeeze(distance).shape, dtype=bool)
                mask[tuple(coords.T)] = True

                expand_axis = [s == 1 for s in distance.shape].index(True)
                mask = np.expand_dims(mask, axis=expand_axis)
            else:
                coords = peak_local_max(distance, min_distance=min_distance)
                markers = np.zeros(distance.shape, dtype=bool)
                markers[tuple(coords.T)] = True

        # label markers and run watershed
        markers, _ = ndi.label(markers)
        marker_ids = np.unique(markers)[1:]

        if len(marker_ids) > 1:
            new_labels = watershed(energy, markers, mask=binary)
            slices = _box_to_slice(shed_box)

            max_label = labels.max()
            labels[slices][binary] = new_labels[binary] + max_label
            print(f'Split label {label_id} to {marker_ids + max_label}')
        else:
            print('Nothing to split.')

        labels_layer.data = labels
        points_layer.data = []

    return widget

def jump_to_label():
    @magicgui(
        call_button='Jump to label',
        layout='vertical',
        label_id=dict(widget_type='LineEdit', label='Label ID', value='1', tooltip='Label to jump to'),
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        label_id
    ):
        assert labels_layer.data.ndim == 3
        label_id = int(label_id)

        bbox = None
        for rp in regionprops(labels_layer.data):
            if rp.label == label_id:
                bbox = rp.bbox
                break

        if bbox is None:
            raise Exception(f'No label {label_id} in {labels_layer.name}')

        # split bounding box by axis
        ndim = len(bbox) // 2
        bbox = tuple([
            *labels_layer._data_to_world(bbox[:ndim]),
            *labels_layer._data_to_world(bbox[ndim:]),
        ])

        bbox = [(bbox[i], bbox[i+ndim]) for i in range(ndim)]

        # get the current axis
        axis = viewer.dims.order[0]

        new_step = list(viewer.dims.current_step)
        new_step[axis] = bbox[axis][0]

        # set the current step to first slice of bbox
        viewer.dims.current_step = tuple(new_step)
        labels_layer.selected_label = label_id

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def delete_labels_widget():
    return delete_labels, {'name': 'Delete Labels'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def merge_labels_widget():
    return merge_labels, {'name': 'Merge Labels'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def split_labels_widget():
    return split_labels, {'name': 'Split Labels'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def jump_to_label_widget():
    return jump_to_label, {'name': 'Jump to Label'}
