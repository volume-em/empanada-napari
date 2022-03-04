from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory, magicgui
from empanada.array_utils import merge_boxes, crop_and_binarize
from skimage.measure import regionprops
#from napari_tools_menu import register_function
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

        # replace labels with minimum of the selected labels
        new_label_id = min(label_ids)
        for l in label_ids:
            if l != new_label_id:
                labels[labels == l] = new_label_id

        labels_layer.data = labels
        points_layer.data = []

        print(f'Merged labels {label_ids} to {new_label_id}')
    return widget

def split_function():
    @magicgui(
        call_button='Split labels',
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

        # make a binary image inside a minimal bounding box
        bboxes = [rp.bbox for rp in regionprops(labels) if rp.label in label_ids]

        shed_box = bboxes[0]
        for box in bboxes[1:]:
            shed_box = merge_boxes(shed_box, box)

        binary = crop_and_binarize(labels, shed_box, label_ids[0])
        for label_id in label_ids[1:]:
            binary += crop_and_binarize(labels, shed_box, label_id)

        print(labels.shape, binary.shape)

        from scipy import ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        #distance = ndi.distance_transform_edt(binary)
        #coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary)

        def translate_point_in_box(point, shed_box):
            n = len(shed_box)
            n_dim = n//2

            return tuple([int(point[i] - shed_box[i]) for i in range(n_dim)])

        def box_to_slice(shed_box):
            n = len(shed_box)
            n_dim = n//2

            slices = []
            for i in range(n_dim):
                slices.append(slice(shed_box[i], shed_box[i+n_dim]))

            return tuple(slices)

        mask = np.zeros(binary.shape, dtype=bool)
        for i in local_points:
            mask[translate_point_in_box(i, shed_box)] = True

        markers, _ = ndi.label(mask)
        n_markers = np.unique(markers)[1:]

        if type(binary) == da.core.Array:
            binary = binary.compute()

        new_labels = watershed(binary, markers, mask=binary)

        slices = box_to_slice(shed_box)


        if type(labels) == da.core.Array:
            #new_labels[binary] += labels.max()
            max_label = 0
            new_labels[~binary] = labels[slices].compute()[~binary]
            labels[slices] = new_labels
        else:
            max_label = labels.max()
            labels[slices][binary] = new_labels[binary] + max_label

        labels_layer.data = labels
        points_layer.data = []

        print(f'Split labels {label_ids} to {n_markers + max_label}')
    return widget

def split_widget_distance():
    @magicgui(
        call_button='Split labels by distance watershed',
        layout='vertical',
        min_distance=dict(widget_type='Slider', label='Minimum Distance', min=1, max=100, value=10, tooltip='Min Distance between Markers'),
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        min_distance: int
    ):
        print(min_distance)
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

        # make a binary image inside a minimal bounding box
        bboxes = [rp.bbox for rp in regionprops(labels) if rp.label in label_ids]

        shed_box = bboxes[0]
        for box in bboxes[1:]:
            shed_box = merge_boxes(shed_box, box)

        binary = crop_and_binarize(labels, shed_box, label_ids[0])
        for label_id in label_ids[1:]:
            binary += crop_and_binarize(labels, shed_box, label_id)

        print(labels.shape, binary.shape)

        from scipy import ndimage as ndi
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        def box_to_slice(shed_box):
            n = len(shed_box)
            n_dim = n//2

            slices = []
            for i in range(n_dim):
                slices.append(slice(shed_box[i], shed_box[i+n_dim]))

            return tuple(slices)


        distance = ndi.distance_transform_edt(binary)

        if np.squeeze(distance).ndim == distance.ndim - 1:
            coords = peak_local_max(np.squeeze(distance), min_distance=min_distance)
            mask = np.zeros(np.squeeze(distance).shape, dtype=bool)
            mask[tuple(coords.T)] = True
            mask = mask[None]
        else:
            coords = peak_local_max(distance, min_distance=min_distance)
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(coords.T)] = True

        markers, _ = ndi.label(mask)

        if type(binary) == da.core.Array:
            binary = binary.compute()

        new_labels = watershed(-distance, markers, mask=binary)
        slices = box_to_slice(shed_box)

        if type(labels) == da.core.Array:
            #new_labels[binary] += labels.max()
            new_labels[~binary] = labels[slices].compute()[~binary]
            labels[slices] = new_labels
        else:
            labels[slices][binary] = new_labels[binary] + labels.max()

        labels_layer.data = labels
        points_layer.data = []

        print('Done')

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

        print('Bounding box', bbox)

        # split bounding box by axis
        ndim = len(bbox) // 2
        bbox = tuple([
            *labels_layer._data_to_world(bbox[:ndim]),
            *labels_layer._data_to_world(bbox[ndim:]),
        ])

        print('Bounding box transformed', bbox)

        bbox = [(bbox[i], bbox[i+ndim]) for i in range(ndim)]

        # get the current axis
        axis = viewer.dims.order[0]

        # set the current step to first slice of bbox
        viewer.dims.current_step = (bbox[axis][0], 0, 0)

        # set viewer range to y and x range of bbox
        bbox2d = [dr for i,dr in enumerate(bbox) if i != axis]
        viewer.window.qt_viewer.view.camera.set_range(y=bbox2d[0], x=bbox2d[1])

    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def split_by_dist():
    return split_widget_distance, {'name': 'Split by Distance'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def jump_to_label_widget():
    return jump_to_label, {'name': 'Jump to Label'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def merge_labels_widget():
    return merge_labels, {'name': 'Merge Labels'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def split_labels_widget():
    return split_function, {'name': 'Split Labels'}

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def delete_labels_widget():
    return delete_labels, {'name': 'Delete Labels'}
