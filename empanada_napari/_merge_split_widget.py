from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
from empanada.array_utils import crop_and_binarize, take, put
from skimage.measure import regionprops
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import draw
import napari
import numpy as np
import dask.array as da


def map_points(world_points, labels_layer):
    assert all(s == 1 for s in labels_layer.scale), "Labels layer must have scale of all ones!"
    #assert all(t == 0 for t in labels_layer.translate), "Labels layer must have translation of (0, 0, 0)!"

    local_points = []
    for pt in world_points:
        local_points.append(tuple([int(c) for c in labels_layer.world_to_data(pt)]))
    
    return local_points


def delete_labels():
    @magicgui(
        call_button='Delete labels',
        layout='vertical',
        apply3d=dict(widget_type='CheckBox', text='Apply in 3D', value=False, tooltip='Check box to delete label in 3D.')
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        apply3d
    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = 'ADD'
            print('Add points!')
            return

        labels = labels_layer.data
        world_points = points_layer.data

        if apply3d and labels.ndim != 3:
            print('Apply 3D checked, but labels are not 3D. Ignoring.')

        # get points as indices in local coordinates
        local_points = map_points(world_points, labels_layer)

        if type(labels) == da.core.Array:
            label_ids = [labels[pt].compute() for pt in local_points]
        else:
            label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))

        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            for l in label_ids:
                labels[labels == l] = 0
        elif labels.ndim == 3:
            # get the current viewer axis
            axis = viewer.dims.order[0]

            # take labels along axis
            for local_pt in local_points:
                labels2d = take(labels, local_pt[axis], axis)
                for l in label_ids:
                    labels2d[labels2d == l] = 0

                put(labels, local_pt[axis], labels2d, axis)
        elif labels.ndim == 4:
            # get the current viewer axes
            assert viewer.dims.order[0] == 0, "Dims expected to be (0, 1, 2, 3) for 4D labels!"
            assert viewer.dims.order[1] == 1, "Dims expected to be (0, 1, 2, 3) for 4D labels!"

            # take labels along axis
            for local_pt in local_points:
                labels2d = labels[local_pt[0], local_pt[1]]
                for l in label_ids:
                    labels2d[labels2d == l] = 0

                labels[local_pt[0], local_pt[1]] = labels2d
            
        labels_layer.data = labels
        points_layer.data = []

        print(f'Removed labels {label_ids}')

    return widget

def merge_labels():

    def _line_to_indices(line, axis):
        if len(line[0]) == 2:
            line = line.ravel().astype('int').tolist()
            indices = np.stack(draw.line(*line), axis=1)
        elif len(line[0]) == 3:
            plane = line[0][axis]
            keep_axes = [i for i in range(3) if i != axis]
            line = line[:, keep_axes]
            line = line.ravel().astype('int').tolist()
            y, x = draw.line(*line)
            # add plane to indices
            z = np.full_like(x, plane)
            indices = [y, x]
            indices.insert(axis, z)
            indices = np.stack(indices, axis=1)
        elif len(line[0]) == 4:
            assert axis == 0
            planes = line[0][:2]
            line = line[:, [2, 3]]
            line = line.ravel().astype('int').tolist()
            y, x = draw.line(*line)
            # add plane to indices
            t = np.full_like(x, planes[0])
            z = np.full_like(x, planes[1])
            indices = np.stack([t, z, y, x], axis=1)
        else:
            raise Exception('Only lines in 2d, 3d, and 4d are supported!')

        return indices 

    @magicgui(
        call_button='Merge labels',
        layout='vertical',
        apply3d=dict(widget_type='CheckBox', text='Apply in 3D', value=False, tooltip='Check box to merge label in 3D.')
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points, 
        shapes_layer: napari.layers.Shapes, 
        apply3d
    ):
        if points_layer is None and shapes_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = 'ADD'
            print('Add points!')
            return

        axis = viewer.dims.order[0]
        labels = labels_layer.data
        world_points = []
        if points_layer is not None:
            world_points.append(points_layer.data)

        if shapes_layer is not None:
            for stype, shape in zip(shapes_layer.shape_type, shapes_layer.data):
                if stype == 'line':
                    world_points.append(_line_to_indices(shape, axis))
                elif stype == 'path':
                    n = len(shape)  # number of vertices
                    for i in range(n):
                        world_points.append(_line_to_indices(shape[i:i + 2], axis))
                        if i == n - 2:
                            break

        world_points = np.concatenate(world_points, axis=0)

        if apply3d and labels.ndim != 3:
            print('Apply 3D checked, but labels are not 3D. Ignoring.')

        # get points as indices in local coordinates
        local_points = map_points(world_points, labels_layer)

        # clip local points outside of labels shape
        for idx,pt in enumerate(local_points):
            clipped_point = ()
            for i,size in enumerate(labels.shape):
                clipped_point += (min(size - 1, max(0, pt[i])), )

            local_points[idx] = clipped_point

        if type(labels) == da.core.Array:
            label_ids = [labels[pt].compute() for pt in local_points]
        else:
            label_ids = [labels[pt].item() for pt in local_points]

        # drop any label_ids equal to 0 in case point
        # was placed on the background
        label_ids = list(filter(lambda x: x > 0, label_ids))
        label_ids = np.unique(label_ids)

        # get merged label value
        # prefer the currently selected label
        if labels_layer.selected_label in label_ids:
            new_label_id = labels_layer.selected_label
        else:
            new_label_id = min(label_ids)

        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            # replace labels with minimum of the selected labels
            for l in label_ids:
                if l != new_label_id:
                    labels[labels == l] = new_label_id
        elif labels.ndim == 3:
            # take labels along axis
            for local_pt in local_points:
                labels2d = take(labels, local_pt[axis], axis)
                # replace labels with minimum of the selected labels
                for l in label_ids:
                    if l != new_label_id:
                        labels2d[labels2d == l] = new_label_id

                put(labels, local_pt[axis], labels2d, axis)
        elif labels.ndim == 4:
            # get the current viewer axes
            assert viewer.dims.order[0] == 0, "Dims expected to be (0, 1, 2, 3) for 4D labels!"
            assert viewer.dims.order[1] == 1, "Dims expected to be (0, 1, 2, 3) for 4D labels!"

            # take labels along axis
            for local_pt in local_points:
                labels2d = labels[local_pt[0], local_pt[1]]
                for l in label_ids:
                    if l != new_label_id:
                        labels2d[labels2d == l] = new_label_id

                labels[local_pt[0], local_pt[1]] = labels2d

        labels_layer.data = labels
        if points_layer is not None:
            points_layer.data = []
        if shapes_layer is not None:
            shapes_layer.data = []

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

    def _distance_markers(binary, min_distance):
        distance = ndi.distance_transform_edt(binary)
        energy = -distance

        # handle irritating quirk of peak_local_max
        # when any dimension has length 1
        if any([s == 1 for s in distance.shape]):
            coords = peak_local_max(np.squeeze(distance), min_distance=min_distance)
            markers = np.zeros(np.squeeze(distance).shape, dtype=bool)
            markers[tuple(coords.T)] = True

            expand_axis = [s == 1 for s in distance.shape].index(True)
            markers = np.expand_dims(markers, axis=expand_axis)
        else:
            coords = peak_local_max(distance, min_distance=min_distance)
            markers = np.zeros(distance.shape, dtype=bool)
            markers[tuple(coords.T)] = True
        
        markers, _ = ndi.label(markers)
        return energy, markers

    def _point_markers(binary, local_points, shed_box):
        markers = np.zeros(binary.shape, dtype=bool)
        for local_pt in local_points:
            markers[_translate_point_in_box(local_pt, shed_box)] = True

        markers, _ = ndi.label(markers)
        energy = binary
        return energy, markers

    @magicgui(
        call_button='Split labels',
        layout='vertical',
        min_distance=dict(widget_type='Slider', label='Minimum Distance', min=1, max=100, value=10, tooltip='Min Distance between Markers'),
        points_as_markers=dict(widget_type='CheckBox', text='Use points as markers', value=False, tooltip='Whether to use the placed points as markers for watershed. If checked, Min. Distance is ignored.'),
        apply3d=dict(widget_type='CheckBox', text='Apply in 3D', value=False, tooltip='Check box to split label in 3D.')
    )

    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: napari.layers.Labels,
        points_layer: napari.layers.Points,
        min_distance: int,
        points_as_markers: bool,
        apply3d
    ):
        if points_layer is None:
            points_layer = viewer.add_points([])
            points_layer.mode = 'ADD'
            return

        labels = labels_layer.data
        world_points = points_layer.data

        if apply3d and labels.ndim != 3:
            print('Apply 3D checked, but labels are not 3D. Ignoring.')

        # get points as indices in local coordinates
        local_points = map_points(world_points, labels_layer)

        if type(labels) == da.core.Array:
            raise Exception(f'Split operation is not supported on Dask Array labels!')

        label_ids = np.array([labels[pt].item() for pt in local_points])
        local_points = np.stack(local_points, axis=0)

        # drop any label_ids equal to 0; in case point
        # was placed on the background
        background_pts = label_ids == 0
        local_points = local_points[~background_pts]
        label_ids = label_ids[~background_pts]

        if len(label_ids) == 0:
            print('No labels selected!')
            return

        label_id = label_ids[0]

        if len(np.unique(label_ids)) > 1:
            print('Split operation only supports 1 label at a time.')
            # drop points corresponding to extraneous labels
            local_points = local_points[label_ids == label_id]

        if labels.ndim == 2 or (labels.ndim == 3 and apply3d):
            shed_box = [rp.bbox for rp in regionprops(labels) if rp.label == label_id][0]
            binary = crop_and_binarize(labels, shed_box, label_id)

            if points_as_markers:
                energy, markers = _point_markers(binary, local_points, shed_box)
            else:
                energy, markers = _distance_markers(binary, min_distance)

            marker_ids = np.unique(markers)[1:]

            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)

                max_label = labels.max()
                labels[slices][binary] = new_labels[binary] + max_label
                print(f'Split label {label_id} to {marker_ids + max_label}')
            else:
                print('Nothing to split.')

        elif labels.ndim == 3:
            # get the current viewer axis
            axis = viewer.dims.order[0]
            plane = local_points[0][axis]
            labels2d = take(labels, plane, axis)
            assert all(local_pt[axis] == plane for local_pt in local_points)

            shed_box = [rp.bbox for rp in regionprops(labels2d) if rp.label == label_id][0]
            binary = crop_and_binarize(labels2d, shed_box, label_id)

            if points_as_markers:
                local_points2d = []
                for lp in local_points:
                    local_points2d.append([p for i,p in enumerate(lp) if i != axis])
                energy, markers = _point_markers(binary, local_points2d, shed_box)
            else:
                energy, markers = _distance_markers(binary, min_distance)

            marker_ids = np.unique(markers)[1:]

            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)

                max_label = labels2d.max()
                labels2d[slices][binary] = new_labels[binary] + max_label
                print(f'Split label {label_id} to {marker_ids + max_label}')
            else:
                print('Nothing to split.')

            put(labels, local_points[0][axis], labels2d, axis)

        elif labels.ndim == 4:
            # get the current viewer axes
            assert viewer.dims.order[0] == 0, "Dims expected to be (0, 1, 2, 3) for 4D labels!"
            assert viewer.dims.order[1] == 1, "Dims expected to be (0, 1, 2, 3) for 4D labels!"
            plane1 = local_points[0][0]
            plane2 = local_points[0][1]
            assert all(local_pt[0] == plane1 for local_pt in local_points)
            assert all(local_pt[1] == plane2 for local_pt in local_points)
        
            labels2d = labels[plane1, plane2]

            shed_box = [rp.bbox for rp in regionprops(labels2d) if rp.label == label_id][0]
            binary = crop_and_binarize(labels2d, shed_box, label_id)

            if points_as_markers:
                energy, markers = _point_markers(binary, local_points, shed_box)
            else:
                energy, markers = _distance_markers(binary, min_distance)

            marker_ids = np.unique(markers)[1:]

            if len(marker_ids) > 1:
                new_labels = watershed(energy, markers, mask=binary)
                slices = _box_to_slice(shed_box)

                max_label = labels2d.max()
                labels2d[slices][binary] = new_labels[binary] + max_label
                print(f'Split label {label_id} to {marker_ids + max_label}')
            else:
                print('Nothing to split.')
                
            labels[local_points[0][0], local_points[0][1]] = labels2d

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
