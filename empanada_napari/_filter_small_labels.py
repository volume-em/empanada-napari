import napari
import numpy as np
from magicgui import magicgui
from skimage.measure import regionprops_table
import pandas as pd
from tqdm import tqdm
from skimage.segmentation import clear_border


def remove_label_from_image(image_array, label):
    image_array[image_array == label] = 0
    return image_array


def filter_out_small_label_areas(img, minimum_area_allowed):
    rp = regionprops_table(img, properties=("label", "area"))
    rp = pd.DataFrame(rp)
    rp = rp.sort_values(by='area')

    smallest_label = rp['label'].iloc[0]
    smallest_area = int(rp['area'].iloc[0])
    # only keep area values less than 1000
    rp = rp[rp['area'] <= minimum_area_allowed]

    # remove label from image
    labels_removed = []
    for label in rp['label']:
        img = remove_label_from_image(img, label)
        labels_removed.append(label)

    if len(labels_removed) == 0:
        print("No labels were removed.")
        print(
            f"The label ID corresponding to the smallest area is {smallest_label} with an area of {smallest_area} pixels/voxels.")
    else:
        num_removed = len(labels_removed)
        print(f"The following label IDs were removed: {labels_removed}")
        print(f"Total number of label IDs removed: {num_removed} ")

    return img, len(labels_removed)


def remove_boundary_labels(labels):
    labels_removed = []
    labels_kept = clear_border(labels)

    for label in np.unique(labels):
        if label not in np.unique(labels_kept):
            labels_removed.append(label)

    if len(labels_removed) == 0:
        print("No label IDs were removed.")
    else:
        num_removed = len(labels_removed)
        print(f"The following label Ids were removed: {labels_removed}")
        print(f"Total number of label IDs removed: {num_removed} ")

    labels = labels_kept

    return labels, len(labels_removed)


def filter_small_labels():

    remove_options = {
        'Small labels': 'Small labels',
        'Boundary labels': 'Boundary labels'
    }

    apply_options = {
        'Current image': 'Current image',
        '2D patches': '2D patches',
        '3D image or z-stack': '3D image or z-stack'
    }

    gui_params = {
        "call_button": 'Remove labels',
        "layout": 'vertical',
        "apply_to": dict(widget_type='RadioButtons', label='Apply to:', choices=list(apply_options.keys()), value=list(apply_options.keys())[0], tooltip='Apply to a single image,a 2D tile stack or a 3D image/z-stack.'),
        "min_area": dict(widget_type='SpinBox', label='Minimum pixel/voxel area:', value=100, min=1, max=10000000, tooltip='Minimum area of labels [px^2] to keep.'),
        "remove_opt": dict(widget_type='RadioButtons', label='Remove:', choices=list(remove_options.keys()), value=list(remove_options.keys())[0], tooltip='Either remove boundary labels or small pixel labels.'),
    }

    @magicgui(
        **gui_params
    )
    def widget(
            viewer: napari.viewer.Viewer,
            labels_layer: napari.layers.Labels,
            remove_opt,
            min_area,
            apply_to,
    ):
        if labels_layer is None:
            return

        labels = np.asarray(labels_layer.data).copy()
        if apply_to == '3D image or z-stack' and labels.ndim != 3:
            print('Apply 3D checked, but labels are not 3D. Ignoring.')
            return
        labels_removed = []
        # print(labels.shape)
        if apply_to == 'Current image':
            plane = viewer.dims.current_step[0]

            if remove_opt == 'Small labels':
                labels[plane], labels_removed = filter_out_small_label_areas(labels[plane], min_area)
            elif remove_opt == 'Boundary labels':
                labels[plane], labels_removed = remove_boundary_labels(labels[plane])

        elif labels.ndim == 3 and apply_to == '2D patches':
            for label in tqdm(range(labels.shape[0])):
                if remove_opt == 'Small labels':
                    labels[label], labels_removed = filter_out_small_label_areas(labels[label], min_area)
                elif remove_opt == 'Boundary labels':
                    labels[label], labels_removed = remove_boundary_labels(labels[label])

        else:
            if remove_opt == 'Small labels':
                labels, labels_removed = filter_out_small_label_areas(labels, min_area)
            elif remove_opt == 'Boundary labels':
                labels, labels_removed = remove_boundary_labels(labels)
        if labels_removed == 0:
            return
        else:
            labels = np.squeeze(labels)
            viewer.add_labels(labels, name=f'Labels_removed_' + labels_layer.name, visible=True)

    return widget


