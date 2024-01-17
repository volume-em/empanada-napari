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

    # only keep area values less than 1000
    rp = rp[rp['area'] <= minimum_area_allowed]

    # remove label from image
    for label in rp['label']:
        img = remove_label_from_image(img, label)

    return img


def filter_out_small_label_areas_by_idx(labels, minimum_area_allowed, plane):
    img = labels[plane]
    img = filter_out_small_label_areas(img, minimum_area_allowed)
    labels[plane] = img
    return labels


def filter_small_labels():

    remove_options = {
        'Small labels': 'Small labels',
        'Boundary labels': 'Boundary labels'
    }

    apply_options = {
        'Current image': 'Current image',
        '2D tile stack': '2D tile stack',
        '3D image or z-stack': '3D image or z-stack'
    }

    gui_params = {
        "call_button": 'Remove labels',
        "layout": 'vertical',
        "apply_to": dict(widget_type='RadioButtons', label='Apply to:', choices=list(apply_options.keys()), value=list(apply_options.keys())[0], tooltip='Apply to a single image,a 2D tile stack or a 3D image/z-stack.'),
        "min_area": dict(widget_type='SpinBox', label='Minimum pixel/voxel area cutoff:', value=100, min=1, max=10000000, tooltip='Minimum area of labels [px^2] to keep.'),
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

        # print(labels.shape)
        if apply_to == 'Current image':
            plane = viewer.dims.current_step[0]

            if remove_opt == 'Small labels':
                labels = filter_out_small_label_areas_by_idx(labels, min_area, plane)
            elif remove_opt == 'Boundary labels':
                labels[plane] = clear_border(labels[plane])

        elif labels.ndim == 3 and apply_to == '2D tile stack':
            for label in tqdm(range(labels.shape[0])):
                if remove_opt == 'Small labels':
                    labels[label] = filter_out_small_label_areas(labels[label], min_area)
                elif remove_opt == 'Boundary labels':
                    labels[label] = clear_border(labels[label])

        else:
            if remove_opt == 'Small labels':
                labels = filter_out_small_label_areas(labels, min_area)
            elif remove_opt == 'Boundary labels':
                labels = clear_border(labels)

        labels = np.squeeze(labels)
        viewer.add_labels(labels, name=labels_layer.name + '_filtered', visible=True)

    return widget


