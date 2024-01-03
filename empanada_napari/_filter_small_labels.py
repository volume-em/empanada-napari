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


def filter_out_small_label_areas(img, minimum_area_allowed, remove_boundary_labels=False):
    if remove_boundary_labels:
        img = clear_border(img)
    rp = regionprops_table(img, properties=("label", "area"))
    rp = pd.DataFrame(rp)
    rp = rp.sort_values(by='area')

    # only keep area values less than 1000
    rp = rp[rp['area'] <= minimum_area_allowed]

    # remove label from image
    for label in rp['label']:
        img = remove_label_from_image(img, label)

    return img


def filter_small_labels():
    gui_params = {
        "call_button": 'Remove small labels',
        "layout": 'vertical',
        "apply3d": dict(widget_type='CheckBox', label='Apply in 3D', value=False,
                        tooltip='Check box to filter small labels in 3D.'),
        "min_area": dict(widget_type='SpinBox', label='Minimum pixel area', value=1, min=1, max=10000000,
                         tooltip='Minimum area of labels [px^2] to keep.'),
        "remove_boundary_labels": dict(widget_type='CheckBox', label='Remove boundary labels', value=False,
                                       tooltip='Check box to remove boundary labels.'),
    }

    @magicgui(
        **gui_params
    )
    def widget(
            viewer: napari.viewer.Viewer,
            labels_layer: napari.layers.Labels,
            apply3d,
            min_area,
            remove_boundary_labels
    ):
        if labels_layer is None:
            return

        labels = np.asarray(labels_layer.data).copy()
        if apply3d and labels.ndim != 3:
            print('Apply 3D checked, but labels are not 3D. Ignoring.')
            return

        # print(labels.shape)

        if labels.ndim == 3 and not apply3d:
            for label in tqdm(range(labels.shape[0])):
                labels[label] = filter_out_small_label_areas(labels[label], min_area, remove_boundary_labels)
        else:
            labels = filter_out_small_label_areas(labels, min_area, remove_boundary_labels)

        labels = np.squeeze(labels)
        viewer.add_labels(labels, name=labels_layer.name + '_filtered', visible=True)

    return widget


