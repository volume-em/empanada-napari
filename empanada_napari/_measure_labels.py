import os
import numpy as np
import pandas as pd
import dask.array as da

import napari
from napari.layers import Labels
from magicgui import magicgui
from skimage.measure import regionprops_table


def _compute_metrics(label_slice: np.ndarray) -> pd.DataFrame:
    """Compute morphometric features for each label in a 2D label image."""
    props = regionprops_table(
        label_slice.astype(np.int32),
        properties=(
            'label',
            'area',
            'perimeter',
            'major_axis_length',
            'minor_axis_length',
            'eccentricity',
            'feret_diameter_max',
            'equivalent_diameter_area',
        )
    )
    df = pd.DataFrame(props)
    df = df[df['label'] != 0]

    # circularity = 4π × area / perimeter²; guard against zero-perimeter edge labels
    df['circularity'] = np.where(
        df['perimeter'] > 0,
        (4 * np.pi * df['area']) / (df['perimeter'] ** 2),
        np.nan
    )
    # aspect ratio = major / minor axis length; guard against zero minor axis
    df['aspect_ratio'] = np.where(
        df['minor_axis_length'] > 0,
        df['major_axis_length'] / df['minor_axis_length'],
        np.nan
    )
    return df


def measure_labels_widget():
    apply_to_opts = {
        'Current slice': 'Current slice',
        'All slices (z-stack)': 'All slices (z-stack)',
    }

    @magicgui(
        call_button='Measure Labels',
        layout='vertical',
        apply_to=dict(
            widget_type='RadioButtons',
            choices=list(apply_to_opts.keys()),
            value='Current slice',
            label='Apply to:',
            tooltip='Measure the current 2D slice or every slice in the z-stack.',
        ),
        export_csv=dict(
            widget_type='CheckBox',
            value=False,
            label='Export measurements (.csv)',
            tooltip='Save per-label measurements as a CSV file.',
        ),
        save_dir=dict(
            widget_type='FileEdit',
            value='',
            label='Save directory',
            mode='d',
            tooltip='Directory in which to save the CSV file.',
        ),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        labels_layer: Labels,
        apply_to: str,
        export_csv: bool,
        save_dir: str,
    ):
        labels = labels_layer.data
        if isinstance(labels, da.Array):
            labels = labels.compute()

        results = []

        if apply_to == 'Current slice' or labels.ndim == 2:
            if labels.ndim > 2:
                axis = viewer.dims.order[0]
                plane = int(viewer.dims.current_step[axis])
                slices = [slice(None)] * labels.ndim
                slices[axis] = plane
                label_slice = labels[tuple(slices)]
            else:
                label_slice = labels
                plane = 0

            df = _compute_metrics(np.asarray(label_slice))
            df.insert(0, 'slice', plane)
            results.append(df)

        else:
            for plane in range(labels.shape[0]):
                label_slice = labels[plane]
                df = _compute_metrics(np.asarray(label_slice))
                df.insert(0, 'slice', plane)
                results.append(df)

        combined = pd.concat(results, ignore_index=True)

        cols = ['slice', 'label', 'area', 'perimeter', 'circularity',
                'aspect_ratio', 'eccentricity', 'feret_diameter_max',
                'equivalent_diameter_area', 'major_axis_length', 'minor_axis_length']
        combined = combined[[c for c in cols if c in combined.columns]]

        print(combined.to_string(index=False))
        print(f'\nTotal labels measured: {len(combined)}')

        if export_csv:
            assert save_dir, "Please select a save directory!"
            os.makedirs(save_dir, exist_ok=True)
            filename = f'{labels_layer.name}_measurements.csv'
            filepath = os.path.join(save_dir, filename)
            combined.to_csv(filepath, index=False)
            print(f'Saved measurements to {filepath}')

    return widget
