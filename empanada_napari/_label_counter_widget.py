from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui
import napari
from napari import Viewer
from napari.layers import Labels
import pandas as pd
import os
from openpyxl import Workbook, load_workbook
import numpy as np
import dask.array as da
import itertools
from scipy import ndimage as ndi


def create_xlsx_file(label_queue, save_dir, labels_layer):
    filename = f'{labels_layer.name}_label_counts.xlsx'
    file_path = os.path.join(save_dir, filename)
    workbook = Workbook()

    for class_id, label_ids in label_queue.items():
        sheet_name = f'Class {class_id}'
        sheet = workbook.create_sheet(title=sheet_name)
        sheet['A1'] = 'Label ID'

        for row_num, label_id in enumerate(label_ids, start=1):
            sheet.cell(row=row_num + 1, column=1, value=label_id)

    default_sheet = workbook['Sheet']
    workbook.remove(default_sheet)
    workbook.save(file_path)


def create_xlsx_from_label_queue_list(label_queue_list, save_dir, labels_layer, labels):
    filename = f'{labels_layer.name}_label_counts.xlsx'
    file_path = os.path.join(save_dir, filename)
    workbook = Workbook()

    for slice_num in range(labels.shape[0]):
        label_queue = label_queue_list[slice_num]
        for class_id, label_ids in label_queue.items():
            sheet_name = f'Image {slice_num}'

            if sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                # Clear the existing contents of the sheet
                sheet.delete_rows(2, sheet.max_row)
            else:
                # Create a new sheet
                sheet = workbook.create_sheet(title=sheet_name)
            sheet['A1'] = 'Label ID'

            for row_num, label_id in enumerate(label_ids, start=1):
                sheet.cell(row=row_num + 1, column=1, value=label_id)

    default_sheet = workbook['Sheet']
    workbook.remove(default_sheet)
    workbook.save(file_path)


def count_labels(label_values, label_divisor):
    class_ids = np.unique(np.floor_divide(label_values, label_divisor)).tolist()
    label_queue = {}
    for ci in class_ids:
        min_id = ci * label_divisor + 1
        max_id = (ci + 1) * label_divisor
        label_ids = label_values[np.logical_and(label_values >= min_id, label_values < max_id)]
        label_queue[ci] = label_ids.tolist()

    return label_queue, class_ids


def label_counter_widget():
    label_params = {
        'Current slice': 'Current slice',
        '2D stack': '2D stack',
        '3D volume': '3D volume',
    }

    @magicgui(
        call_button='Count Labels',
        layout='vertical',
        label_type=dict(widget_type='RadioButtons', choices=list(label_params.keys()),
                        value=list(label_params.keys())[0], label='Apply to:',
                        tooltip='Calculate number of instance labels'),
        label_divisor=dict(widget_type='LineEdit', label='Label Divisor', value='10000',
                           tooltip='Label divisor that separates objects of different classes.'),

        save_op_head=dict(widget_type='Label', label=f'<h3 text-align="center">Export Excel File (optional)</h3>',
                          tooltip='Export csv file with labels counted by class.'),
        export_xlsx=dict(widget_type='CheckBox', value=False, label='Export .xlsx file',
                         tooltip='Export list of label IDs as an excel file.'),
        save_dir=dict(widget_type='FileEdit', value='', label='Save directory', mode='d',
                      tooltip='Directory in which to save label counter excel file.'),
    )
    def widget(
            viewer: napari.viewer.Viewer,
            labels_layer: Labels,
            label_type: str,
            label_divisor: str,

            save_op_head,
            export_xlsx: bool,
            save_dir: str

    ):
        label_divisor = int(label_divisor)
        assert label_divisor > 0, "Label divisor must be a positive integer!"

        labels = labels_layer.data

        if labels.ndim > 2 and label_type == 'Current slice':
            # assert viewer.dims.order[0] == 0, "Must be viewing axis 0 (xy plane)!"
            plane = viewer.dims.current_step[0]
        else:
            plane = 'null'

        if plane != 'null':
            labels = labels[plane]

        class_ids = []

        if isinstance(labels, da.Array):
            label_values = []
            for inds in itertools.product(*map(range, labels.blocks.shape)):
                chunk = labels.blocks[inds].compute()
                label_values.append(np.unique(chunk)[1:])

            label_values = np.concatenate(label_values)
        else:
            label_values = np.unique(labels)[1:]

        if label_type == 'Current slice':
            label_queue, class_ids = count_labels(label_values, label_divisor)
            if label_queue and class_ids:
                for class_id in class_ids:
                    if class_id in label_queue:
                        label_list = np.unique((label_queue[class_id]))
                        print(f'Labels in class {class_id}: {label_list}')
                        print(f'Total number of labels in class {class_id}:', len(label_list))
                    else:
                        print(f'No labels in class {class_id}')
                    if export_xlsx:
                        os.makedirs(save_dir, exist_ok=True)
                        create_xlsx_file(label_queue, save_dir, labels_layer)
                        print(f'Saved Excel file to {save_dir}')
            else:
                print('No labels found in current slice!')

        elif label_type == '2D stack':
            class_ids_list = []
            label_queues_list = []
            labels_list = []

            for slice_num in range(labels.shape[0]):
                label_slice = labels[slice_num]
                label_queue, class_ids = count_labels(np.unique(label_slice)[1:], label_divisor)
                label_queues_list.append(label_queue)
                class_ids_list.append(class_ids)
                unique_labels = set()

                if label_queue and class_ids:
                    for class_id in class_ids:
                        if class_id in label_queue:
                            label_list = np.unique((label_queue[class_id]))
                            print(f'Total number of labels in class {class_id} in image {slice_num}:', len(label_list))
                        else:
                            print(f'No labels in class {class_id}')

                labels_list.append(list(unique_labels))

            if export_xlsx:
                os.makedirs(save_dir, exist_ok=True)
                create_xlsx_from_label_queue_list(label_queues_list, save_dir, labels_layer, labels)
                print(f'Saved Excel file to {save_dir}')

        elif label_type == '3D volume':
            class_ids_list = []
            label_queues_list = []
            labels_list = []

            label_queue, class_ids = count_labels(np.unique(labels)[1:], label_divisor)
            label_queues_list.append(label_queue)
            class_ids_list.append(class_ids)
            unique_labels = set()

            if label_queue and class_ids:
                for class_id in class_ids:
                    if class_id in label_queue:
                        label_list = np.unique((label_queue[class_id]))
                        print(f'Total number of labels in class {class_id} in volume:', len(label_list))

                    else:
                        print(f'No labels in class {class_id}')

            labels_list.append(list(unique_labels))

            if export_xlsx:
                os.makedirs(save_dir, exist_ok=True)
                create_xlsx_file(label_queue, save_dir, labels_layer)
                print(f'Saved Excel file to {save_dir}')

    return widget


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def label_count_widget():
    return label_counter_widget, {'name': 'Count Labels'}
