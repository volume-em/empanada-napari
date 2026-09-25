from napari_plugin_engine import napari_hook_implementation


def _path_from_dask_task(task):
    r"""Return the file path argument from a dask imread task.

    Older dask stored tasks as ``(func, path, ...)`` tuples. Newer dask
    (e.g. 2026.x) stores ``Task`` objects where ``task.args[0]`` is the path.
    See https://github.com/volume-em/empanada-napari/issues/77.
    """
    if isinstance(task, (tuple, list)):
        if len(task) >= 2 and isinstance(task[1], str):
            return task[1]
        return None

    args = getattr(task, 'args', None) or ()
    for arg in args:
        if isinstance(arg, str):
            return arg
    return None


def _get_impaths_from_dask(dask_array):
    r"""Extract source image paths from a napari Open Folder dask stack.

    Paths are ordered to match stack axis 0 when a ``stack-`` layer is
    present. Falls back to imread-layer order if stack wiring cannot be
    resolved.
    """
    graph = dask_array.dask
    deps = getattr(graph, 'dependencies', {}) or {}

    imread_paths = {}
    for name in graph.layers:
        if 'imread' not in str(name):
            continue
        path = _path_from_dask_task(graph[name])
        if path is not None:
            imread_paths[str(name)] = path

    if not imread_paths:
        return []

    def _imread_for_from_value(from_value):
        for dep in deps.get(from_value, ()) or ():
            if 'imread' in str(dep) and str(dep) in imread_paths:
                return imread_paths[str(dep)]
        layer = graph.layers.get(from_value)
        if layer is not None:
            for item in layer.values():
                for dep in getattr(item, 'dependencies', ()) or ():
                    if 'imread' in str(dep) and str(dep) in imread_paths:
                        return imread_paths[str(dep)]
        return None

    for layer_name, layer in graph.layers.items():
        if 'stack' not in str(layer_name):
            continue
        paths_by_z = {}
        for key, task in layer.items():
            if not (isinstance(key, tuple) and len(key) >= 2):
                continue
            z = key[1]
            if z in paths_by_z:
                continue
            from_value = None
            if isinstance(task, tuple) and len(task) >= 2:
                ref = task[1]
                if isinstance(ref, tuple) and ref:
                    from_value = ref[0]
                elif isinstance(ref, str):
                    from_value = ref
            if from_value is None:
                continue
            path = _imread_for_from_value(from_value)
            if path is not None:
                paths_by_z[z] = path
        if paths_by_z:
            return [paths_by_z[z] for z in sorted(paths_by_z)]

    return list(imread_paths.values())


def export_batch_segs():
    import os
    import math
    import dask.array as da
    import numpy as np
    from skimage import io

    import napari
    from napari.layers import Image, Labels
    from magicgui import magicgui
    from empanada_napari.utils import enable_layer_rename_refresh

    save_ops = {
        '2D images': '2D images',
        '3D image': '3D image',
    }

    @magicgui(
        call_button='Export labels',
        layout='vertical',
        dataset_name=dict(widget_type='LineEdit', value='', label='Folder name',
                          tooltip='Name to use for the dataset folder, creates a directory in the Save Directory with this name, it appends if directory already exists.'),
        save_dir=dict(widget_type='FileEdit', value='', label='Save directory', mode='d', tooltip='Directory in which to save segmentations'),
        export_type=dict(widget_type='RadioButtons', choices=list(save_ops.keys()), value=list(save_ops.keys())[0], label='Export type', tooltip='Exports segmentations as individual 2D images or a single 3d image.'),
        grayscale=dict(widget_type='CheckBox', value=False, label='Export grayscale image', tooltip='Convert to grayscale'),
    )
    def widget(
        viewer: napari.viewer.Viewer,
        image_layer: Image,
        labels_layer: Labels,
        export_type: str,
        dataset_name: str,
        save_dir: str,
        grayscale: bool,
    ):
        assert dataset_name, "Must provide a dataset name!"
        assert len(viewer.dims.order) <= 3, "Please use 'Save training patches' instead."

        outdir = os.path.join(save_dir, dataset_name)
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
            print('Created directory', outdir)
        else:
            print('Adding images to existing directory', outdir)

        if grayscale:
            os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(outdir, 'masks'), exist_ok=True)

        export_option = save_ops[export_type]
        image = image_layer.data
        mask = labels_layer.data

        assert image.shape[0] == mask.shape[0], \
        f"Image and labels layer must have the same number of images, got {image.shape} and {mask.shape}"

        if image.ndim == 3:
            if isinstance(image, da.Array):
                impaths = _get_impaths_from_dask(image)
                if len(impaths) == image.shape[0]:
                    imnames = [
                        '.'.join(os.path.basename(imp).split('.')[:-1]) + '.tiff'
                        for imp in impaths
                    ]
                else:
                    # Paths unavailable (unexpected graph); keep export working.
                    zpad = math.ceil(math.log(max(image.shape[0], 1), 10))
                    imnames = [
                        image_layer.name + '_' + str(n).zfill(zpad) + '.tiff'
                        for n in range(image.shape[0])
                    ]
            else:
                zpad = math.ceil(math.log(image.shape[0], 10))
                imnames = [image_layer.name + '_' + str(n).zfill(zpad) + '.tiff' for n in range(image.shape[0])]
                # imnames = [str(n).zfill(zpad) + '.tiff' for n in range(len(image))]

            if export_option == '3D image':
                # Creates a 3D image or 2D stack of images from the label image layer
                i, h, w = image.shape
                seg_stack = np.squeeze(mask[:i, :h, :w]).astype(np.int32)
                imname = image_layer.name + '.tiff'

                if grayscale:
                    img_stack = np.squeeze(image[:i, :h, :w])
                    io.imsave(os.path.join(outdir, f'images/{imname}'), img_stack, check_contrast=False)
                io.imsave(os.path.join(outdir, f'masks/{imname}'), seg_stack, check_contrast=False)

            else:
                for i in range(image.shape[0]):
                    imname = imnames[i]
                    if isinstance(image, da.Array):
                        h, w = image[i].compute().shape
                    else:
                        h, w = image[i].shape

                    img = np.squeeze(image[i, :h, :w])
                    seg = np.squeeze(mask[i, :h, :w]).astype(np.int32)

                    if grayscale:
                        io.imsave(os.path.join(outdir, f'images/{imname}'), img, check_contrast=False)
                    io.imsave(os.path.join(outdir, f'masks/{imname}'), seg, check_contrast=False)

        else:
            imname = image_layer.name + '.tiff'
            if grayscale:
                #imname = image_layer.name + '.tiff'
                io.imsave(os.path.join(outdir, f'images/{imname}'), image, check_contrast=False)
            io.imsave(os.path.join(outdir, f'masks/{imname}'), mask.astype(np.int32), check_contrast=False)


        print('Segmentations exported!')

    enable_layer_rename_refresh(widget)
    return widget

@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def export_batch_segs_widget():
    return export_batch_segs, {'name': 'Export Segmentations'}
