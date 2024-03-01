# empanada-napari

> [!IMPORTANT]
> **New Version Announcement!**
> * New modules 
>   * Morph labels - applies morphological operations to labels
>   * Count labels - counts and lists the label IDs within the dataset
>   * Filter labels - removes small pixel/voxel area labels or labels touching the image boundaries
>   * Export and import a model - export or import locally saved model files to use within empanada-napari
> * Updated modules 
>   * Export segmentations - now allows 3D segmentations to be exported as a single .tiff image
>   * Pick and save finetune/training patches - now allows paired grayscale and label mask images to create training patches 
>   * Split label - now allows users to specify new label IDs 
> * Updated documentation
>   * Check out the updated documentation [here](https://empanada.readthedocs.io/en/latest/empanada-napari.html)!

**The paper describing this work is now available [on Cell Systems](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00494-X).**

**Documentation for the plugin, including more detailed installation instructions, can be found [here](https://empanada.readthedocs.io/en/latest/empanada-napari.html).**

[empanada](https://github.com/volume-em/empanada) is a tool for deep learning-based panoptic segmentation of 2D and 3D electron microscopy images of cells.
This plugin allows the running of panoptic segmentation models trained in empanada within [napari](https://napari.org).
For help with this plugin please open an [issue](https://github.com/volume-em/empanada-napari/issues), for issues with napari specifically
raise an [issue here instead](https://github.com/napari/napari/issues).

## Implemented Models

  - *MitoNet*: A generalist mitochondrial instance segmentation model.

## Example Datasets

Volume EM datasets for benchmarking mitochondrial instance segmentation are available from
[EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/).

## Installation

It's recommended to have installed napari through [conda](https://docs.conda.io/en/latest/miniconda.html).
Then to install this plugin:

```shell
pip install empanada-napari==1.1.0
```

Launch napari:

```shell
napari
```

Look for empanada-napari under the "Plugins" menu.

![empanada](images/demo.gif)

## GPU Support

**Note: Mac doesn't support NVIDIA GPUS. This section only applies to Windows and Linux systems.**

As for any deep learning models, having a GPU installed on your system will significantly
increase model throughput (although we ship CPU optimized versions of all models with the plugin).

This plugin relies on torch for running models. If a GPU was found on your system, then you will see that the
"Use GPU" checkbox is checked by default in the "2D Inference" and "3D Inference" plugin widgets. Or if when running
inference you see a message that says "Using CPU" in the terminal that means a GPU is not being used.

Make sure that GPU drivers are correctly installed. In terminal or command prompt:

```shell
nvidia-smi
```

If this returns "command not found" then you need to [install the driver from NVIDIA](https://www.nvidia.com/download/index.aspx). Instead, if
if the driver is installed correctly, you may need to switch to the GPU enabled version of torch.

First, uninstall the current version of torch:

```shell
pip uninstall torch
```

Then [install torch >= 1.10 using conda for your system](https://pytorch.org/get-started/locally/).
This command should work:

```shell
conda install pytorch cudatoolkit=11.3 -c pytorch
```

## Citing this work

If you use results generated by this plugin in a publication, please cite:

```bibtex
@article {Conrad2022.03.17.484806,
	author = {Conrad, Ryan and Narayan, Kedar},
	title = {Instance segmentation of mitochondria in electron microscopy images with a generalist deep learning model},
	elocation-id = {2022.03.17.484806},
	year = {2022},
	doi = {10.1101/2022.03.17.484806},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/03/18/2022.03.17.484806},
	eprint = {https://www.biorxiv.org/content/early/2022/03/18/2022.03.17.484806.full.pdf},
	journal = {bioRxiv}
}
```
