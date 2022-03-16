# empanada-napari

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
pip install empanada-napari
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
