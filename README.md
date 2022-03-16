# Empanada-Napari

**Documentation for the plugin, including more detailed installation instructions, can be found at [empanada.readthedocs.io/en/latest/empanada-napari.html](empanada.readthedocs.io/en/latest/empanada-napari.html).

[empanada](https://github.com/volume-em/empanada) is a tool for deep learning-based panoptic segmentation of 2D and 3D electron microscopy images of cells.
This plugin allows the running of panoptic segmentation models trained in empanada within [napari](https://napari.org).
For help with this plugin please open an [issue](https://github.com/volume-em/empanada-napari/issues), for issues with napari specifically
raise an [issue here instead](https://github.com/napari/napari/issues).

## Implemented Models

*MitoNet*: A generalist mitochondrial instance segmentation model.

## Installation

It's recommended to have installed napari through (conda)[https://docs.conda.io/en/latest/miniconda.html].
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


## Example Datasets

Volume EM datasets for benchmarking mitochondrial instance segmentation are available from
[EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/).
