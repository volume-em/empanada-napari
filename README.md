# empanada-napari

> [!IMPORTANT]
> **New Version 1.2 Announcement!**
> * New Models:
>   * NucleoNet: A base model for EM instance nucleus segmentation
>   * DropNet: A base model for EM Lipid Droplet instance segmentation
> * New modules 
>   * Archive Model - archives the model into a hidden archive folder 
>   * Create Tiles - Split big images winto small tiles (with Mask/ optional)
>   * Merge Tiles - Opposite of split tiles, create full image from tiles created by create tiles
>   * Morph labels - applies morphological operations to labels
>   * Count labels - counts and lists the label IDs within the dataset
>   * Filter labels - removes small pixel/voxel area labels or labels touching the image boundaries
>   * Export and import a model - export or import locally saved model files to use within empanada-napari
> * Updated modules
>   * 2D Inference - now allows ROI inference and filling holes in segmentation label maps
>   * 3D Inference - allows erosion, dilation and fill holes in segmentation to help tackle big split merge errors
>   * Export segmentations - now allows 3D segmentations to be exported as a single .tiff image
>   * Pick and save finetune/training patches - now allows paired grayscale and label mask images to create training patches 
>   * Split label - now allows users to specify new label IDs 
> * Updated documentation
>   * Check out the updated documentation [here](https://empanada.readthedocs.io/en/latest/index.html)!

**The paper describing this work is now available [on Cell Systems](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00494-X).**

**Documentation for the plugin, including more detailed installation instructions, can be found [here](https://empanada.readthedocs.io/en/latest/empanada-napari.html).**

empanada is a tool for deep learning-based panoptic segmentation of 2D and 3D electron microscopy images of cells.
This plugin allows the running of panoptic segmentation models trained in empanada within [napari](https://napari.org).
For help with this plugin please open an [issue](https://github.com/volume-em/empanada-napari/issues), for issues with napari specifically
raise an [issue here instead](https://github.com/napari/napari/issues).

## Implemented Models

  - *MitoNet*: A generalist mitochondrial instance segmentation model
  - *NucleoNet*: A base model for EM instance nucleus segmentation
  - *DropNet*: A base model for EM Lipid Droplet instance segmentation

## Example Datasets

Volume EM datasets for benchmarking mitochondrial instance segmentation are available from
[EMPIAR-10982](https://www.ebi.ac.uk/empiar/EMPIAR-10982/).

## Installation

### New Users

empanada-napari requires **Python ≥ 3.13** and **napari ≥ 0.9** (including 0.10.x).

#### pip

```shell
python3.13 -m venv empanada-env
source empanada-env/bin/activate   # Windows: empanada-env\Scripts\activate
python -m pip install -U pip
python -m pip install "napari[all]>=0.9"
python -m pip install empanada-napari
```

#### conda

```shell
conda create -n empanada-env python=3.13 -y
conda activate empanada-env
python -m pip install -U pip
python -m pip install "napari[all]>=0.9"
python -m pip install empanada-napari
```

Launch napari from the same environment:

```shell
python -m napari
```

Prefer `python -m napari` over bare `napari` so the plugin loads from this environment (some systems still have an older `napari` earlier on `PATH`).

Look for **empanada-napari** under the Plugins menu.


### Returning Users

Create or activate a Python 3.13 environment, then upgrade:

```shell
conda activate empanada-env
python -m pip install -U "napari[all]>=0.9" empanada-napari
python -m napari
```

Older installs on Python 3.10–3.12 or napari 0.6.x are not compatible with this release. Use a new 3.13 environment instead of upgrading in place.


### Development

Clone the repository, then:

```shell
conda create -n empanada-dev python=3.13 -y
conda activate empanada-dev
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m napari   # launch napari with the plugin
pytest             # run the tests
```



![empanada](images/demo.gif)

## GPU Support

**Note: Apple Silicon GPUs are supported through torch's MPS backend when available. The setup instructions below apply to NVIDIA GPUs on Windows and Linux.**

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

Reinstall torch with a CUDA build matching your system using the selector at
[pytorch.org](https://pytorch.org/get-started/locally/), for example:

```shell
pip install --upgrade --force-reinstall torch torchvision
```


## Citing this work

If you use results generated by this plugin in a publication, please cite:

```bibtex
@article { Conrad2023,
    author = {Conrad, Ryan and Narayan, Kedar},
    title = {Instance segmentation of mitochondria in electron microscopy images with a generalist deep learning model trained on a diverse dataset},
    journal = {Cell Systems},
    year = {2023},
    month = {Jan},
    day = {18},
    publisher = {Elsevier},
    volume = {14},
    number = {1},
    pages = {58-71.e5},
    issn = {2405-4712},
    doi = {10.1016/j.cels.2022.12.006},
    url = {https://doi.org/10.1016/j.cels.2022.12.006}
}
```
