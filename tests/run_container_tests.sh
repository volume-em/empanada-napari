#!/bin/bash

# Run Installation Tests:
echo "Running Installation Tests..."
python -m pytest -s -vv test_installation.py

echo "Tests Complete."

# Print empanada version + changelog link
python --version
napari --version
python -c "import empanada_napari; print(f'Empanada-Napari Version: {empanada_napari.__version__}')"
echo "Documentation: https://empanada.readthedocs.io/en/latest/ | Changelog: https://github.com/volume-em/empanada-napari/releases"
# Print versions of models with this version of Empanada:
python -c "from empanada_napari.utils import get_configs; model_configs=get_configs(); model_names=list(model_configs.keys()); print('Available Models:', *model_names)"