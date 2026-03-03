#!/bin/bash

BENCHMARK=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --help | -h)
            echo "Usage: $0 [--benchmark] flag will run performance benchmark tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run Unit Tests:
echo "Running Unit Tests..."
python -m pytest -s -vv test_array_utils.py
python -m pytest -s -vv test_zarr_utils.py
python -m pytest -s -vv test_transforms.py

# Run Integration Tests:
echo "Running Integration Tests..."
python -m pytest -s -vv test_button_widgets.py

if [ "$BENCHMARK" = true ]; then
    # Run Model Benchmarks:
    echo "Running Model Benchmarks..."
    python -m pytest -s -vv test_model_benchmark.py
fi

echo "Tests Complete."

# Print empanada version + changelog link
python --version
napari --version
python -c "import empanada_napari; print(f'Empanada-Napari Version: {empanada_napari.__version__}')"
echo "Documentation: https://empanada.readthedocs.io/en/latest/ | Changelog: https://github.com/volume-em/empanada-napari/releases"
# Print versions of models with this version of Empanada:
python -c "from empanada_napari.utils import get_configs; model_configs=get_configs(); model_names=list(model_configs.keys()); print('Available Models:', *model_names)"
