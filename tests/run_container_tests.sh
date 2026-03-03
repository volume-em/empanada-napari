#!/bin/bash

# Whether to run a longer tests, if /tests/test_install_extended.py is provided
EXTENDED_TESTS=${EXTENDED_TESTS:-false}
# Whether to skip start-up tests entirely
SKIP_TESTS=${SKIP_TESTS:-false} 
# Name for this container application
NAME="Empanada-napari"

if [ "$SKIP_TESTS" = "true" ]; then
    echo "Skipping installation tests (SKIP_TESTS=true)"
    echo ""
	exit 0
fi

echo -e "$NAME Container Tests\n"

if ! python -m pytest /tests/test_install.py -v --tb=line; then
	echo -e "WARNING: Some tests failed. Container may not run correctly."
fi

if [[ "$EXTENDED_TESTS" = "true" && /tests/test_install_extended.py ]]; then
	if ! python -m pytest /tests/test_install_extended.py -v --tb=line; then
		echo -e "WARNING: Some extended tests failed."
	fi
fi

echo -e "\nTests complete"

# Print empanada version + changelog link
echo -e "\nInstallation information"
python --version
napari --version
python -c "import empanada_napari; print(f'Empanada-Napari Version: {empanada_napari.__version__}')"
echo "Documentation: https://empanada.readthedocs.io/en/latest/ | Changelog: https://github.com/volume-em/empanada-napari/releases"
# Print versions of models with this version of Empanada:
python -c "from empanada_napari.utils import get_configs; model_configs=get_configs(); model_names=list(model_configs.keys()); print('Available Models:', *model_names)"
echo ""
