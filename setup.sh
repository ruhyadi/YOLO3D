#!/bin/bash
# YOLO3D Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 YOLO3D Project Setup${NC}"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for conda/mamba
if command_exists mamba; then
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓ Found mamba${NC}"
elif command_exists conda; then
    CONDA_CMD="conda"
    echo -e "${GREEN}✓ Found conda${NC}"
else
    echo -e "${RED}✗ Neither conda nor mamba found. Please install Miniconda or Anaconda.${NC}"
    exit 1
fi

# Check if environment exists
ENV_NAME="yolo3d"
if $CONDA_CMD env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment ${ENV_NAME} already exists.${NC}"
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing environment...${NC}"
        $CONDA_CMD env remove -n $ENV_NAME -y
    else
        echo -e "${GREEN}Using existing environment.${NC}"
        $CONDA_CMD activate $ENV_NAME
        exit 0
    fi
fi

# Create conda environment
echo -e "${GREEN}Creating conda environment...${NC}"
$CONDA_CMD env create -f environment.yaml -n $ENV_NAME

# Activate environment
echo -e "${GREEN}Activating environment...${NC}"
source $($CONDA_CMD info --base)/etc/profile.d/conda.sh
$CONDA_CMD activate $ENV_NAME

# Verify installation
echo -e "${GREEN}Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import lightning; print(f'Lightning version: {lightning.__version__}')"
python -c "import hydra; print(f'Hydra version: {hydra.__version__}')"

echo -e "${GREEN}✅ Environment setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Download KITTI dataset to data/KITTI/"
echo "3. Train model: python src/train.py --config-name=train_yolo3d"
echo "4. Serve model: python serve.py"
echo "5. Test API: python test_api.py --create-dummy"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"