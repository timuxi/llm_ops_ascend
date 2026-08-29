#!/bin/bash
set -e

DEBUG_MODE="OFF"
while getopts ":hd" opt; do
    case ${opt} in
        d ) DEBUG_MODE="ON" ;;
        h )
            echo "Usage: ./build.sh [SOC_VERSION]"
            echo "  Builds the Python/PyTorch wrapper only."
            echo "  Compile operator C++ in ops-transformer (--pkg), not in this repo."
            echo "  SOC_VERSION default: Ascend910B3"
            exit 0
            ;;
        \? ) echo "Unknown flag: -$OPTARG" 1>&2; exit 1 ;;
    esac
done
shift $((OPTIND -1))

export DEBUG_MODE=$DEBUG_MODE
SOC_VERSION="${1:-Ascend910B3}"

_CANN_TOOLKIT_INSTALL_PATH=$(grep "Toolkit_InstallPath" /etc/Ascend/ascend_cann_install.info | awk -F'=' '{print $2}')
source "${_CANN_TOOLKIT_INSTALL_PATH}/set_env.sh"
echo -e "\e[1;32mCANN Toolkit: ${_CANN_TOOLKIT_INSTALL_PATH}\e[0m"
echo -e "\e[1;32mASCEND_HOME_PATH: ${ASCEND_HOME_PATH}\e[0m"
echo -e "\e[1;32mSOC_VERSION: ${SOC_VERSION}\e[0m"

CURRENT_DIR=$(pwd)
OUTPUT_DIR=$CURRENT_DIR/output
mkdir -p "$OUTPUT_DIR"

JOBS="$(nproc)"

TORCH_BUILD_DIR=$CURRENT_DIR/build/torch
rm -rf "$TORCH_BUILD_DIR"
mkdir -p "$TORCH_BUILD_DIR"

cmake -S "$CURRENT_DIR" -B "$TORCH_BUILD_DIR" \
  -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION="$SOC_VERSION"

cmake --build "$TORCH_BUILD_DIR" --target install -j"$JOBS"

if ! pip3 show wheel >/dev/null 2>&1; then
    pip3 install wheel==0.45.1
fi

cd utils/python || exit
rm -rf dist build *.egg-info
cp -v "${CURRENT_DIR}/config.ini" "${CURRENT_DIR}/utils/python/llm_ops/"
python3 setup.py clean --all
python3 setup.py bdist_wheel
mv -v dist/llm_ops*.whl "${OUTPUT_DIR}/"
rm -rf dist build *.egg-info
cd - >/dev/null

echo -e "\e[1;32mWheel: $(ls ${OUTPUT_DIR}/llm_ops*.whl)\e[0m"
echo -e "\e[1;33mOperator C++ is not built here. Compile ops-transformer with --pkg if you need libcust_opapi.so.\e[0m"
