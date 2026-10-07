# XRT, then the IRON venv and mlir-aie's own environment (PATH, PYTHONPATH,
# PEANO_INSTALL_DIR, NPU detection) for every login shell.
. /opt/xilinx/xrt/setup.sh >/dev/null
. /opt/mlir-aie/ironenv/bin/activate
. /opt/mlir-aie/utils/env_setup.sh >/dev/null
