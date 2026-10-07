# Source this file from the project directory after installing requirements.
COFFEE_CUDA_LIBS=$(.venv/bin/python -c 'from pathlib import Path; import site; print(":".join(str(p) for p in (Path(site.getsitepackages()[0]) / "nvidia").glob("*/lib")))')
export LD_LIBRARY_PATH="${COFFEE_CUDA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${PWD}/.venv/bin:${PATH}"
