# encoding: utf-8
# Cython-based ReID evaluation module
# Author: liaoxingyu
# Contact: sherlockliao01@gmail.com


def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess
    import sys

    path = os.path.abspath(os.path.dirname(__file__))

    # Windows compatibility: use setup.py instead of make
    if os.name == 'nt':  # Windows
        setup_py = os.path.join(path, "setup.py")
        if os.path.exists(setup_py):
            ret = subprocess.run([sys.executable, setup_py, "build_ext", "--inplace"], cwd=path)
        else:
            print("setup.py not found for Windows compilation")
            return
    else:  # Unix-like systems
        ret = subprocess.run(["make", "-C", path])

    if ret.returncode != 0:
        print("Making cython reid evaluation module failed, exiting.")
        sys.exit(1)
