"""Builds advgen's Cython extension (advgen/utils_cython.pyx) in place for the
active Python, producing advgen/utils_cython.<abi>.so. The generated C file
and the compiled extension are build output and are not tracked.

Needs a C compiler (Ubuntu: apt install build-essential), Cython 0.29.x (the
version the extension was written for) and numpy. Run from the repository root:

    pip install "cython>=0.29.34,<3" && python scripts/responsibility/build_cython.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    import numpy
    from Cython.Build import cythonize
    from setuptools import Extension, setup

    os.chdir(ROOT)
    extension = Extension("advgen.utils_cython", ["advgen/utils_cython.pyx"], include_dirs=[numpy.get_include()])
    setup(
        name="advgen-utils-cython",
        ext_modules=cythonize([extension], compiler_directives={"language_level": 3}, quiet=True),
        script_args=["build_ext", "--inplace"],
    )
    sys.path.insert(0, str(ROOT))
    import advgen.utils_cython  # noqa: F401

    print(f"built {advgen.utils_cython.__file__}")


if __name__ == "__main__":
    main()
