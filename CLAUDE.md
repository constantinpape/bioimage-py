We will build a Python library for efficient and scalable image analysis together.
The draft for the design is explained in DESIGN_DOC.md.

# Library Structure

TODO: the library will be in the folder bioimage_py.
It will create a sub-package `runner` to implement the runner logic and dedicated sub-packages for other
functionality, such as `stats` (e.g. to impleent max, min, mean, std, etc.), `segmentation`, `wrapper` (for on-the-fly transformations of the input data), etc.

# Installation 

TODO: create pyproject and setup.cfg for this.

# Tests

TODO: Tests will be written with pytest

# Coding standards etc.

Code should be PEP8-compliant (line limit 120), use type annotations in the function definitions and
google-style doc strings. The documentation will later be build with pdoc (so you can already use specific
conventions from it if needed).

Use pyflakes and flake8 for linting.
