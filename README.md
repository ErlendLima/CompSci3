# CompSci - Project 3
## Overview

The source code is written in Python and is available at [src](src), with the Jupyter notebooks
at [notebooks](notebooks). The most important is [Plotting](notebooks/Plotting.ipynb), for creating
and plotting the data. The raw data sets are at [data](data), along with the stylesheet used. 
Running the entire notebooks takes about half an hour, but the plots consume several tens of gigabytes.

## Overview of sourcecode
Description of the contents of [src](src) directory.

| Script         | Function                                                                 |
|----------------|--------------------------------------------------------------------------|
 | `model.py`     | `Model` classes and `Noise` classes for modelling the data and its noise |
| `data.py`      | Reading, preprocessing and handling of the data                          |
| `posterior.py` | Analyses the output of multinest                                         |
| `marginal.py`  | Plotting the marginalised posterior distributions                        |
| `stubs.py`     | Stub file for mypy.                                                      |
| `utils.py`     | Miscellaneous functions, mainly for plotting.                            |



