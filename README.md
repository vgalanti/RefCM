[![Python Versions](https://img.shields.io/badge/python-3.11-blue)](https://pypi.org/project/alpaca-py)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<center> <h1>RefCM: Reference Cluster-Mapping</h1> </center>


RefCM is an automated tool enabling cell-type annotation across different scRNA-seq datasets. The model bases itself on the geometric properties of optimal transport to map cell-type clusters across tissues, sequencing methods, and species.

![overview](/vignettes/overview.jpeg)

## Table of Contents
- [Table of Contents](#table-of-contents)
- [About](#about)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Examples ](#examples)

## About <a name="about"></a>

The primary purpose of this repository is to enable the reproduction of the results reported in the paper, and help others utilize this method towards their own work.


## Installation <a name="installation"></a>

We recommend using [uv](https://docs.astral.sh/uv/). After installation, simply run `uv sync` to install the required virtual environment and compatible python version.

For conda users, we have provided a [yaml file](./env.yml):
```shell
conda env create -n refcm -f env.yml
conda activate refcm
```

Additionally, users will need to install GLPK. On MacOS, this can be done via homebrew with `brew install glpk`. Windows users require a few more steps, described in [this guide](https://stackoverflow.com/questions/17513666/installing-glpk-gnu-linear-programming-kit-on-windows). Please restart your IDE after setup.

To download the datasets in our study, please visit our [Google Drive link](https://drive.google.com/drive/folders/1fWWaxBLUdacBT9r-1CymdyRICMPStvBJ?usp=share_link) and accompanying [data setup notebook](./data/setup.ipynb) which includes source links and data preprocessing instructions.

## Requirements <a name="requirements"></a>

The package has been tested on:

* macOS Sequoia (Apple M1 Pro, 32 GB RAM)
* Windows 11 (Intel i5 4-core CPU, 8 GB RAM)

There are no strict hardware requirements, aside from the ability to load the query and reference datasets into memory. All analyses, except for the large embryogenesis datasets, were reproducible on the 8 GB Windows system. On the M1 Pro, even the largest datasets completed in under 20 minutes.

Installation typically takes under 10 minutes (from cloning the repository to running example scripts), depending on network speed.


## Usage <a name="usage"></a>

Running RefCM on a given `query: AnnData` and `reference: AnnData` dataset pair, assuming clustering information under their respective `.obs['cluster']` attributes, can be achieved by running the following code and tweaking the remaining hyperparameters. The method expects raw counts to be provided under each dataset's `.X` attribute.

```python
    from refcm import RefCM
    
    rcm = RefCM()
    rcm.setref(reference, 'reference', 'cluster')
    m = rcm.annotate(query, 'query', 'cluster')
```

The resulting annotations are written out to the `.obs['refcm_annot']` field in the query's `AnnData` object, leaving the remaining fields unchanged.

For a graphical representation of the resulting matching/cluster-mapping:

```python
    m.display_matching_costs()
```

Provided a ground-truth `.obs` field, the matching's performance can be evaluated as follows:

```python
    m.display_matching_costs(ground_truth_obs_key='ground_truth_key')
    m.eval(ground_truth_obs_key='ground_truth_key')
```


### Examples <a name="examples"></a>

We’ve put together a few example Jupyter notebooks under [/vignettes](./vignettes/) to help you start using RefCM right away. Our starter example, [brain.ipynb](vignettes/brain.ipynb), walks you through applying the method to the Allen Brain Atlas datasets.