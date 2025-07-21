[![PyPI version](https://img.shields.io/pypi/v/refcm.svg?color=blue)](https://pypi.org/project/refcm)
[![Python versions](https://img.shields.io/pypi/pyversions/refcm.svg)](https://pypi.org/project/refcm/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<center> <h1>RefCM: Reference Cluster-Mapping</h1> </center>

RefCM is an automated tool for cell type annotation across single-cell RNA-seq datasets. It leverages optimal transport to align cell-type clusters across tissues, sequencing technologies, and species.

This repository supports reproducing the results from our paper and helps others apply RefCM to their own datasets.

![overview](https://github.com/vgalanti/RefCM/blob/main/vignettes/overview.jpeg?raw=true)


## Table of Contents
- [Table of Contents](#table-of-contents)
- [About](#about)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Examples](#examples)
- [Citation & Contact](#citation--contact)


## About <a name="about"></a>

The primary purpose of this repository is to enable the reproduction of the results reported in the paper, and help others utilize this method towards their own work.


## Installation <a name="installation"></a>

RefCM is available on PyPI. You can install it using pip:
```shell
pip install refcm
```

If you want to build from source, you can use [uv](https://docs.astral.sh/uv/):
```shell
uv pip install -e ".[dev]"
```

**Note**: RefCM depends on the [GLPK solver](https://www.gnu.org/software/glpk/).

- On macOS, install via Homebrew with `brew install glpk`.
- On Windows, follow [this guide](https://stackoverflow.com/questions/17513666/installing-glpk-gnu-linear-programming-kit-on-windows).

After installation, restart your IDE or terminal to ensure the solver is recognized.

To download the datasets used in our study, please visit our [Google Drive folder](https://drive.google.com/drive/folders/1fWWaxBLUdacBT9r-1CymdyRICMPStvBJ?usp=share_link), along with the accompanying [data setup notebook](https://github.com/vgalanti/RefCM/blob/main/data/setup.ipynb), which includes source links and preprocessing instructions.


## Requirements <a name="requirements"></a>

The package has been tested on:

- macOS Sequoia (Apple M1 Pro, 32 GB RAM)
- Windows 11 (Intel i5 4-core CPU, 8 GB RAM)

There are no strict hardware requirements, aside from the ability to load the query and reference datasets into memory. All analyses, except for the large embryogenesis datasets, were reproducible on the 8 GB Windows system. On the M1 Pro, even the largest datasets completed in under 20 minutes.

Installation typically takes under 5 minutes, depending on network speed.


## Usage <a name="usage"></a>

Running RefCM on a given `query: AnnData` and `reference: AnnData` dataset pair, assuming clustering information under their respective `.obs['cluster']` attributes, can be done as follows. The method expects raw counts to be provided under each dataset's `.X` attribute.

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

We’ve put together a few example Jupyter notebooks under the [vignettes](https://github.com/vgalanti/RefCM/tree/main/vignettes) folder.

We recommend starting with [brain.ipynb](https://github.com/vgalanti/RefCM/blob/main/vignettes/brain.ipynb), which applies RefCM to the Allen Brain Atlas.


## Citation & Contact <a name="citation--contact"></a>

If you use RefCM in your work, please consider citing our paper (link coming soon).  
Feel free to open an issue or pull request — contributions and feedback are welcome!