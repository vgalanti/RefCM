[![Python Versions](https://img.shields.io/badge/python-3.11-blue)](https://pypi.org/project/alpaca-py)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

<center> <h1>RefCM: Reference Cluster-Mapping</h1> </center>

RefCM is an automated tool enabling cell-type annotation across different scRNA-seq datasets. The model bases itself on the geometric properties of optimal transport to map cell-type clusters across tissues, sequencing methods, and species.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [About ](#about-)
- [Installation ](#installation-)
- [Usage ](#usage-)
  - [Jupyter Notebook Examples ](#jupyter-notebook-examples-)
<!-- ## Documentation <a name="documentation"></a> -->

## About <a name="about"></a>

The primary purpose of this repository is to enable the reproduction of the results reported in the paper, and help others utilize this method towards their own work.


## Installation <a name="installation"></a>

We recommend creation of a new conda environment with the required dependencies using the provided [yaml](./env.yml) file:

```shell
  conda env create -n refcm -f env.yml
  conda activate refcm
```

To download the datasets we used, as well as see their source links and how we processed them, please view/run the [data setup notebook](./data/setup.ipynb).


## Usage <a name="usage"></a>

Running RefCM on a given `query: AnnData` and `reference: AnnData` dataset pair, assuming clustering information under their respective `.obs['cluster']` attributes, can be achieved by running the following code and tweaking the remaining hyperparameters. The method expects raw counts to be provided under each dataset's `.X` attribute.

```python
    from refcm import RefCM
    
    rcm = RefCM()
    m = rcm.annotate(query, 'query', reference, 'reference', 'cluster', 'cluster')
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


For convenience, we have also included a CLI tool.

```shell
    refcm-cli --usage
```


### Jupyter Notebook Examples <a name="notebook-examples"></a>

We have put together some examples in jupyter notebooks under [/experiments](./experiments/) so that you can start utilizing RefCM right away!


