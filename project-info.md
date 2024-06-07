<center> <h1>Type-Equiv: Establishing equivalence between dataset types.</h1> </center>


## Description <a name="Description"></a>

One of the common issues when mapping between datasets, is the naming that the authors chose to annotate their data. Some cases are very clear cut, where the naming only depends on lowercase or uppercase spelling. For example, the authors of the PBMC datasets chose to label their cell types with uppercase names, whereas authors of the Celltypist/TS datasets instead chose to label theirs with lowercase names. Some will chose to call cells by the short-form acronyms, say "Treg", whereas others opt for the full name, "Regulatory T Cell". In other cases, the distinction is a little more subtle. Here is what trying to run the matching algorithm between the PBMC Smart-Seq2 and Celltypist Bone-marrow datasets yields:
```shell
[refcm           ] [DEBUG   ] : starting LP optimization
[refcm           ] [DEBUG   ] : optimization terminated w. status "Optimal"
[matchings       ] [DEBUG   ] : [-] B cell               mapped to naive B cell        
[matchings       ] [DEBUG   ] : [-] CD14+ monocyte       mapped to classical monocyte  
[matchings       ] [DEBUG   ] : [-] CD16+ monocyte       mapped to non-classical monocyte
[matchings       ] [DEBUG   ] : [-] CD4+ T cell          mapped to effector memory CD8-positive, alpha-beta T cell
[matchings       ] [DEBUG   ] : [-] Cytotoxic T cell     mapped to effector memory CD8-positive, alpha-beta T cell, terminally differentiated
[matchings       ] [DEBUG   ] : [-] Megakaryocyte        mapped to megakaryocyte       
[matchings       ] [INFO    ] : pbmc_Smart-Seq2      to Bone_marrow  
```
Clearly, our methodly incorrectly states that the mapping "Megakaryocte" to "megakaryocyte" is incorrect. This case is simple enough to adapt to, the others not so much. Can we equate a "B cell" from the query dataset to a "naive B cell" of the reference dataset? How did either dataset authors converge on their respective classification of these clusters? Similarly, can we equate "CD14+ monocyte" to "classical monocyte," or "CD16+ monocyte" with "non-classical monocyte"? If these are not the same, should we consider these mappings as incorrect as instead mapping either of the T cell clusters to the monocyte clusters instead? 

Additionally, some of the annotations directly depend on the tissue the cells are sampled from. For example, mapping the Heart dataset to the Liver dataset (celltypist) yields the following mapping:

```shell
[matchings       ] [DEBUG   ] : [-] endothelial cell of lymphatic vessel mapped to endothelial cell of pericentral hepatic sinusoid
```
Clearly, given that the "pericentral hepatic sinusoid" is a blood vessel in the liver, the user attempting to map their Heart data could easily infer that our method suggests these are more generally endothelial cells, rather than cells stemming from the liver entirely. In summary, a user possesses knowledge on the data they are providing and the data they are mapping to, and is able to make straight-forward inferences for some mappings. So, if not "how accurate" a certain mapping is, how can we think about quantifying "how valuable" or "how informative" our annotations are?

## Ideas <a name="Ideas"></a>
Cell types follow a certain hierarchy (see the Allen-Brain datasets [in this notebook](/experiments/hierarchical.ipynb)). How can we represent it? How can we quantify the loss of valuable information as the "common ancestor" of these types becomes more distant (metric)? What are some entirely equivalent classes across these datasets, that are only a question of re-naming?

Here is a potentially interesting reference to check out: [link](https://academic.oup.com/nargab/article/5/3/lqad070/7231336?login=false).

Also, Saturn does some renaming. Check their [Github](https://github.com/snap-stanford/SATURN). Under `Vignettes/frog_zebrafish_embryogenesis/data` you will see some mappings the authors chose to simplify mapping the datasets to one another. Check their original paper to see how they came to these conclusions (if they specifically address it).

In general, this project will require you dig into the datasets' original papers and the biology behind the cell-types. From a computational standpoint, you will want to explore how to represent these types as equivalence classes and a hierarchy, and perhaps come up with a metric on the tree to measure the similarity between types. Ultimately, we should aim to be able to equate cell types that are indeed the same up to naming, and provide some, as quantitave as possible, measure of information that our mapping provides, if it is not 100% accurate.

## Getting started
### Git Branches

Before anything, make sure that your `local` git repo is up to date with the following lines. These will ensure that all `remote` branches are visible to you. You should not need to run these two commands again.
```shell
git fetch --all
git pull --all
```

Next, make sure you are in the correct branch. You can do this by running: 
```shell
git branch -a
```
This command will show a list of all branches that are available on your `local` and `remote` git repository, and will mark the branch you are currently in with an asterisk `*`. By default, you should be in `main`. Most of the branches here should follow the format `remotes/origin/<branch>`. This indicates that these are `remote` branches that you can view, but not yet edit and contribute to.

To instead switch to the branch you want to work on, you will need to run:
```shell
git switch <branch>
```
where `<branch>` was the `remote` branch listed when running `git branch -a`, ignoring the "remotes/origin" prefix. For example, for this particular branch, you would want to run
```shell
git switch type-equiv
```
This will create a local version of the branch that you can work on and push to `remote` as you see fit. If you run `git branch -a`, you should now be able to see this new branch without the "remotes/origin" prefix, and it should be marked as your current branch with `*`.

To then navigate between branches, use the `git checkout <branch>` command. It is vital that you make sure that you are in the right branch before making any changes to file.

## Git Commit, Push & Pull

Once you have made changes to your `local` repository that you would like to push to `remote`, i.e. so that they are visible on GitHub, you will need to follow the following set of steps.

- First, run `git status` to view a summary of the changes that you have made since the last commit.
- Run `git add .` to add all these changes to your current commit.
- Run `git commit -m <commit-message>` to bundle these added changes together along with a concise explanation of what your changes bring. For example, `git commit -m "fixed dependency issues"`.
- Finally, run `git push` to push your local commit onto the `remote`.

If you are working on the same branch with a collaborator, you may in some instances need to use `git pull` to update your local repository to match their latest pushes. In most cases, if you are not working on the exact same lines of code, git should be able to automatically merge your changes together. However, in other cases, you may need to resolve merge conflicts. Try to coordinate with your collaborator so you don't run into these merge conflicts, or do VS code live-sharing sessions to edit simultaneously. If merge conflicts happen (and they do) please reach out for help on how to resolve them.

## Code !
For this specific task, look for the function `eval_link` in the file `src/matchings.py` to find more information (and hints) on how and where you should add your code. Happy coding! :)