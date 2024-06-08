<center> <h1>Alt-Embed: Exploring Alternate Embeddings for RefCM.</h1> </center>


## Description <a name="Description"></a>

Because query and reference datasets sometimes drastically differ in provenance, the gene expression space that our data lies in can also be quite different. For instance, mice don't have all of the same genes as us humans do, and even within a species, some tissues don't present the same gene expressions as others do*. The very first step and requirement in the RefCM pipeline is finding a common embedding space between query and reference datasets, from which we can then evaluate similarity between clusters. Currently, we intersect the gene spaces, and select those genes which provide the most information for each dataset according to scanpy's `hvg` function. However, there are many embeddings left to try out for potentially equal or better performance in the downstream matching tasks, i.e. the OT- and LP-steps. The goal for this project, is to examine current embedding methods in the scRNA-seq literature -- especially non-linear ones like [Saturn](https://www.nature.com/articles/s41592-024-02191-z), [VARs](https://backend.orbit.dtu.dk/ws/portalfiles/portal/216210029/btaa293.pdf), [Graph Neural Networks](https://www.nature.com/articles/s41467-021-24172-y), or [Geometric Embeddings](https://www.nature.com/articles/s41467-021-22851-4)  -- and their effect on RefCM matching. 

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
git switch alt-embed
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

This specific project will require a very crisp understanding of the RefCM pipeline as found in our paper draft, and especially in `src/refcm.py`. If this interests you, please let me know by email so we can walk through the code together and decide on what embeddings to try out!