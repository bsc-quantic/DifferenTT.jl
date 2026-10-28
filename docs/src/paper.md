# Reproducing the paper

The library was developed together with the paper P. Torrente et al., *Factorized approximate inverse preconditioners in
the tensor train format* (in preparation), which introduces the factorized approximate inverse (FAI) preconditioner of
[`fai_preconditioner`](@ref) and tests it on the anisotropic elliptic operators of its Sec. 5.1 and on the Chorin
projection around a cylinder of its Sec. 5.2. The scripts that generate the figures and tables of the paper are in the
repository, in
[`experiments/paper`](https://github.com/bsc-quantic/DifferenTT.jl/tree/master/experiments/paper), and its
[`README.md`](https://github.com/bsc-quantic/DifferenTT.jl/blob/master/experiments/paper/README.md) lists, for each
figure and table, the script that produces it, the environment, and the SLURM settings of the runs on the cluster.

The figures and tables of the paper were produced with the scripts of `experiments/paper` at the tag
[`paper-ttfai-v0.1-data`](https://github.com/bsc-quantic/DifferenTT.jl/tree/paper-ttfai-v0.1-data) (commit `3945c32`),
on the release version of the library, and the data are archived on Zenodo:
[doi:10.5281/zenodo.23663326](https://doi.org/10.5281/zenodo.23663326). The README of the archive gives the provenance of
each file and where to copy it to redraw the figures with the scripts of `experiments/paper`.

Earlier drafts of the paper used the data of the tag
[`paper-ttfai-data`](https://github.com/bsc-quantic/DifferenTT.jl/tree/paper-ttfai-data) (commit `6850540`), produced
with an earlier version of the library and the scripts of the paper repository. Several of the changes since then alter
numbers, for instance the orientation of the FAI factor and the convention of the cutoff of [`ResidualALS`](@ref), and
the arguments of some functions changed, so the results of the two versions must not be mixed in one collection or plot.
The changes are listed in the release notes (`NEWS.md`), and the README of `experiments/paper` lists the three that
matter for the paper.

The heavy runs of the paper need a cluster and are not part of the documentation build or of the tests. The experiment
scripts also have a smoke mode, with tiny sizes, which only checks that they run.
