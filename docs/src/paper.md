# Reproducing the paper

The library was developed together with the paper P. Torrente et al., *Factorized approximate inverse preconditioners in
the tensor train format* (in preparation), which introduces the factorized approximate inverse (FAI) preconditioner of
[`fai_preconditioner`](@ref) and tests it on the anisotropic elliptic operators of its Sec. 5.1 and on the Chorin
projection around a cylinder of its Sec. 5.2. The scripts that generate the figures and tables of the paper are in the
repository, in
[`experiments/paper`](https://github.com/bsc-quantic/DifferenTT.jl/tree/master/experiments/paper), and its
[`README.md`](https://github.com/bsc-quantic/DifferenTT.jl/blob/master/experiments/paper/README.md) lists, for each
figure and table, the script that produces it, the environment, and the SLURM settings of the runs on the cluster.

The library has changed while the paper was written. Several of the changes alter numbers (for instance the orientation
of the FAI factor and the convention of the cutoff of [`ResidualALS`](@ref)), and the arguments of some functions
changed. The scripts follow the current API, and the data of the paper are regenerated with the current version before
submission, so numbers obtained with older versions need not agree digit by digit. The changes are listed in the release
notes (`NEWS.md`).

The heavy runs of the paper need a cluster and are not part of the documentation build or of the tests. The experiment
scripts also have a smoke mode, with tiny sizes, which only checks that they run.
