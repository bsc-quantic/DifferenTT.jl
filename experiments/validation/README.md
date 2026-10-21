# Validation

These scripts check the Navier–Stokes solvers of DifferenTT against references and run flows that are not part of the
benchmarks. We keep them apart from `fai_benchmarks/` because they measure the accuracy of a flow solution, not the
speed of a preconditioner.

| Folder | What it does | Reference |
|---|---|---|
| `liddriven_cavity` | Long evolution of the lid-driven cavity with the vorticity–streamfunction scheme (`INSVorticityExperiment`, walls as ghost nodes): `worker.jl` runs it, `plot_snapshots.jl` draws the fields and `ghia_comparison.jl` compares the centerline velocities and the primary vortex at Re = 1000 | Ghia, Ghia & Shin (1982); the setup follows Kiffner & Jaksch (arXiv:2303.03010) |
| `tunnel2d` | Flow past an obstacle (a smoothed circle mask) in a channel with the Chorin projection scheme, in the QTT format | Physical setup of Sec. 5.2 of the paper |
| `tunnel2dsteps` | The same flow, written as the individual stages of the Chorin step with a picture after each stage, to inspect advection, diffusion, divergence and pressure correction separately | – |

`tunnel2d` and `tunnel2dsteps` are exploratory scripts that still need to be brought up to date: both load `Revise` and
`MKL`, which the environment of `experiments/` does not provide, and they write their output next to the script. This
update is pending (`TODO(5a-4)`), and `examples/` will carry a didactic version of `tunnel2d`.

## How to run

The cavity follows the worker–`submit.sh` sequence of `experiments/README.md`: from the root of the repository,

```bash
bash experiments/validation/liddriven_cavity/submit.sh nocond
```

`CAVITY_LIDS=2` selects the doubly-driven cavity (the Ghia comparison exists only for the single lid). The tunnel scripts
are submitted with `sbatch experiments/validation/tunnel2d/tunnel2d.slurm` (8 CPUs, 6 h). Smoke mode: `TODO(5a-2)`.
