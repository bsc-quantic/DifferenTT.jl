# DifferenTT.jl

[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://bsc-quantic.github.io/DifferenTT.jl/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://bsc-quantic.github.io/DifferenTT.jl/dev/)
[![Build Status](https://github.com/bsc-quantic/DifferenTT.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/bsc-quantic/DifferenTT.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/bsc-quantic/DifferenTT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/bsc-quantic/DifferenTT.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

DifferenTT is a Julia library for solving partial differential equations in the tensor train (TT) format, built on
[ITensors.jl](https://github.com/ITensor/ITensors.jl) and
[ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl). We represent fields as MPS and finite-difference operators
as tensor train operators (MPO), either on a TT grid, with one site per spatial dimension, or on a quantized TT (QTT)
grid, with one site of dimension 2 per bit of each coordinate. On top of this representation, the library provides:

- Grids, scalar and vector fields, and Dirichlet, Neumann and periodic boundary conditions on collocated grids.
- Finite-difference operators of arbitrary stencil order, assembled directly in the TT format.
- Linear solvers for the resulting systems: ALS, residual-based MALS, AMEn, and global TT-CG and TT-GMRES.
- Preconditioners for these solvers, including the factorized approximate inverse (FAI) preconditioner in the TT
  format, which is positive definite by construction, and a sinc-quadrature approximation of the inverse Laplacian.
- Time steppers (explicit Euler and RK4) and two incompressible Navier–Stokes solvers, by Chorin projection and by
  the vorticity–streamfunction formulation.

## Installation

DifferenTT is not yet registered in the Julia General registry. It requires Julia 1.12 or later, and is installed from
this repository with

```julia
pkg> add https://github.com/bsc-quantic/DifferenTT.jl
```

The first installation precompiles the common paths of the library, which takes about two minutes once and makes the
first calls of every later session 3 to 5 times faster; `DifferenTT.set_precompile_workload(false)` turns this off,
e.g. while developing the package.

The plotting functions `visualize_scalar`, `visualize_vector` and `visualize_vorticity` are defined in a package
extension, and become available after `using Plots`.

## Quick start

We solve a two-dimensional Poisson problem in the QTT format with MALS (`ResidualALS`), preconditioned by the FAI
preconditioner. The grid has 2⁵ × 2⁵ nodes on [0, 2] × [0, 1], with a Neumann and a Dirichlet side in x and periodic
sides in y, as for the pressure of a channel flow, and the stencil order is q = 8. The same MALS solver, with bond
dimension χ_G = 4, builds the factor of the preconditioner; we take μ = 10⁴, close to the top of the spectrum of A
(about 8 · 10³ on this grid), as the manual recommends.

```julia
using DifferenTT, ITensors, ITensorMPS
using Random

# Poisson problem -Δp = b on [0, 2] × [0, 1] with 2⁵ × 2⁵ nodes in QTT format: Neumann (left) and Dirichlet
# (right) sides in x, periodic in y, as for the pressure of a channel flow
bcs = BoundaryConditions(
    BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), BoundaryCond1d(PeriodicBC(), PeriodicBC())
)
arch = QTTArchitecture((5, 5))
sites = siteinds(arch)
grid = Grid(arch, (2.0, 1.0); periodic=periodic_axes(bcs))
A = -1.0 * symmetrized_laplacian(ScalarField(sites, arch, bcs), grid; stencil_order=8)  # SPD
b = random_mps(Xoshiro(2), sites; linkdims=4)
b /= norm(b)

# MALS (ResidualALS) with local CG solves, used both to build the FAI preconditioner and to solve the system
mals(χ, rtol; nsweeps) = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps, maxdim=χ, cutoff=0.0),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol, krylovdim=1, maxiter=3000),
    residual_tol=1e-6,
)
M = fai_preconditioner(A; μ=1e4, solver=mals(4, 1e-8; nsweeps=30), totalmaxdim=4, residual_tol=0.1, rng=Xoshiro(1))
x = linearsolve(mals(64, 1e-7; nsweeps=20), A, b, M, nothing)
println("relative residual ", norm(apply(A, x) - b))
```

On our machine (Julia 1.12.3, one thread) the script takes about 5 s once the package is loaded, and prints a relative
residual of 1.3 · 10⁻⁹. Tutorials [01][tutorial-01] and [04][tutorial-04] explain these steps, and the latter compares
the preconditioned solve with the unpreconditioned one.

## Documentation

The [documentation][docs-stable] contains a getting-started guide, a manual (architectures and conventions, grids,
fields and boundary conditions, operators, linear solvers, preconditioners, Navier–Stokes), nine tutorials, among them
two on the discretization of boundary conditions, and the API reference. The [development version][docs-dev] follows the
`master` branch.

## Known limitations

We list here the limitations of v0.1 that can change a result; they are also stated in the docstrings concerned, the
[release notes](NEWS.md) and the [Navier–Stokes manual page][docs-ns], and each has an issue.

- **Chorin projection.** When the pressure has no Dirichlet side, the mean removal of its singular Poisson system
  subtracts the wrong constant ([#49]), and the projection also changes the velocity on the rows next to a Dirichlet
  velocity face ([#50]). Both are recorded as `@test_broken` in the test suite. Moreover, `chorin_step` returns the
  provisional velocity of the next step rather than the projected one ([#53]).
- **Dirichlet closure.** With `stencil_order = 8`, the boundary closure of a Dirichlet side, and with it the solution,
  is only first-order accurate; it is exact for `stencil_order = 2` ([#51]).
- **Periodic axes.** A `Grid` built without the `periodic` flag gives the stencils of a periodic axis the period L + h
  instead of L; build it with `Grid(arch, box_size; periodic = periodic_axes(bcs))` ([#52]).

The other known issues of v0.1, among them missing options, performance and upstream behavior, are listed in the
[issue tracker][issues]; those planned for the next version are in the [v0.2 milestone][milestone-v0.2].

**GPU support is experimental.** The keyword `array_type` selects the array backend of the tensors (for instance
`CuArray` from CUDA.jl), and we checked by reading the code that the library keeps every tensor on the backend of its
inputs, but no test runs on a GPU yet ([#73]). The [GPU page][docs-gpu] of the manual states what is covered.

## Citing

If you use DifferenTT in your work, please cite the paper that introduces the FAI preconditioner in the TT format:

```bibtex
@misc{torrente_fai_tt,
  title  = {Factorized approximate inverse preconditioners in the tensor train format},
  author = {Torrente Badia, Pau and P{\'e}rez Garc{\'i}a, Daniel and Carignano, Stefano and Garc{\'i}a S{\'a}ez, Artur},
  note   = {In preparation},
}
```

TODO(release): the arXiv identifier of the preprint (`eprint`, `archivePrefix = {arXiv}`) and its year.

To cite the code itself, use the metadata of [`CITATION.cff`](CITATION.cff) (GitHub shows it as "Cite this repository").

The scripts that produce the figures and tables of the paper are in [`experiments/paper/`](experiments/paper/), with the
instructions to rerun them. The data of the paper were produced at the tag
[`paper-ttfai-v0.1-data`](https://github.com/bsc-quantic/DifferenTT.jl/tree/paper-ttfai-v0.1-data) and are archived on
Zenodo: [doi:10.5281/zenodo.23663326](https://doi.org/10.5281/zenodo.23663326).

## License

DifferenTT is released under the MIT license; see [`LICENSE`](LICENSE).

[tutorial-01]: https://bsc-quantic.github.io/DifferenTT.jl/stable/tutorials/01_qtt_basics/
[tutorial-04]: https://bsc-quantic.github.io/DifferenTT.jl/stable/tutorials/04_fai_preconditioner/
[docs-stable]: https://bsc-quantic.github.io/DifferenTT.jl/stable/
[docs-dev]: https://bsc-quantic.github.io/DifferenTT.jl/dev/
[docs-ns]: https://bsc-quantic.github.io/DifferenTT.jl/stable/manual/navier_stokes/
[docs-gpu]: https://bsc-quantic.github.io/DifferenTT.jl/stable/manual/gpu/
[issues]: https://github.com/bsc-quantic/DifferenTT.jl/issues
[milestone-v0.2]: https://github.com/bsc-quantic/DifferenTT.jl/milestone/1
[#49]: https://github.com/bsc-quantic/DifferenTT.jl/issues/49
[#50]: https://github.com/bsc-quantic/DifferenTT.jl/issues/50
[#51]: https://github.com/bsc-quantic/DifferenTT.jl/issues/51
[#52]: https://github.com/bsc-quantic/DifferenTT.jl/issues/52
[#53]: https://github.com/bsc-quantic/DifferenTT.jl/issues/53
[#73]: https://github.com/bsc-quantic/DifferenTT.jl/issues/73
