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

TODO(release): a short example that builds a Poisson problem, preconditions it with FAI and solves it with MALS.

## Citing

TODO(release): citation of the FAI preprint and of the code (`CITATION.cff`).

## License

DifferenTT is released under the MIT license; see [`LICENSE`](LICENSE).
