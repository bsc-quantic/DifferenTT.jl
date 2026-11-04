```@meta
CurrentModule = DifferenTT
```

# DifferenTT.jl

DifferenTT is a Julia library for solving partial differential equations in the tensor train (TT) format, built on
[ITensors.jl](https://github.com/ITensor/ITensors.jl) and
[ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl). The cost of a grid-based solver grows exponentially with the
dimension and, for fine grids, polynomially with the number of points per dimension. A field that is smooth enough has
a low-rank structure, though, and the TT format exploits it: we store a field on ``n^N`` points as a matrix product
state (MPS) whose number of parameters grows only linearly with ``N`` for a fixed bond dimension ``\chi``, and we store
a finite-difference operator as a tensor train operator (MPO). The library is a workbench for TT methods, so it keeps
several algorithms for the same task side by side and documents what each one is for.

## Features

  - **Grids and layouts.** A TT grid has one site of dimension ``n_i`` per spatial dimension, and a quantized TT (QTT)
    grid has one site of dimension 2 per bit of each coordinate, so that ``2^{n_1} \times \dots \times 2^{n_N}`` points
    take ``n_1 + \dots + n_N`` sites ([Architectures and conventions](@ref)).
  - **Fields and boundary conditions.** Scalar and vector fields on collocated grids, with periodic, open, Dirichlet and
    Neumann conditions ([Grids, fields and boundary conditions](@ref)).
  - **Finite-difference operators.** Stencils of order 2 and 8, with their boundary closures, assembled directly in the
    TT format ([Finite-difference operators](@ref)).
  - **Tensor-train utilities.** Element-wise products and functions by randomized sketching, and cross interpolation of
    functions ([Tensor-train utilities](@ref)).
  - **Linear solvers.** The alternating solvers ALS, ResidualALS (MALS with a residual-based truncation), AMEn and
    ExpandedALS, and global TT-CG and TT-GMRES ([Linear solvers](@ref manual_linear_solvers)).
  - **Preconditioners.** The factorized approximate inverse (FAI) in the TT format, which is positive definite by
    construction, and a sinc-quadrature approximation of the inverse Laplacian ([Preconditioners](@ref)).
  - **Navier–Stokes.** Explicit Euler and RK4 time steppers, and two incompressible solvers: the Chorin projection and
    the vorticity–streamfunction formulation ([Navier–Stokes](@ref)).

## Installation

DifferenTT is not yet registered in the Julia General registry. It requires Julia 1.12 or later, and is installed from
its repository:

```julia
using Pkg
Pkg.add(url="https://github.com/bsc-quantic/DifferenTT.jl")
```

The plotting functions [`visualize_scalar`](@ref), [`visualize_vector`](@ref) and [`visualize_vorticity`](@ref) live in
a package extension and become available after `using Plots`.

## A first example

We solve the Poisson problem ``-\Delta u = f`` with homogeneous Dirichlet conditions on a ``2^5 \times 2^5`` grid, in
the QTT format. The right-hand side is a Gaussian, which we compress by cross interpolation, and the solver is
ResidualALS. [Getting started](@ref) explains every line.

```julia
using DifferenTT, ITensors, ITensorMPS, TensorCrossInterpolation, LinearAlgebra

arch = QTTArchitecture((5, 5))                     # 2⁵ × 2⁵ points, one bit per site
grid = Grid(arch, (1.0, 1.0))
bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
field = ScalarField(arch, BoundaryConditions(bc, bc))

A = -1.0 * symmetrized_laplacian(field, grid; stencil_order=2)   # SPD operator, an MPO
f(x) = exp(-20 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2))
tci, = crossinterpolate2(Float64, f, grid, arch; tolerance=1e-8)
b = replace_siteinds!(MPS(tci), field.usites)                    # right-hand side, an MPS

solver = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=20, cutoff=0.0, (observer!)=ResidualObserver(; rtol=1e-8)),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-10),
)
x = linearsolve(solver, A, b, nothing)
norm(apply(A, x) - b) / norm(b)                                  # relative residual
```

## Where to go next

The [Manual](@ref "Architectures and conventions") describes the objects of the library and the conventions that all of
its functions share, and links to the [API reference](@ref api_tensortrains) for the signature of every function. The
[Tutorials](@ref tutorials_overview) are runnable scripts that build each part of the library step by step. The preconditioner
of the accompanying paper is described in [Preconditioners](@ref), and the scripts that produced the paper's figures
are in [Reproducing the paper](@ref).

The module docstring, which `?DifferenTT` prints in the REPL, summarizes the same:

```@docs
DifferenTT
```

## Citing

If you use DifferenTT in your work, please cite the paper that introduces the FAI preconditioner, P. Torrente et al.,
*Factorized approximate inverse preconditioners in the tensor train format*,
[arXiv:2611.03859](https://arxiv.org/abs/2611.03859), and the code itself. The file
[`CITATION.cff`](https://github.com/bsc-quantic/DifferenTT.jl/blob/master/CITATION.cff) of the repository holds the
metadata of both, and the README gives a BibTeX entry for the paper. The scripts that reproduce the paper's figures and
tables are described in [Reproducing the paper](@ref).

## License

DifferenTT is released under the MIT license.
