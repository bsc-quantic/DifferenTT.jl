```@meta
CurrentModule = DifferenTT
```

# Tensor-train utilities

Three operations are needed again and again once a field and an operator are in the TT format: building a tensor train
from a function, multiplying two tensor trains element by element, and applying a nonlinear function to every entry.
None of them can be done core by core with an exact result, and none of them should form the dense vector, whose size is
exponential in the number of sites. This page explains how DifferenTT performs each of them, and what accuracy to expect.
The signatures are in the [API reference](@ref api_tensortrains).

## Constructors and conversions

[`identity_mpo`](@ref), [`zeros_mpo`](@ref), [`ones_mpo`](@ref), [`zeros_mps`](@ref) and [`ones_mps`](@ref) build the
elementary tensor trains for an architecture or a list of site indices. [`vec2mps`](@ref) and [`mat2mpo`](@ref) compress
a dense vector or matrix by successive SVDs, and [`contract_mps`](@ref) and [`contract_mpo`](@ref) go the other way.
All of these are meant for small systems and for tests, since the dense array has ``d^N`` entries. The Hadamard-type
helpers [`diagmpo`](@ref) and [`diagmps`](@ref) convert between a vector and the diagonal operator that has it on its
diagonal, which is what the FAI construction uses ([Preconditioners](@ref)).

## Functions by cross interpolation

Cross interpolation builds the TT format of a function by evaluating it only at a small set of entries that it selects,
so it applies to grids far too large to be stored densely, provided that the function has low rank in the chosen layout.
DifferenTT extends `TensorCrossInterpolation.crossinterpolate2` with methods that take a [`Grid`](@ref) and an
architecture, and maps every requested multi-index to the coordinates of its node:

```julia
f(x) = exp(-20 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2))
tci, ranks, errors = crossinterpolate2(Float64, f, grid, arch; tolerance=1e-8)
b = replace_siteinds!(MPS(tci), sites)
```

The algorithm starts from `initialpivots`, by default the node at the origin, so a function that vanishes there needs
another pivot, for instance from `optfirstpivot`. The one-site variant `crossinterpolate1` has the same interface; it is kept to
compare the two methods, and in our tests it was much less accurate than the two-site one. Once a function is in the TT
format, [`sample`](@ref) evaluates it at a point, [`samples`](@ref) at a regular set of points, and
[`integrate`](@ref) computes the integral over the box.

## Element-wise products

The Hadamard product of two tensor trains of bond dimensions ``\chi_1`` and ``\chi_2`` has bond dimension ``\chi_1
\chi_2`` if computed exactly, although the result usually needs far less. [`tt_mul`](@ref) computes it for two MPS by
randomized sketch-and-interpolate (RSI) (Meng, Khoo, Li & Stoudenmire, arXiv:2602.17974), which is the default, and
falls back to the exact product if the sketch fails; [`tt_mul_fallbacks`](@ref) counts the fallbacks. This default is
accurate only if the rank of the product fits in the sketch, and the direct product was faster in some of our tests with
factors of low bond dimension, so both algorithms are kept (`_direct_tt_mul` is the exact one). The products that involve
an MPO are exact followed by a truncation.

## Element-wise functions

[`elementwise_function`](@ref) returns an MPS for ``f(x_i)`` given an MPS for ``x_i``, by the same sketching idea applied
to the entries of the cores (Sec. 3.2 of the paper). It is the tool that computes the scaling ``D =
[\mathrm{diag}(\tilde{G}_L)]^{-1/2}`` of the FAI. For functions defined only on positive arguments, we draw positive
sketches; the result is approximate, with an accuracy set by the sketch size (`oversample`). Pass an `rng` to make a run
reproducible.
