```@meta
CurrentModule = DifferenTT
DocTestSetup = :(using DifferenTT, ITensors, ITensorMPS)
```

# Architectures and conventions

A tensor train stores a function on an ``N``-dimensional grid as a chain of sites, and every other object of the library
(operators, boundary corrections, solvers) must agree with the user on which grid index each site carries. In order to
fix this once for the whole library, we describe the layout of the chain by an *architecture*, and we state here the
conventions that every function follows. The signatures are in the [API reference](@ref api_tensortrains).

## The two architectures

A [`TTArchitecture`](@ref) has one site per variable. The site that carries variable ``i`` has dimension ``n_i``, the number of
grid points along that axis, so that a ``n_1 \times \dots \times n_N`` grid takes ``N`` sites of possibly different
dimensions. This is the layout for problems with many dimensions and a moderate number of points per dimension.

A [`QTTArchitecture`](@ref) has one site of dimension ``d = 2`` per bit of each variable. The index of a variable with
``2^n`` points is written in binary, ``i = \sum_{p=1}^{n} i_p 2^{n-p}``, and each bit ``i_p`` takes a site (Sec. 2 of
Torrente et al., *Factorized approximate inverse preconditioners in the tensor train format*, which we call "the
paper"). The grid then takes ``n_1 + \dots + n_N`` sites, and a field that is smooth along an axis has a bond dimension
that grows only slowly with ``n``. This is the layout for fine grids in a few dimensions.

```jldoctest
julia> Tuple(QTTArchitecture((3, 2)).site_map)
(1, 1, 1, 2, 2)

julia> Tuple(QTTArchitecture((3, 2); interleaved=true).site_map)
(1, 2, 1, 2, 1)

julia> TTArchitecture((5, 3)).var_dims
(5, 3)
```

A QTT architecture can be *blocked*, with the sites of each variable contiguous, or *interleaved*, taking one site of
each variable in turn. The optional `var_map` permutes the variables along the chain, and a negative entry of the number
of sites reverses the order of the bits of that variable. The functions [`total_sites`](@ref),
[`nsites_per_variable`](@ref), [`site_dim`](@ref), [`site_dims`](@ref) and [`variable_dims`](@ref) query a layout.

## Site indices

Fields and operators live on a list of ITensors indices, one per site. `siteinds(arch)` creates a fresh list for an
architecture, and the constructors of fields create one for themselves, which `field.usites` returns. **A field, an
operator and a right-hand side can only be combined if they were built on the same indices.** A tensor train that comes
from elsewhere, such as the result of cross interpolation, is moved onto the indices of the problem with
`replace_siteinds!`.

## Conventions

  - **Site 1 is the most significant.** [`contract_mps`](@ref) and [`contract_mpo`](@ref) return dense arrays in which
    site 1 is the most significant index, so that `kron(A₁, A₂, …)` corresponds to sites 1, 2, …. In a blocked QTT
    layout, the first bit of a variable is its most significant bit. The helpers [`vec2mps`](@ref), [`vec2tensor`](@ref),
    [`mat2mpo`](@ref) and [`mat2tensor`](@ref) use the same order.
  - **Prime levels of an MPO.** An MPO acts from sites at prime level 0 (input) to sites at prime level 1 (output).
  - **Array backend.** `array_type` (default `Array`) selects only the array backend of the tensors, such as `CuArray`
    for a GPU, and not their element type ([GPU (experimental)](@ref)).
  - **Truncation.** `cutoff` is a relative truncation of the singular values, and `maxdim` a cap on the bond dimension.
    Library functions that truncate accept both, and document their defaults.
  - **Randomness.** Every randomized routine accepts `rng::AbstractRNG`, which defaults to `Random.default_rng()`. Pass
    a seeded generator to reproduce a result.
  - **Output.** The library prints nothing by default. Informational output is controlled by `outputlevel` and
    `verbosity`, and diagnostics use `@debug`.
