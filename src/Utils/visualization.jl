# The plotting methods live in the package extension `ext/DifferenTTPlotsExt.jl`, which Julia loads together with
# Plots.jl. Only the generic functions and their documentation are defined here.

"""
    visualize_scalar(mps::MPS, grid::Grid{N}, arch::Architecture, save_path=nothing; npoints=50, slice_dims=(1, 2),
                     slice_vals=nothing, kwargs...)
    visualize_scalar(field::ScalarField{N}, grid::Grid{N}, save_path=nothing; npoints=50, slice_dims=(1, 2),
                     slice_vals=nothing, kwargs...)
    visualize_scalar(field::ScalarField{N}, step::Int, grid::Grid{N}, save_path=nothing; npoints=50, slice_dims=(1, 2),
                     slice_vals=nothing, kwargs...)

Plot a scalar field sampled on `npoints` points per dimension of `grid`.

In one dimension we draw a scatter plot, and in two dimensions a heatmap. For ``N \\geq 3`` we draw a heatmap of the
two-dimensional slice spanned by `slice_dims`, with every other coordinate fixed at the sample nearest to the
corresponding entry of `slice_vals`, or at the middle sample if `slice_vals` is `nothing`. The `ScalarField` methods
plot the last stored time step, or the time step `step`. If `save_path` is a `String`, the figure is also saved there,
creating its directory if needed. The remaining keywords are passed to Plots.jl.

These methods are defined in the package extension `DifferenTTPlotsExt` and require `using Plots`; without it, a
`MethodError` with a hint is thrown.

# Returns

  - `Plots.Plot`: the figure.

# Examples

```julia
using Plots
visualize_scalar(experiment.p_field, experiment.grid, "figures/pressure.png"; npoints=128)
```
"""
function visualize_scalar end

"""
    visualize_vector(mpss::NTuple{N,MPS}, grid::Grid{N}, arch::Architecture, save_path=nothing; npoints=20,
                     quiver_npoints=20, arrow_scale=1.0, slice_dims=(1, 2), slice_vals=nothing, kwargs...)
    visualize_vector(field::VectorField{N}, grid::Grid{N}, save_path=nothing; npoints=20, quiver_npoints=20,
                     arrow_scale=1.0, slice_dims=(1, 2), kwargs...)
    visualize_vector(field::VectorField{N}, step::Int, grid::Grid{N}, save_path=nothing; npoints=20, quiver_npoints=20,
                     arrow_scale=1.0, slice_dims=(1, 2), kwargs...)

Plot the magnitude of a vector field as a heatmap on `npoints` points per dimension, overlaid with a quiver plot of
its components on `quiver_npoints` points per dimension, scaled by `arrow_scale`.

For ``N \\geq 3`` we plot the two-dimensional slice spanned by `slice_dims`, with every other coordinate fixed at the
middle sample; `slice_vals` is currently ignored. The `VectorField` methods plot the last stored time step, or the
time step `step`. If `save_path` is a `String`, the figure is also saved there. The remaining keywords are passed to
Plots.jl.

These methods are defined in the package extension `DifferenTTPlotsExt` and require `using Plots`. One-dimensional
fields throw an `ArgumentError`.

# Returns

  - `Plots.Plot`: the figure.
"""
function visualize_vector end

"""
    visualize_vorticity(mpss::NTuple{N,MPS}, grid::Grid{N}, arch::Architecture, save_path=nothing; npoints=50,
                        quiver_npoints=20, arrow_scale=1.0, slice_dims=(1, 2), kwargs...)
    visualize_vorticity(field::VectorField{N}, grid::Grid{N}, save_path=nothing; npoints=50, quiver_npoints=20,
                        arrow_scale=1.0, slice_dims=(1, 2), kwargs...)
    visualize_vorticity(field::VectorField{N}, step::Int, grid::Grid{N}, save_path=nothing; npoints=50,
                        quiver_npoints=20, arrow_scale=1.0, slice_dims=(1, 2), kwargs...)

Plot the vorticity of a velocity field as a heatmap, overlaid with a quiver plot of the velocity.

We sample the velocity on `npoints` points per dimension and evaluate the vorticity on the interior samples by
second-order central differences, with spacing `box_size[i] / (npoints - 1)`; the boundary samples are set to zero.
In two dimensions we plot the scalar vorticity ``\\partial_x v - \\partial_y u``, and for ``N \\geq 3`` the magnitude
of the vorticity of the first three components, on the slice spanned by `slice_dims` through the middle sample. The
`VectorField` methods plot the last stored time step, or the time step `step`. If `save_path` is a `String`, the
figure is also saved there. The remaining keywords are passed to Plots.jl.

These methods are defined in the package extension `DifferenTTPlotsExt` and require `using Plots`. One-dimensional
fields throw an `ArgumentError`, and for ``N \\geq 4`` only the first three components and coordinates enter the
vorticity, which is then not meaningful.

# Returns

  - `Plots.Plot`: the figure.
"""
function visualize_vorticity end

"""
    _register_plots_error_hint() -> nothing

Register a `MethodError` hint that tells users to load Plots.jl before calling the `visualize_*` functions; it is
called when the module is initialized.
"""
function _register_plots_error_hint()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f in (visualize_scalar, visualize_vector, visualize_vorticity) &&
            isnothing(Base.get_extension(@__MODULE__, :DifferenTTPlotsExt))
            print(
                io, "\n`$(nameof(exc.f))` is defined in the Plots extension of DifferenTT; load it with `using Plots`."
            )
        end
    end
    return nothing
end
