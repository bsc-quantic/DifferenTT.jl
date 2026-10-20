"""
    DifferenTTPlotsExt

Package extension of DifferenTT, loaded together with Plots.jl, that defines the plotting methods of
[`visualize_scalar`](@ref DifferenTT.visualize_scalar), [`visualize_vector`](@ref DifferenTT.visualize_vector) and
[`visualize_vorticity`](@ref DifferenTT.visualize_vorticity). The fields are sampled on a regular grid with
`DifferenTT.samples`, which contracts the MPS at every sample, so the cost grows with `npoints^N` and the plots are
meant for small figures, not for dense evaluation of large fields. Headless runs (e.g. CI) need the environment
variable `GKSwstype=100` for the GR backend.
"""
module DifferenTTPlotsExt

using DifferenTT
using DifferenTT: Architecture, Grid, ScalarField, VectorField, samples
using ITensorMPS: MPS
using Plots

import DifferenTT: visualize_scalar, visualize_vector, visualize_vorticity

function visualize_scalar(
    field::ScalarField{N},
    step::Int,
    grid::Grid{N},
    save_path::Union{String,Nothing}=nothing;
    npoints=50,
    slice_dims=(1, 2),
    slice_vals=nothing,
    kwargs...,
) where {N}
    mps = field.u[step]
    return visualize_scalar(
        mps, grid, field.arch, save_path; npoints=npoints, slice_dims=slice_dims, slice_vals=slice_vals, kwargs...
    )
end

function visualize_scalar(
    field::ScalarField{N},
    grid::Grid{N},
    save_path::Union{String,Nothing}=nothing;
    npoints=50,
    slice_dims=(1, 2),
    slice_vals=nothing,
    kwargs...,
) where {N}
    mps = last(field)
    return visualize_scalar(
        mps, grid, field.arch, save_path; npoints=npoints, slice_dims=slice_dims, slice_vals=slice_vals, kwargs...
    )
end

function visualize_scalar(
    mps::MPS,
    grid::Grid{N},
    arch::Architecture,
    save_path::Union{String,Nothing}=nothing;
    npoints=50,
    slice_dims=(1, 2),
    slice_vals=nothing,
    kwargs...,
) where {N}
    if N == 1
        ranges, data = samples(mps, grid, arch, npoints)
        plot = scatter(ranges[1], data; title="Scalar Field", xlabel="x", ylabel="y", kwargs...)
    elseif N == 2
        ranges, data = samples(mps, grid, arch, npoints)
        plot = heatmap(ranges[1], ranges[2], data'; title="Scalar Field", xlabel="x", ylabel="y", kwargs...)
    elseif N >= 3
        ranges, data = samples(mps, grid, arch, npoints)

        # Determine slice indices
        slice_indices = Any[Colon() for _ in 1:N]
        val_strs = String[]

        for d in 1:N
            if d ∉ slice_dims
                if isnothing(slice_vals) || length(slice_vals) < (d - count(x -> x < d, slice_dims))
                    slice_idx = cld(npoints, 2)
                    push!(val_strs, "mid")
                else
                    val_idx = d - count(x -> x < d, slice_dims)
                    val = slice_vals[val_idx]
                    slice_idx = argmin(abs.(ranges[d] .- val))
                    push!(val_strs, string(round(ranges[d][slice_idx]; digits=2)))
                end
                slice_indices[d] = slice_idx
            end
        end

        slice_data = data[slice_indices...]
        x_range = ranges[slice_dims[1]]
        y_range = ranges[slice_dims[2]]

        fixed_dims_str = join(["x$d ≈ $(val_strs[i])" for (i, d) in enumerate(filter(x -> x ∉ slice_dims, 1:N))], ", ")
        title_str = "Scalar Field ($fixed_dims_str)"

        plot = heatmap(x_range, y_range, slice_data'; title=title_str, kwargs...)
    else
        throw(ArgumentError("visualize_scalar needs a field with N ≥ 1 dimensions, got N = $N"))
    end
    if !isnothing(save_path)
        mkpath(dirname(save_path))
        savefig(plot, save_path)
    end
    return plot
end

function visualize_vector(
    field::VectorField{N},
    step::Int,
    grid::Grid{N},
    save_path::Union{String,Nothing}=nothing;
    npoints=20,
    quiver_npoints=20,
    arrow_scale=1.0,
    slice_dims=(1, 2),
    kwargs...,
) where {N}
    mpss = Tuple(field.components[i].u[step] for i in 1:N)
    return visualize_vector(
        mpss,
        grid,
        field.arch,
        save_path;
        npoints=npoints,
        quiver_npoints=quiver_npoints,
        arrow_scale=arrow_scale,
        slice_dims=slice_dims,
        kwargs...,
    )
end

function visualize_vector(
    field::VectorField{N},
    grid::Grid{N},
    save_path::Union{String,Nothing}=nothing;
    npoints=20,
    quiver_npoints=20,
    arrow_scale=1.0,
    slice_dims=(1, 2),
    kwargs...,
) where {N}
    mpss = Tuple(last(field.components[i]) for i in 1:N)
    return visualize_vector(
        mpss,
        grid,
        field.arch,
        save_path;
        npoints=npoints,
        quiver_npoints=quiver_npoints,
        arrow_scale=arrow_scale,
        slice_dims=slice_dims,
        kwargs...,
    )
end

function visualize_vector(
    fields_mps::NTuple{N,MPS},
    grid::Grid{N},
    arch::Architecture,
    save_path::Union{String,Nothing}=nothing;
    npoints=20,
    quiver_npoints=20,
    arrow_scale=1.0,
    slice_dims=(1, 2),
    slice_vals=nothing,
    kwargs...,
) where {N}
    if N == 2
        h_ranges, h_u_data = samples(fields_mps[1], grid, arch, npoints)
        _, h_v_data = samples(fields_mps[2], grid, arch, npoints)

        mag = sqrt.(h_u_data .^ 2 .+ h_v_data .^ 2)
        plot = heatmap(
            h_ranges[1],
            h_ranges[2],
            mag';
            title="Velocity Magnitude & Streamlines",
            xlabel="x",
            ylabel="y",
            color=:viridis,
            kwargs...,
        )

        q_ranges, q_u_data = samples(fields_mps[1], grid, arch, quiver_npoints)
        _, q_v_data = samples(fields_mps[2], grid, arch, quiver_npoints)

        X = [x for x in q_ranges[1], y in q_ranges[2]]
        Y = [y for x in q_ranges[1], y in q_ranges[2]]
        quiver!(X[:], Y[:]; quiver=(arrow_scale .* q_u_data[:], arrow_scale .* q_v_data[:]), color=:red, kwargs...)

    elseif N >= 3
        ranges, all_data = [], []
        for i in 1:N
            rngs, d = samples(fields_mps[i], grid, arch, npoints)
            if isempty(ranges)
                ranges = rngs
            end
            push!(all_data, d)
        end

        mag = sqrt.(sum(d .^ 2 for d in all_data))

        slice_indices = Any[Colon() for _ in 1:N]
        val_strs = String[]
        for d in 1:N
            if d ∉ slice_dims
                slice_idx = cld(npoints, 2)
                slice_indices[d] = slice_idx
                push!(val_strs, "mid")
            end
        end

        slice_mag = mag[slice_indices...]
        x_range = ranges[slice_dims[1]]
        y_range = ranges[slice_dims[2]]

        fixed_dims_str = join(["x$d ≈ $(val_strs[i])" for (i, d) in enumerate(filter(x -> x ∉ slice_dims, 1:N))], ", ")
        plot = heatmap(
            x_range, y_range, slice_mag'; title="Velocity Magnitude ($fixed_dims_str)", color=:viridis, kwargs...
        )

        q_ranges, q_all_data = [], []
        for i in 1:N
            q_rngs, q_d = samples(fields_mps[i], grid, arch, quiver_npoints)
            if isempty(q_ranges)
                q_ranges = q_rngs
            end
            push!(q_all_data, q_d)
        end

        q_slice_indices = Any[Colon() for _ in 1:N]
        for d in 1:N
            if d ∉ slice_dims
                q_slice_indices[d] = cld(quiver_npoints, 2)
            end
        end

        q_slice_u = q_all_data[slice_dims[1]][q_slice_indices...]
        q_slice_v = q_all_data[slice_dims[2]][q_slice_indices...]

        X = [x for x in q_ranges[slice_dims[1]], y in q_ranges[slice_dims[2]]]
        Y = [y for x in q_ranges[slice_dims[1]], y in q_ranges[slice_dims[2]]]

        quiver!(X[:], Y[:]; quiver=(arrow_scale .* q_slice_u[:], arrow_scale .* q_slice_v[:]), color=:red, kwargs...)
    else
        throw(ArgumentError("visualize_vector needs a field with N ≥ 2 components, got N = $N"))
    end
    if !isnothing(save_path)
        mkpath(dirname(save_path))
        savefig(plot, save_path)
    end
    return plot
end

function visualize_vorticity(
    field::VectorField{N},
    step::Int,
    grid::Grid{N},
    save_path::Union{String,Nothing}=nothing;
    npoints=50,
    quiver_npoints=20,
    arrow_scale=1.0,
    slice_dims=(1, 2),
    kwargs...,
) where {N}
    mpss = Tuple(field.components[i].u[step] for i in 1:N)
    return visualize_vorticity(
        mpss,
        grid,
        field.arch,
        save_path;
        npoints=npoints,
        quiver_npoints=quiver_npoints,
        arrow_scale=arrow_scale,
        slice_dims=slice_dims,
        kwargs...,
    )
end

function visualize_vorticity(
    field::VectorField{N},
    grid::Grid{N},
    save_path::Union{String,Nothing}=nothing;
    npoints=50,
    quiver_npoints=20,
    arrow_scale=1.0,
    slice_dims=(1, 2),
    kwargs...,
) where {N}
    mpss = Tuple(last(field.components[i]) for i in 1:N)
    return visualize_vorticity(
        mpss,
        grid,
        field.arch,
        save_path;
        npoints=npoints,
        quiver_npoints=quiver_npoints,
        arrow_scale=arrow_scale,
        slice_dims=slice_dims,
        kwargs...,
    )
end

function visualize_vorticity(
    fields_mps::NTuple{N,MPS},
    grid::Grid{N},
    arch::Architecture,
    save_path::Union{String,Nothing}=nothing;
    npoints=50,
    quiver_npoints=20,
    arrow_scale=1.0,
    slice_dims=(1, 2),
    kwargs...,
) where {N}
    # The vorticity is defined for two components (scalar) and three (vector); sampling comes after this check
    N in (2, 3) || throw(ArgumentError("visualize_vorticity needs a velocity field with 2 or 3 components, got N = $N"))
    ranges, all_data = [], []
    for i in 1:N
        rngs, d = samples(fields_mps[i], grid, arch, npoints)
        if isempty(ranges)
            ranges = rngs
        end
        push!(all_data, d)
    end

    d_steps = [grid.box_size[i] / (npoints - 1) for i in 1:N]

    if N == 2
        u1 = all_data[1]
        u2 = all_data[2]
        dx, dy = d_steps[1], d_steps[2]

        vort = zeros(npoints, npoints)
        for i in 2:(npoints-1)
            for j in 2:(npoints-1)
                dv_dx = (u2[i+1, j] - u2[i-1, j]) / (2 * dx)
                du_dy = (u1[i, j+1] - u1[i, j-1]) / (2 * dy)
                vort[i, j] = dv_dx - du_dy
            end
        end

        plot = heatmap(
            ranges[1], ranges[2], vort'; title="Vorticity Magnitude & Streamlines", xlabel="x", ylabel="y", kwargs...
        )

        q_ranges, q_u_data = samples(fields_mps[1], grid, arch, quiver_npoints)
        _, q_v_data = samples(fields_mps[2], grid, arch, quiver_npoints)

        X = [x for x in q_ranges[1], y in q_ranges[2]]
        Y = [y for x in q_ranges[1], y in q_ranges[2]]
        quiver!(X[:], Y[:]; quiver=(arrow_scale .* q_u_data[:], arrow_scale .* q_v_data[:]), color=:white, kwargs...)

    elseif N >= 3
        u1, u2, u3 = all_data[1], all_data[2], all_data[3]
        dx, dy, dz = d_steps[1], d_steps[2], d_steps[3]

        vort_mag = zeros(npoints, npoints, npoints)
        for i in 2:(npoints-1)
            for j in 2:(npoints-1)
                for k in 2:(npoints-1)
                    dw_dy = (u3[i, j+1, k] - u3[i, j-1, k]) / (2 * dy)
                    dv_dz = (u2[i, j, k+1] - u2[i, j, k-1]) / (2 * dz)
                    wx = dw_dy - dv_dz

                    du_dz = (u1[i, j, k+1] - u1[i, j, k-1]) / (2 * dz)
                    dw_dx = (u3[i+1, j, k] - u3[i-1, j, k]) / (2 * dx)
                    wy = du_dz - dw_dx

                    dv_dx = (u2[i+1, j, k] - u2[i-1, j, k]) / (2 * dx)
                    du_dy = (u1[i, j+1, k] - u1[i, j-1, k]) / (2 * dy)
                    wz = dv_dx - du_dy

                    vort_mag[i, j, k] = sqrt(wx^2 + wy^2 + wz^2)
                end
            end
        end

        slice_idx = cld(npoints, 2)
        slice_indices = Any[Colon() for _ in 1:N]
        for d in 1:N
            if d ∉ slice_dims
                slice_indices[d] = slice_idx
            end
        end

        slice_data = vort_mag[slice_indices[1:3]...]

        plot = heatmap(
            ranges[slice_dims[1]],
            ranges[slice_dims[2]],
            slice_data';
            title="Vorticity Magnitude (slice) & Streamlines",
            xlabel="x",
            ylabel="y",
            kwargs...,
        )

        q_ranges, q_all_data = [], []
        for i in 1:N
            q_rngs, q_d = samples(fields_mps[i], grid, arch, quiver_npoints)
            if isempty(q_ranges)
                q_ranges = q_rngs
            end
            push!(q_all_data, q_d)
        end

        q_slice_indices = Any[Colon() for _ in 1:N]
        for d in 1:N
            if d ∉ slice_dims
                q_slice_indices[d] = cld(quiver_npoints, 2)
            end
        end

        q_slice_u = q_all_data[slice_dims[1]][q_slice_indices...]
        q_slice_v = q_all_data[slice_dims[2]][q_slice_indices...]

        X = [x for x in q_ranges[slice_dims[1]], y in q_ranges[slice_dims[2]]]
        Y = [y for x in q_ranges[slice_dims[1]], y in q_ranges[slice_dims[2]]]

        quiver!(X[:], Y[:]; quiver=(arrow_scale .* q_slice_u[:], arrow_scale .* q_slice_v[:]), color=:white, kwargs...)
    else
        throw(ArgumentError("visualize_vorticity needs a field with N ≥ 2 components, got N = $N"))
    end
    if !isnothing(save_path)
        mkpath(dirname(save_path))
        savefig(plot, save_path)
    end
    return plot
end

end
