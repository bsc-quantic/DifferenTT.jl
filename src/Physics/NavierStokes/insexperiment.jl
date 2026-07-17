abstract type INSExperimentBase end

"""
    struct INSExperiment{N} <: INSExperimentBase

Defines an Incompressible Navier-Stokes experiment configuration and state.

# Fields
- `grid::Grid{N}`: The spatial grid.
- `time_evol_settings::TimeEvol`: Time evolution parameters.
- `u_fields::VectorField{N}`: Velocity vector field.
- `p_field::ScalarField{N}`: Pressure scalar field.
- `Re::Number`: Reynolds number.
- `object_mask::Union{MPS, Nothing}`: Optional mask for obstacles (MPS).
- `stencil_order::Int`: Order of finite difference stencil.
- `maxdim::Int`: Maximum bond dimension allowed throughout the simulation.
- `cutoff::Float64`: SVD cutoff performed throughout the simulation.
"""
struct INSExperiment{N} <: INSExperimentBase
    grid::Grid{N}
    time_evol_settings::TimeEvol
    u_fields::VectorField{N}
    p_field::ScalarField{N}
    Re::Number
    object_mask::Union{MPS,Nothing}
    stencil_order::Int
    maxdim::Int
    cutoff::Float64

    """
        INSExperiment(grid::Grid{N}, time_evol_settings::TimeEvol, u_fields::VectorField{N}, p_field::ScalarField{N}, Re::Number, object_mask::Union{MPS, Nothing}; stencil_order::Int = 8) where N

    Construct an INSExperiment. Checks for consistency between field, grid, and mask indices.

    # Arguments
    - `grid::Grid{N}`: The spatial grid.
    - `time_evol_settings::TimeEvol`: Time evolution settings.
    - `u_fields::VectorField{N}`: Initial velocity field.
    - `p_field::ScalarField{N}`: Initial pressure field (usually zero).
    - `Re::Number`: Reynolds number.
    - `object_mask`: Optional obstacle mask.
    - `stencil_order`: Finite difference stencil order.
    - `maxdim::Int`: Maximum bond dimension allowed throughout the simulation.
    - `cutoff::Float64`: SVD cutoff performed throughout the simulation.
    """
    function INSExperiment(
        grid::Grid{N},
        time_evol_settings::TimeEvol,
        u_fields::VectorField{N},
        p_field::ScalarField{N},
        Re::Number,
        object_mask::Union{MPS,Nothing};
        stencil_order::Int=8,
        maxdim::Int=100,
        cutoff::Float64=1e-8
    ) where N

        if !isnothing(object_mask) && u_fields.usites != p_field.usites != siteinds(object_mask)
            throw(ArgumentError("Physical sites for velocity, pressure and mask fields do not match"))
        end

        check_cfl(grid, time_evol_settings, Re)
        check_kolmogorov_scale(grid, Re)

        return new{N}(
            grid, time_evol_settings,
            u_fields,
            p_field,
            Re,
            object_mask,
            stencil_order,
            maxdim,
            cutoff
        )
    end
end

"""
    check_cfl(grid::Grid{N}, time_evol_settings::TimeEvol, Re::Number)

Checks if the temporal and spatial dimensions of a given INSExperiment comply with the CFL condition for stability
"""
# TODO THIS NEEDS TO TAKE INTO ACCOUNT TOO THE VELOCITY MAGNITUDES!!!
function check_cfl(grid::Grid{N}, time_evol_settings::TimeEvol, Re::Number) where N
    dt = time_evol_settings.dt
    max_dt = typemax(Float64)
    for i in 1:N
        max_advection_dt = grid.dx[i] # Velocity is assumed to be always u ~ 1.0
        if dt > max_advection_dt
            @warn "CFL condition violated for advection term in dimension $i"
            max_dt = max_advection_dt
        end

        max_diffusion_dt = Re * grid.dx[i]^2
        if dt > max_diffusion_dt
            @warn "CFL condition violated for diffusion term in dimension $i"
            max_dt = min(max_dt, max_diffusion_dt)
        end
    end

    if dt > max_dt
        @info "CFL condition violated, must set dt<=$max_dt"
    else
        @info "CFL condition satisfied"
    end
end

function check_cfl(experiment::INSExperiment{N}) where N
    check_cfl(experiment.grid, experiment.time_evol_settings, experiment.Re)
end

function check_kolmogorov_scale(grid::Grid{N}, Re::Number) where N 
    kolmogorov_scale = Re ^ (-3/4)
    too_coarse = false
    for i in 1:N
        if grid.dx[i] > kolmogorov_scale
            too_coarse = true
            @warn "Spacing in the $(i)-th dimension does not resolve to Kolmogorov scale dx = $(grid.dx[i]) > Re^-(3/4) = $(kolmogorov_scale)"
        end
    end
    !too_coarse && @info "Kolmogorov scale is resolved"
end

function check_kolmogorov_scale(experiment::INSExperiment{N}) where N
    check_kolmogorov_scale(experiment.grid, experiment.Re)
end
