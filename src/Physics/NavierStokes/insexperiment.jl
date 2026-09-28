abstract type INSExperimentBase end

"""
    struct INSExperiment{N} <: INSExperimentBase

Defines an Incompressible Navier-Stokes experiment configuration and state. It describes the physical
problem only: the time integrator and the Poisson solver are passed to the solving scheme (e.g. `chorin_evolve!`).

# Fields
- `grid::Grid{N}`: The spatial grid.
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
    u_fields::VectorField{N}
    p_field::ScalarField{N}
    Re::Number
    object_mask::Union{MPS,Nothing}
    stencil_order::Int
    maxdim::Int
    cutoff::Float64

    """
        INSExperiment(grid::Grid{N}, u_fields::VectorField{N}, p_field::ScalarField{N}, Re::Number, object_mask::Union{MPS, Nothing}; stencil_order::Int = 8, maxdim::Int = 100, cutoff::Float64 = 1e-8) where N

    Construct an INSExperiment. Checks for consistency between field, grid, and mask indices.

    # Arguments
    - `grid::Grid{N}`: The spatial grid.
    - `u_fields::VectorField{N}`: Initial velocity field.
    - `p_field::ScalarField{N}`: Initial pressure field (usually zero).
    - `Re::Number`: Reynolds number.
    - `object_mask`: Optional obstacle mask.
    - `stencil_order`: Finite difference stencil order.
    - `maxdim::Int`: Maximum bond dimension for TT arithmetic in the evolution (operators, Poisson RHS, projection, time step). Not the solver bond dimension.
    - `cutoff::Float64`: SVD cutoff for that same TT arithmetic. Not the solver truncation or stopping tolerance; those live on the `solver`
    """
    function INSExperiment(
        grid::Grid{N},
        u_fields::VectorField{N},
        p_field::ScalarField{N},
        Re::Number,
        object_mask::Union{MPS,Nothing};
        stencil_order::Int=8,
        maxdim::Int=100,
        cutoff::Float64=1e-8
    ) where N

        if !isnothing(object_mask) && !(u_fields.usites == p_field.usites == siteinds(object_mask))
            throw(ArgumentError("Physical sites for velocity, pressure and mask fields do not match"))
        end

        check_kolmogorov_scale(grid, Re)

        return new{N}(
            grid,
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

# TODO THIS SHOULD ALSO TAKE INTO ACCOUNT THE VELOCITY MAGNITUDES
"""
    check_cfl(grid::Grid{N}, time_evol_kwargs::TimeEvolArgs, Re::Number)

Checks if the temporal and spatial dimensions of a given INSExperiment comply with the CFL condition for stability
"""
function check_cfl(grid::Grid{N}, time_evol_kwargs::TimeEvolArgs, Re::Number) where N
    dt = time_evol_kwargs.dt
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

"""
    check_cfl(experiment::INSExperiment{N}, time_evol_kwargs::TimeEvolArgs) where N
    check_cfl(experiment::INSExperiment{N}, stepper::AbstractTimeStepper) where N

Checks the CFL condition of `experiment` for the time step given by `time_evol_kwargs`, or by the
`TimeEvolArgs` stored in `stepper`.
"""
function check_cfl(experiment::INSExperiment{N}, time_evol_kwargs::TimeEvolArgs) where N
    check_cfl(experiment.grid, time_evol_kwargs, experiment.Re)
end

function check_cfl(experiment::INSExperiment{N}, stepper::AbstractTimeStepper) where N
    check_cfl(experiment, stepper.time_evol_kwargs)
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
