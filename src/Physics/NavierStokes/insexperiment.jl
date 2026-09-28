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

"""
    cfl_spectral_radii(experiment::INSExperiment{N}; vmax::Float64=1.0) where N

Upper bounds on the spectral radii of the discretized advection and diffusion operators of `experiment`,
built from the coefficients of its finite-difference stencils (Gershgorin bound). Contributions from all
spatial dimensions add, since the time stepper advances the full combined RHS.

The bound is exact for diffusion (the second-derivative coefficients alternate in sign, so it is attained at
the highest wavenumber) and conservative for advection.

# Returns
- `(adv_eig, diff_eig)`: bounds for the advection term (with velocity magnitude `vmax`) and the diffusion term.
"""
function cfl_spectral_radii(experiment::INSExperiment{N}; vmax::Float64=1.0) where N
    grid = experiment.grid
    s1 = sum(abs, values(derivative_coefficients(1; stencil_order=experiment.stencil_order)))  # advection
    s2 = sum(abs, values(derivative_coefficients(2; stencil_order=experiment.stencil_order)))  # diffusion

    adv_eig  = sum(vmax * s1 / grid.dx[i]                for i in 1:N)
    diff_eig = sum((1/experiment.Re) * s2 / grid.dx[i]^2 for i in 1:N)
    return adv_eig, diff_eig
end

function _report_cfl(dt, dt_max)
    if dt > dt_max
        @info "CFL condition violated, must set dt<=$dt_max"
    else
        @info "CFL condition satisfied" dt dt_max
    end
end

"""
    check_cfl(experiment::INSExperiment{N}, stepper::RK4; vmax::Float64=1.0) where N
    check_cfl(experiment::INSExperiment{N}, stepper::ExplicitEuler; vmax::Float64=1.0) where N

Checks the linear stability (CFL) condition of `experiment` for the time step stored in `stepper`, using the
stability region of the given time integration scheme and the experiment's finite-difference stencils.
`vmax` is the assumed maximum velocity magnitude of the flow.
"""
function check_cfl(experiment::INSExperiment{N}, stepper::RK4; vmax::Float64=1.0) where N
    dt = stepper.time_evol_kwargs.dt
    adv_eig, diff_eig = cfl_spectral_radii(experiment; vmax=vmax)

    # RK4 stability boundaries: |z|<=2√2 on the imaginary axis (advection is skew-symmetric),
    # |z|<=2.785 on the negative real axis (diffusion is symmetric negative-definite).
    dt_adv_max  = 2*sqrt(2) / adv_eig
    dt_diff_max = 2.785 / diff_eig
    dt_max = min(dt_adv_max, dt_diff_max)

    if dt > dt_adv_max
        @warn "CFL condition violated for advection term" dt dt_adv_max
    end
    if dt > dt_diff_max
        @warn "CFL condition violated for diffusion term" dt dt_diff_max
    end
    _report_cfl(dt, dt_max)
end

function check_cfl(experiment::INSExperiment{N}, stepper::ExplicitEuler; vmax::Float64=1.0) where N
    dt = stepper.time_evol_kwargs.dt
    _, diff_eig = cfl_spectral_radii(experiment; vmax=vmax)

    # Forward Euler is stable iff |1 + z| <= 1, a disk that only touches the imaginary axis at the origin, so
    # central-difference advection is never stable on its own: it must be damped by diffusion. On the negative
    # real axis the boundary is |z| <= 2, which gives the pure diffusion limit.
    dt_diff_max = 2 / diff_eig

    # For a Fourier mode with advection symbol a and diffusion symbol d, |1 + dt(-d + ia)| <= 1 iff
    # dt <= 2d / (d² + a²). The stencils in use satisfy κ₂(θ) >= κ₁(θ)² for the modified wavenumbers of the
    # second and first derivatives, so a² <= N vmax² Re d, and dt <= 2 / (diff_eig + N vmax² Re) is sufficient.
    dt_max = 2 / (diff_eig + N * vmax^2 * experiment.Re)

    if dt > dt_diff_max
        @warn "CFL condition violated for diffusion term" dt dt_diff_max
    end
    if dt > dt_max
        @warn "CFL condition violated for the combined advection-diffusion term" dt dt_max
    end
    _report_cfl(dt, dt_max)
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
