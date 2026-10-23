"""
    abstract type INSExperimentBase

Supertype of the incompressible Navier–Stokes experiments, [`INSExperiment`](@ref) (velocity–pressure, Chorin
projection) and [`INSVorticityExperiment`](@ref) (vorticity–streamfunction). Its subtypes have the fields `grid`, `Re`
and `stencil_order`, which [`cfl_spectral_radii`](@ref), [`check_cfl`](@ref) and [`check_kolmogorov_scale`](@ref) read.
"""
abstract type INSExperimentBase end

"""
    INSExperiment(grid::Grid{N}, u_fields::VectorField{N}, p_field::ScalarField{N}, Re::Number,
                  object_mask::Union{MPS,Nothing}; stencil_order::Int=8, maxdim::Int=100, cutoff::Float64=1e-8) where N

Incompressible Navier–Stokes experiment in the velocity–pressure formulation, solved by the Chorin projection.

The experiment describes the physical problem and its state: the grid, the velocity and pressure fields with their
boundary conditions, the Reynolds number ``Re``, an optional obstacle mask, and the stencil order and TT truncation of
the evolution. The time integrator and the Poisson solver are passed to the scheme, [`chorin_evolve!`](@ref) or
[`chorin_step`](@ref), which follows the QTT pipeline of Peddinti et al. (2024) as described in Sec. 5.2 of Torrente et
al., *Factorized approximate inverse preconditioners in the tensor train format*. The fields live on a collocated grid
([`Grid`](@ref), with nodes at both ends), the obstacle enters by multiplying the velocity by the mask at every step,
and the pressure system uses the compact Laplacian of [`symmetrized_laplacian`](@ref) rather than the composition of the
discrete divergence and gradient, so the projected velocity is only approximately divergence-free.

The constructor checks that the velocity, the pressure and the mask share their site indices, and calls
[`check_kolmogorov_scale`](@ref).

# Fields

  - `grid::Grid{N}`: the grid of the stored nodes.
  - `u_fields::VectorField{N}`: the velocity, initialized with its first state.
  - `p_field::ScalarField{N}`: the pressure, empty until the evolution stores it.
  - `Re::Number`: the Reynolds number.
  - `object_mask::Union{MPS,Nothing}`: the obstacle mask (1 in the fluid, 0 in the obstacle), or `nothing`.
  - `stencil_order::Int`: ``q``, 2 or 8.
  - `maxdim::Int`: maximum bond dimension of the TT arithmetic of the evolution (operators, Poisson right-hand side,
    projection, time step). It is not the bond dimension of the solver.
  - `cutoff::Float64`: SVD cutoff of the same TT arithmetic. It is not the truncation or the stopping tolerance of the
    solver, which live on the solver itself.

# Throws

  - `ArgumentError`: if `object_mask` is given and the site indices of the fields and the mask differ, or if `grid`
    flags an axis as periodic whose boundary conditions are not periodic.

# Known limitations

The Chorin projection of v0.1 has two known limitations, which are recorded as `@test_broken` in the test suite and
will be fixed in a later version:

  - **Mean removal.** When the pressure has no `DirichletBC` side, the Poisson system is singular and
    [`poisson_rhs`](@ref) makes its right-hand side consistent by subtracting a constant. That constant is
    ``\\sum_j b_j\\, \\prod_i h_i / \\prod_i L_i = \\sum_j b_j / \\prod_i (n_i - 1)`` instead of the node mean
    ``\\sum_j b_j / \\prod_i n_i``, so the mean of the right-hand side becomes ``1 - \\prod_i n_i/(n_i - 1)`` times its
    original value instead of zero, and the regularized system is not exactly consistent. This is inactive whenever the
    pressure has a Dirichlet side, as in the channel of Sec. 5.2 of the paper (Dirichlet outlet pressure).
  - **Wall rows.** The projection ``\\mathbf{u} = \\mathbf{u}^* - dt\\, \\nabla p`` (`∇p_correction`) is applied to
    every row, including the ``q/2`` rows next to the Dirichlet velocity faces that the time steppers otherwise keep
    fixed ([`BoundaryNull`](@ref)). The tangential pressure gradient therefore changes the boundary velocity a little at
    every step, which in the setup of Sec. 5.2 acts on the inlet rows.

# Examples

```julia
arch = QTTArchitecture((7, 6))
inlet_outlet = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))
periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
u_bcs = BoundaryConditions(inlet_outlet, periodic)
v_bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), periodic)
p_bcs = BoundaryConditions(BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), periodic)
sites = siteinds(arch)
mask = …                                       # obstacle mask on `sites`, e.g. from crossinterpolate2
u_fields = VectorField((mask, zeros_mps(sites)), arch, (u_bcs, v_bcs))
grid = Grid(arch, (2.0, 1.0); periodic=periodic_axes(p_bcs))   # period 1 in y
experiment = INSExperiment(grid, u_fields, ScalarField(sites, arch, p_bcs), 3000, mask)
```

See also [`ChorinCache`](@ref), [`chorin_evolve!`](@ref), and the boundary-condition tutorial
[Velocity–pressure formulation](@ref bc_velocity_pressure).
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

    function INSExperiment(
        grid::Grid{N},
        u_fields::VectorField{N},
        p_field::ScalarField{N},
        Re::Number,
        object_mask::Union{MPS,Nothing};
        stencil_order::Int=8,
        maxdim::Int=100,
        cutoff::Float64=1e-8,
    ) where {N}
        if !isnothing(object_mask) && !(u_fields.usites == p_field.usites == siteinds(object_mask))
            throw(ArgumentError("Physical sites for velocity, pressure and mask fields do not match"))
        end

        _check_grid_periodicity(grid, u_fields.components..., p_field)
        check_kolmogorov_scale(grid, Re)

        return new{N}(grid, u_fields, p_field, Re, object_mask, stencil_order, maxdim, cutoff)
    end
end

"""
    _check_grid_periodicity(grid::Grid{N}, fields::ScalarField{N}...) where N

Throw an `ArgumentError` if an axis that `grid` flags as periodic is not periodic in the boundary conditions of
every field ([`periodic_axes`](@ref)). The opposite case, a `PeriodicBC` axis on a grid built without the flag, is
accepted for backward compatibility: its stencils then have period ``L + h`` (see [`PeriodicBC`](@ref)).
"""
function _check_grid_periodicity(grid::Grid{N}, fields::ScalarField{N}...) where {N}
    for field in fields, i in 1:N
        if grid.periodic[i] && !periodic_axes(field.bcu)[i]
            throw(ArgumentError("axis $i of the grid is periodic, but the boundary conditions of a field are not"))
        end
    end
    return nothing
end

"""
    cfl_spectral_radii(experiment::INSExperimentBase; vmax::Real=1.0) -> Tuple{Float64,Float64}

Return upper bounds on the spectral radii of the discretized advection and diffusion operators of `experiment`, from
the coefficients of its finite-difference stencils (Gershgorin bound),

```math
\\lambda_\\text{adv} = \\sum_i \\frac{v_\\max}{h_i} \\sum_k |c^{(1)}_k|, \\qquad
\\lambda_\\text{diff} = \\sum_i \\frac{1}{Re\\, h_i^2} \\sum_k |c^{(2)}_k|,
```

where ``c^{(1)}`` and ``c^{(2)}`` are the first- and second-derivative coefficients of
[`derivative_coefficients`](@ref) and ``v_\\max`` = `vmax` bounds the velocity magnitude. The contributions of all axes
add, since the time stepper advances the combined right-hand side. The bound is attained for the diffusion, whose
coefficients alternate in sign so that the highest wavenumber reaches it, and is conservative for the advection.

# Returns

  - `(adv_eig, diff_eig)`: the bounds ``\\lambda_\\text{adv}`` and ``\\lambda_\\text{diff}``.

See also [`check_cfl`](@ref).
"""
function cfl_spectral_radii(experiment::INSExperimentBase; vmax::Real=1.0)
    grid = experiment.grid
    N = length(grid.dx)
    s1 = sum(abs, values(derivative_coefficients(1; stencil_order=experiment.stencil_order)))  # advection
    s2 = sum(abs, values(derivative_coefficients(2; stencil_order=experiment.stencil_order)))  # diffusion

    adv_eig = sum(vmax * s1 / grid.dx[i] for i in 1:N)
    diff_eig = sum((1/experiment.Re) * s2 / grid.dx[i]^2 for i in 1:N)
    return adv_eig, diff_eig
end

"""
    _report_cfl(dt, dt_max)

Log at `@info` level whether the time step `dt` satisfies the combined stability limit `dt_max` of [`check_cfl`](@ref).
"""
function _report_cfl(dt, dt_max)
    if dt > dt_max
        @info "CFL condition violated, must set dt<=$dt_max"
    else
        @info "CFL condition satisfied" dt dt_max
    end
end

"""
    check_cfl(experiment::INSExperimentBase, stepper::RK4; vmax::Real=1.0)
    check_cfl(experiment::INSExperimentBase, stepper::ExplicitEuler; vmax::Real=1.0)

Check the linear stability (CFL) condition of `experiment` for the time step ``dt`` of `stepper`, warn when it is
violated, and log the result at `@info` level.

An explicit step is stable when ``dt\\,\\lambda`` lies in the stability region of the scheme for every eigenvalue
``\\lambda`` of the linearized right-hand side. We bound these eigenvalues with [`cfl_spectral_radii`](@ref): the
central advection is skew-symmetric, so its eigenvalues lie on the imaginary axis, within ``\\lambda_\\text{adv}``,
and the diffusion is symmetric negative semidefinite, so its eigenvalues lie on the negative real axis, within
``\\lambda_\\text{diff}``. For RK4, whose stability region reaches ``2\\sqrt{2}`` on the imaginary axis and ``2.785`` on
the negative real axis, the limits are

```math
dt \\le \\frac{2\\sqrt{2}}{\\lambda_\\text{adv}}, \\qquad dt \\le \\frac{2.785}{\\lambda_\\text{diff}}.
```

The stability disk of forward Euler, ``|1 + z| \\le 1``, only touches the imaginary axis at the origin, so the central
advection is never stable on its own and must be damped by the diffusion. For a Fourier mode with advection symbol
``a`` and diffusion symbol ``d``, the step is stable if and only if ``dt \\le 2d/(d^2 + a^2)``, and since the stencils
satisfy ``a^2 \\le v_\\max^2\\, Re\\, d`` (by the Cauchy–Schwarz inequality on the velocity vector), the sufficient
limits are

```math
dt \\le \\frac{2}{\\lambda_\\text{diff}}, \\qquad dt \\le \\frac{2}{\\lambda_\\text{diff} + v_\\max^2\\, Re}.
```

Note that these are bounds for the linearized, constant-coefficient problem; the nonlinear flow and the TT
truncation are not covered. `vmax` is the assumed maximum velocity magnitude of the flow.

See also [`check_kolmogorov_scale`](@ref).
"""
function check_cfl(experiment::INSExperimentBase, stepper::RK4; vmax::Real=1.0)
    dt = stepper.time_evol_kwargs.dt
    adv_eig, diff_eig = cfl_spectral_radii(experiment; vmax=vmax)

    # RK4 stability boundaries: |z|<=2√2 on the imaginary axis (advection is skew-symmetric),
    # |z|<=2.785 on the negative real axis (diffusion is symmetric negative-definite).
    dt_adv_max = 2*sqrt(2) / adv_eig
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

function check_cfl(experiment::INSExperimentBase, stepper::ExplicitEuler; vmax::Real=1.0)
    dt = stepper.time_evol_kwargs.dt
    _, diff_eig = cfl_spectral_radii(experiment; vmax=vmax)

    # Forward Euler is stable iff |1 + z| <= 1, a disk that only touches the imaginary axis at the origin, so
    # central-difference advection is never stable on its own: it must be damped by diffusion. On the negative
    # real axis the boundary is |z| <= 2, which gives the pure diffusion limit.
    dt_diff_max = 2 / diff_eig

    # For a Fourier mode with advection symbol a and diffusion symbol d, |1 + dt(-d + ia)| <= 1 iff
    # dt <= 2d / (d² + a²). The stencils in use satisfy κ₂(θ) >= κ₁(θ)². With |u| <= vmax that gives
    # a² = |∑ u_i κ₁(θ_i)/h_i|² <= |u|² ∑ κ₂(θ_i)/h_i² = vmax² Re d, by Cauchy-Schwarz on the velocity
    # vector: the components cannot all equal vmax at once, so the dimension does not multiply the bound.
    # Hence dt <= 2 / (d + vmax² Re). Since d <= diff_eig, dt <= 2 / (diff_eig + vmax² Re) is sufficient.
    dt_max = 2 / (diff_eig + vmax^2 * experiment.Re)

    if dt > dt_diff_max
        @warn "CFL condition violated for diffusion term" dt dt_diff_max
    end
    if dt > dt_max
        @warn "CFL condition violated for the combined advection-diffusion term" dt dt_max
    end
    _report_cfl(dt, dt_max)
end

"""
    check_kolmogorov_scale(grid::Grid{N}, Re::Number) where N
    check_kolmogorov_scale(experiment::INSExperimentBase)

Check whether the grid resolves the Kolmogorov length scale ``\\eta = Re^{-3/4}`` (in the units of the domain):
warn for every axis whose spacing ``h_i`` exceeds ``\\eta``, and log at `@info` level when none does. A coarser grid
does not resolve the smallest scales of a turbulent flow, so the simulation is then under-resolved. The experiment
constructors call it.
"""
function check_kolmogorov_scale(grid::Grid{N}, Re::Number) where {N}
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

function check_kolmogorov_scale(experiment::INSExperimentBase)
    check_kolmogorov_scale(experiment.grid, experiment.Re)
end
