```@meta
CurrentModule = DifferenTT
```

# Navier–Stokes

The incompressible Navier–Stokes equations couple the velocity ``\mathbf{u}`` and the pressure ``p`` through the
constraint ``\nabla \cdot \mathbf{u} = 0``, which makes the pressure a Lagrange multiplier that is determined by a Poisson
equation at every time step. DifferenTT offers two formulations of the equations, with the same fields, operators,
time steppers and linear solvers, and we describe here which type to pass and what each one imposes. The derivation of the
boundary conditions of each formulation is in the corresponding tutorial ([Velocity–pressure formulation](@ref bc_velocity_pressure)
and [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction)), and the exact rules are in the docstrings.
The signatures are in the [API reference](@ref api_physics).

## Time steppers

An evolution is advanced by an explicit scheme, [`ExplicitEuler`](@ref) (first order in ``dt``) or [`RK4`](@ref) (fourth
order), configured by a [`TimeEvolArgs`](@ref) with the initial and final times and the number of steps. [`timestep`](@ref)
advances a state by one step given a right-hand side ([`AbstractRHS`](@ref)). The implicit steppers are a work in progress
and are not part of the supported API.

## Velocity–pressure formulation: Chorin projection

An [`INSExperiment`](@ref) holds the grid, the velocity ([`VectorField`](@ref)), the pressure ([`ScalarField`](@ref)), the
Reynolds number and an optional obstacle mask (1 in the fluid, 0 in the obstacle). It follows the QTT pipeline of
Peddinti et al. (2024) as used in Sec. 5.2 of the paper: the velocity is predicted by a time step of the momentum equation
([`NavierStokesPredictorRHS`](@ref)), the obstacle enters by multiplying the velocity by the mask, and the projection
solves a Poisson equation for the pressure with the operator of [`symmetrized_laplacian`](@ref) and then subtracts its
gradient. The fields live on a collocated grid, and the pressure system uses the compact Laplacian instead of the
composition of the divergence and the gradient, so the projected velocity is only approximately divergence-free.

[`chorin_evolve!`](@ref) runs the evolution, taking any linear solver for the pressure system and any explicit stepper.
The caches that it builds once ([`ChorinCache`](@ref)), the single step [`chorin_step`](@ref), and the checks
[`check_cfl`](@ref) and [`check_kolmogorov_scale`](@ref) are available separately. The boundary conditions are the types of
[Grids, fields and boundary conditions](@ref): in the setup of the paper the velocity has a Dirichlet inlet and a Neumann
outlet, the pressure a Dirichlet outlet, and the second axis is periodic. For such an axis we recommend the grid
`Grid(arch, box_size; periodic = DifferenTT.periodic_axes(p_bcs))`, which gives the stencils their intended period
``L`` (the plain `Grid` gives ``L + h``). The tutorial [Velocity–pressure formulation](@ref bc_velocity_pressure) derives
the closures of this formulation, and [Channel flow past a cylinder](@ref) runs the pipeline of the paper in miniature.

## Vorticity–streamfunction formulation

An [`INSVorticityExperiment`](@ref) evolves the vorticity ``\omega`` and recovers the velocity from the stream function
``\psi``, which solves a Poisson equation ([`poisson_streamfunction`](@ref), [`streamfunction_velocity`](@ref)). The
formulation has no pressure and its velocity is divergence-free by construction. Instead of boundary conditions on fields,
each side of the domain is a [`VorticityFace`](@ref) that fixes the conditions of both ``\psi`` and ``\omega``:
[`NoSlipWall`](@ref), [`SlipWall`](@ref), [`InflowFace`](@ref), [`OutflowFace`](@ref) and [`PeriodicFace`](@ref).
[`vorticity_boundary_conditions`](@ref) derives the boundary conditions of the fields from the faces, [`vorticity_grid`](@ref)
the grid, and [`cavity_faces`](@ref) the faces of the lid-driven cavity (Ghia, Ghia & Shin, 1982). The evolution is run
by [`vorticity_evolve!`](@ref). Walls and inflows are ghost nodes one spacing outside the stored nodes, which is why the
grid of this formulation is not the default one.
The derivation of every face is in the tutorial [Vorticity–streamfunction formulation](@ref bc_vorticity_streamfunction),
and [Lid-driven cavity](@ref) runs the formulation on the classical benchmark.

## Known limitations

The following limitations of v0.1 are recorded in the test suite, as `@test_broken` where a test applies, and in the release
notes. They are planned for v0.2 and are not hidden by any option.

  - **Mean removal in the Chorin projection.** When the pressure has no Dirichlet side, its Poisson system is singular and
    the right-hand side is made consistent by subtracting a constant. That constant divides by ``\prod_i (n_i - 1)`` instead
    of ``\prod_i n_i``, so the regularized system is not exactly consistent. This is inactive when the pressure has a
    Dirichlet side, as in the channel of Sec. 5.2 of the paper.
  - **Wall rows in the Chorin projection.** The projection ``\mathbf{u} = \mathbf{u}^* - dt\, \nabla p`` is applied to every
    row, including the ``q/2`` rows next to the Dirichlet velocity faces that the time steppers otherwise keep fixed, so the
    tangential pressure gradient changes the boundary velocity a little at every step.
  - **Dirichlet closure with ``q = 8``.** The lift fills every ghost node beyond a Dirichlet side with the boundary value,
    which places the Dirichlet point one spacing beyond the last stored node and makes the closure only first-order
    accurate for ``q = 8`` (second order for ``q = 2``). This affects the outlet pressure of the Chorin projection.
  - **Order of the Chorin step.** `chorin_step` masks and projects the velocity it receives and then predicts, so the velocity
    that it returns, and that `chorin_evolve!` stores, is the provisional field ``\mathbf{u}^*`` of the next step and not
    the projected one. The order predict, mask, project, which solves the same pressure systems, is to be reconsidered.
  - **Period of a periodic axis.** `Grid` without the `periodic` flag keeps a node at both ends of a periodic axis, so its
    stencils have period ``L + h`` ([Grids, fields and boundary conditions](@ref)).
