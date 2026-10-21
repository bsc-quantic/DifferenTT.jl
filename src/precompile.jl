# Precompilation workload (PrecompileTools). We run the compile-heavy paths of the tutorials on tiny problems while the
# package is precompiled, so that their first calls in a session load cached native code instead of compiling it: the
# Laplacians of stencil orders 2 and 8 with Dirichlet, Neumann and periodic sides, the dense conversions, the local and
# global solvers with and without a preconditioner, the FAI and linear-solve preconditioners, cross interpolation on a
# grid, and one RK4 step of the Chorin projection and of the vorticity–streamfunction scheme. The workload prints
# nothing, writes no files, uses seeded generators, and restores the state of the global RNG (the Physics functions draw
# from it) and the order-warning threshold of ITensors.
#
# Methods that depend on the number of variables are compiled for one-variable and two-variable QTT problems and for a
# five-variable TT problem, and the dense conversions (`contract_mpo`, `contract_mps`, `mat2mpo`, `vec2mps`), which
# specialize on the tensor order, for the site counts used here. Other cases, other array backends (e.g. `CuArray`),
# element types and random number generators other than `Xoshiro` and `Random.default_rng()` compile on their first
# call, as before.
@setup_workload begin
    dirichlet = BoundaryCond1d(DirichletBC(1.0), DirichletBC(2.0))
    neumann_dirichlet = BoundaryCond1d(NeumannBC(0.0), DirichletBC(1.0))
    periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    local_args(; kwargs...) = LocalUpdaterArgs(;
        ishermitian=true, isposdef=true, rtol=1e-6, krylovdim=1, maxiter=50, kwargs...
    )
    sweep_args(; kwargs...) = SweepingArgs(; nsweeps=2, maxdim=8, cutoff=0.0, kwargs...)

    @compile_workload begin
        with_logger(NullLogger()) do
            rng_state = copy(Random.default_rng())
            Random.seed!(1)
            warn_order = ITensors.disable_warn_order() # the dense conversions print a warning for large orders
            try
                rng = Xoshiro(1)
                plain = ResidualALS(; sweep_kwargs=sweep_args(), updater_kwargs=local_args(), residual_tol=1e-6)
                recorded = ResidualALS(;
                    sweep_kwargs=sweep_args(; (observer!)=ResidualObserver(; rtol=1e-6)),
                    updater_kwargs=local_args(; numiter_store=KrylovSweepStats()),
                    residual_tol=1e-6,
                )

                # One variable: the dense forms of the Laplacian on 2^4 to 2^10 nodes
                for nbits in 4:10
                    arch = QTTArchitecture((nbits,))
                    sites = siteinds(arch)
                    field = ScalarField(sites, arch, BoundaryConditions(dirichlet))
                    A = -1.0 * symmetrized_laplacian(field, interior_grid(arch, (1.0,)); stencil_order=2)
                    contract_mpo(A)
                    contract_mps(ones_mps(sites))
                end
                # One variable on 2^4 nodes: Dirichlet, Neumann and periodic Laplacians, and the solves of the tutorials
                arch = QTTArchitecture((4,))
                sites = siteinds(arch)
                f(x) = exp(-x[1]) * sin(8π * x[1])
                for (bcs, grid) in (
                    (dirichlet, interior_grid(arch, (1.0,))),
                    (neumann_dirichlet, interior_grid(arch, (1.0,), (1,))),
                    (periodic, Grid(arch, 1.0; periodic=true)),
                )
                    field = ScalarField(sites, arch, BoundaryConditions(bcs))
                    pivot = optfirstpivot(Float64, f, grid, arch; rng)
                    tci, _, _ = crossinterpolate2(Float64, f, grid, arch; initialpivots=[pivot], tolerance=1e-8)
                    b = MPS(tci)
                    replace_siteinds!(b, sites)
                    for q in (2, 8)
                        A = -1.0 * symmetrized_laplacian(field, grid; stencil_order=q)
                        -1.0 * kronecker_laplacian(field, grid; stencil_order=q)
                        correction = scalar_dirichlet_correction(field, grid, 2; stencil_order=q)
                        isnothing(correction) || +(b, apply(correction, ones_mps(sites, 1)); alg="directsum")
                        symmetrize_rhs(field, b)
                        bcs === periodic && (A = +(A, identity_mpo(sites); alg="directsum"))
                        linearsolve(recorded, A, b, nothing)
                    end
                end

                # Two variables on 2^4 x 2^4 nodes, blocked as in the tutorials
                arch = QTTArchitecture((4, 4); interleaved=false)
                sites = siteinds(arch)
                pressure_bcs = BoundaryConditions(neumann_dirichlet, periodic)
                grid = Grid(arch, (2.0, 1.0); periodic=periodic_axes(pressure_bcs))
                field = ScalarField(sites, arch, pressure_bcs)
                symmetrized_laplacian(field, grid; stencil_order=2)
                A = -1.0 * symmetrized_laplacian(field, grid; stencil_order=8)
                for bcs in (BoundaryConditions(dirichlet, dirichlet), BoundaryConditions(neumann_dirichlet, dirichlet))
                    bc_field = ScalarField(sites, arch, bcs)
                    bc_grid = interior_grid(arch, (1.0, 1.0))
                    for q in (2, 8)
                        symmetrized_laplacian(bc_field, bc_grid; stencil_order=q)
                    end
                end
                mat2mpo(contract_mpo(A), sites; cutoff=1e-12)
                b = random_mps(rng, sites; linkdims=2)
                vec2mps(contract_mps(b), sites; cutoff=1e-12)

                # Local and global solvers, with and without a preconditioner
                x0 = random_mps(rng, sites)
                linearsolve(plain, A, b, nothing)
                linearsolve(recorded, A, b, x0)
                linearsolve(ALS(; sweep_kwargs=sweep_args(), updater_kwargs=local_args()), A, b, nothing)
                linearsolve(AMEn(; sweep_kwargs=sweep_args(), updater_kwargs=local_args()), A, b, nothing)
                linsolve_preconditioner(A, plain; residual_tol=1e-3, rng)
                fai = fai_preconditioner(A; μ=1e3, solver=plain, totalmaxdim=2, residual_tol=0.1, rng)
                linearsolve(plain, A, b, fai, nothing)
                linearsolve(recorded, A, b, fai, x0)
                global_args = GlobalUpdaterArgs(;
                    ishermitian=true, isposdef=true, rtol=1e-4, maxiter=5, maxdim=8, cutoff=1e-12, numiter_store=Ref(0)
                )
                linearsolve(GlobalCG(; updater_kwargs=global_args), A, b, nothing)
                linearsolve(GlobalCG(; updater_kwargs=global_args), A, b, fai, nothing)

                # One RK4 step of the Chorin projection around an obstacle, whose mask comes from cross interpolation
                mask_function(x) = 0.5 * (1 + tanh(50 * ((x[1] - 0.5)^2 + (x[2] - 0.5)^2 - 0.2^2)))
                pivot = optfirstpivot(Float64, mask_function, grid, arch; rng)
                tci, _, _ = crossinterpolate2(
                    Float64, mask_function, grid, arch; initialpivots=[pivot], tolerance=1e-6, maxbonddim=8
                )
                mask = MPS(tci)
                replace_siteinds!(mask, sites)
                u_bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0)), periodic)
                v_bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), periodic)
                chorin = INSExperiment(
                    grid,
                    VectorField((copy(mask), zeros_mps(sites)), arch, (u_bcs, v_bcs)),
                    ScalarField(sites, arch, pressure_bcs),
                    100,
                    mask;
                    stencil_order=8,
                    maxdim=8,
                    cutoff=1e-8,
                )
                stepper = RK4(; time_evol_kwargs=TimeEvolArgs(0.0, 1e-4, 1), cutoff=1e-8, maxdim=8)
                chorin_evolve!(chorin, plain, stepper)

                # One RK4 step of the vorticity–streamfunction scheme in the lid-driven cavity
                faces = cavity_faces(1.0)
                vorticity_bcs = vorticity_boundary_conditions(faces)
                vorticity = INSVorticityExperiment(
                    vorticity_grid(arch, (1.0, 1.0), faces),
                    ScalarField(zeros_mps(sites), arch, vorticity_bcs),
                    ScalarField(sites, arch, vorticity_bcs),
                    100,
                    faces;
                    stencil_order=2,
                    maxdim=8,
                    cutoff=1e-8,
                )
                vorticity_evolve!(vorticity, plain, stepper)

                # Five variables on 4^5 nodes in the TT layout: equicorrelated Laplacian and Sinc preconditioner
                N = 5
                arch = TTArchitecture(ntuple(_ -> 4, N))
                sites = siteinds(arch)
                grid = interior_grid(arch, ntuple(_ -> 1.0, N))
                zero_dirichlet = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
                field = ScalarField(sites, arch, BoundaryConditions(ntuple(_ -> zero_dirichlet, N)...))
                symmetrized_laplacian(field, grid; stencil_order=2)
                A = -1.0 * equicorrelated_laplacian(field, grid; correlation=0.5, stencil_order=2)
                T = -onedim_differential_operator(2, arch, 1, zero_dirichlet; stencil_order=2) ./ grid.dx[1]^2
                sinc = sincquadrature_preconditioner(arch, sites, fill(T, N), 3)
                mat2mpo(contract_mpo(A), sites; maxdim=2, cutoff=0.0)
                b = ones_mps(sites)
                linearsolve(recorded, A, b, sinc, random_mps(rng, sites))
            finally
                ITensors.set_warn_order(warn_order)
                copy!(Random.default_rng(), rng_state)
            end
        end
    end
end
