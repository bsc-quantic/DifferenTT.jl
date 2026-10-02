using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra
using Random
using JLD2

# -------------------------------------------------------------------------------------------------
# Dense references of the vorticity-streamfunction scheme with ghost walls. Fields are stored as F[x, y]
# on the K₁ × K₂ interior nodes; `pad` adds the ghost walls with value 0.
# -------------------------------------------------------------------------------------------------

function dense_field(mps::MPS, grid::Grid{2}, arch)
    K1, K2 = variable_dims(arch)
    return [DifferenTT.sample(mps, ((i - 1) * grid.dx[1], (j - 1) * grid.dx[2]), grid, arch) for i in 1:K1, j in 1:K2]
end

function pad(F::AbstractMatrix)
    P = zeros(size(F, 1) + 2, size(F, 2) + 2)
    P[2:end-1, 2:end-1] .= F
    return P
end

dense_dx(P, hx) = (P[3:end, 2:end-1] .- P[1:end-2, 2:end-1]) ./ (2hx)
dense_dy(P, hy) = (P[2:end-1, 3:end] .- P[2:end-1, 1:end-2]) ./ (2hy)

function dense_laplacian(P, hx, hy)
    C = P[2:end-1, 2:end-1]
    return (P[1:end-2, 2:end-1] .+ P[3:end, 2:end-1] .- 2C) ./ hx^2 .+ (P[2:end-1, 1:end-2] .+ P[2:end-1, 3:end] .- 2C) ./ hy^2
end

# Fill the ghost walls of the padded vorticity with the wall vorticity formula (lid on top, moving in +x)
function dense_wall_vorticity!(Ωp, Ψ, hx, hy, U; formula)
    c1, c2, cU = DifferenTT.wall_vorticity_coefficients(formula)
    K1, K2 = size(Ψ)
    Ωp[1, 2:K2+1] .= (c1 .* Ψ[1, :] .+ c2 .* Ψ[2, :]) ./ hx^2
    Ωp[K1+2, 2:K2+1] .= (c1 .* Ψ[K1, :] .+ c2 .* Ψ[K1-1, :]) ./ hx^2
    Ωp[2:K1+1, 1] .= (c1 .* Ψ[:, 1] .+ c2 .* Ψ[:, 2]) ./ hy^2
    Ωp[2:K1+1, K2+2] .= (c1 .* Ψ[:, K2] .+ c2 .* Ψ[:, K2-1]) ./ hy^2 .+ cU * U / hy
    return Ωp
end

dense_full_laplacian(Ω, Ψ, hx, hy, U; formula) = dense_laplacian(dense_wall_vorticity!(pad(Ω), Ψ, hx, hy, U; formula=formula), hx, hy)

# Conservative advection: the normal velocity vanishes on the walls, so the ghost fluxes are 0
function dense_transport_rhs(Ω, Ψ, hx, hy, U, Re; formula)
    Ψp = pad(Ψ)
    u = dense_dy(Ψp, hy)
    v = -dense_dx(Ψp, hx)
    advection = dense_dx(pad(u .* Ω), hx) .+ dense_dy(pad(v .* Ω), hy)
    return -advection .+ dense_full_laplacian(Ω, Ψ, hx, hy, U; formula=formula) ./ Re
end

# -Δ acting on vec(F), with x the fastest index
function dense_poisson_matrix(K1, K2, hx, hy)
    second_derivative(n, h) = Matrix(SymTridiagonal(fill(-2.0, n), ones(n - 1))) ./ h^2
    return -(kron(Matrix(1.0I, K2, K2), second_derivative(K1, hx)) + kron(second_derivative(K2, hy), Matrix(1.0I, K1, K1)))
end

# One-solve-per-step scheme: velocity and wall vorticity frozen over the stages
function dense_evolve(Ω0, nsteps, dt, hx, hy, U, Re; formula, scheme)
    A = dense_poisson_matrix(size(Ω0)..., hx, hy)
    Ω = copy(Ω0)
    Ψ = zero(Ω0)
    for _ in 1:nsteps
        Ψ = reshape(A \ vec(Ω), size(Ω))
        R(W) = dense_transport_rhs(W, Ψ, hx, hy, U, Re; formula=formula)
        if scheme == :euler
            Ω = Ω .+ dt .* R(Ω)
        else
            k1 = R(Ω)
            k2 = R(Ω .+ (dt / 2) .* k1)
            k3 = R(Ω .+ (dt / 2) .* k2)
            k4 = R(Ω .+ dt .* k3)
            Ω = Ω .+ (dt / 6) .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
    end
    return Ω, Ψ
end

relerr(a, b) = norm(a .- b) / norm(b)

arch_label(arch::QTTArchitecture) = "QTT $(arch.var_nsites)$(arch.is_interleaved ? " interleaved" : "")"
arch_label(arch::TTArchitecture) = "TT $(arch.var_dims)"

@testset "Vorticity-streamfunction" begin
    # DirichletBC logs at every construction: build the wall conditions once
    wall = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    walls = BoundaryConditions(wall, wall)
    periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    function cavity_experiment(arch; Re=100.0, U=1.0, formula=:jensen, maxdim=256, cutoff=1e-14, bcs=walls)
        sites = siteinds(arch)
        grid = interior_grid(arch, (1.0, 1.0))
        omega_field = ScalarField(zeros_mps(sites), arch, bcs)
        psi_field = ScalarField(sites, arch, bcs)
        return INSVorticityExperiment(grid, omega_field, psi_field, Re, U; wall_formula=formula, maxdim=maxdim, cutoff=cutoff)
    end

    exact_solver() = GlobalCG(updater_kwargs=GlobalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-13, maxiter=2000, maxdim=256, cutoff=1e-15, orth_maxdim=256, orth_cutoff=1e-15))
    # On a 2-site TT, a single 2-site ALS update is a direct CG solve of the full system
    tt_direct_solver(rtol) = ALS(; sweep_kwargs=SweepingArgs(; nsweeps=1, maxdim=256, cutoff=1e-15), updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=rtol, maxiter=1000, krylovdim=1))

    qtt_archs = (QTTArchitecture((4, 4)), QTTArchitecture((4, -4)), QTTArchitecture((4, 4); interleaved=true))
    archs = (qtt_archs..., TTArchitecture((16, 16)))

    # interior_grid, entry_mpo and the 1D wall vorticity corrections are tested with their modules
    # (test_grid.jl, test_qttconstructors.jl, test_bcoperators.jl)

    @testset "Constructor checks" begin
        arch = QTTArchitecture((4, 4))
        @test cavity_experiment(arch) isa INSVorticityExperiment{2}
        @test_throws ArgumentError cavity_experiment(arch; formula=:unknown)
        @test_throws ArgumentError cavity_experiment(arch; bcs=BoundaryConditions(periodic, wall))
        @test_throws ArgumentError cavity_experiment(arch; bcs=BoundaryConditions(wall, BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))))

        # Stored settings; an integer lid velocity is stored as Float64
        experiment = cavity_experiment(arch; Re=250, U=2, formula=:thom, maxdim=17, cutoff=1e-9)
        @test experiment.U_lid === 2.0
        @test experiment.Re == 250
        @test experiment.wall_formula == :thom
        @test experiment.stencil_order == 2
        @test experiment.maxdim == 17
        @test experiment.cutoff == 1e-9

        # Only 2D is supported
        arch_3d = QTTArchitecture((2, 2, 2))
        sites_3d = siteinds(arch_3d)
        walls_3d = BoundaryConditions(wall, wall, wall)
        @test_throws ArgumentError INSVorticityExperiment(interior_grid(arch_3d, (1.0, 1.0, 1.0)), ScalarField(zeros_mps(sites_3d), arch_3d, walls_3d), ScalarField(sites_3d, arch_3d, walls_3d), 100.0, 1.0)

        sites = siteinds(arch)
        grid = interior_grid(arch, (1.0, 1.0))
        omega_field = ScalarField(zeros_mps(sites), arch, walls)
        psi_field = ScalarField(sites, arch, walls)
        @test_throws ArgumentError INSVorticityExperiment(grid, omega_field, psi_field, 100.0, 1.0; stencil_order=8)
        @test_throws ArgumentError INSVorticityExperiment(grid, omega_field, ScalarField(siteinds(arch), arch, walls), 100.0, 1.0)
    end

    @testset "Poisson operator ($(arch_label(arch)))" for arch in archs
        experiment = cavity_experiment(arch)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        K1, K2 = variable_dims(arch)

        Random.seed!(1)
        psi = random_mps(experiment.psi_field.usites; linkdims=4)
        Ψ = dense_field(psi, grid, arch)
        A_dense = dense_poisson_matrix(K1, K2, grid.dx...)
        @test relerr(dense_field(apply(cache.poisson_op, psi), grid, arch), reshape(A_dense * vec(Ψ), K1, K2)) < 1e-10

        A = contract_mpo(cache.poisson_op)
        @test A ≈ A'
        @test eigmin(Symmetric(A)) > 0
        @test A ≈ -contract_mpo(cache.laplacian)

        # The right-hand side of -Δψ = ω is the vorticity itself
        @test DifferenTT.poisson_rhs(experiment, cache, psi) === psi
    end

    @testset "Wall vorticity operator and lid source" begin
        arch = QTTArchitecture((4, 4))
        K1, K2 = variable_dims(arch)
        sites = siteinds(arch)
        grid = interior_grid(arch, (1.0, 1.0))
        hx, hy = grid.dx
        Random.seed!(6)
        psi = random_mps(sites; linkdims=4)
        Ψ = dense_field(psi, grid, arch)

        for formula in (:jensen, :thom)
            c1, c2, cU = DifferenTT.wall_vorticity_coefficients(formula)

            # Periodic in x, walls in y: only the y-walls contribute, (c₁ψ₁ + c₂ψ₂) / h⁴ next to each of them
            field = ScalarField(sites, arch, BoundaryConditions(periodic, wall))
            S = DifferenTT.wall_vorticity_operator(field, grid; formula=formula)
            expected = zeros(K1, K2)
            expected[:, 1] .= (c1 .* Ψ[:, 1] .+ c2 .* Ψ[:, 2]) ./ hy^4
            expected[:, K2] .= (c1 .* Ψ[:, K2] .+ c2 .* Ψ[:, K2-1]) ./ hy^4
            @test relerr(dense_field(apply(S, psi), grid, arch), expected) < 1e-12

            # Lid term c_U U / h_y³ on the top row only
            lid = DifferenTT.lid_vorticity_source(field, grid, 1.3; formula=formula)
            expected_lid = zeros(K1, K2)
            expected_lid[:, K2] .= cU * 1.3 / hy^3
            @test dense_field(lid, grid, arch) ≈ expected_lid
        end

        # No wall at all: no operator
        @test isnothing(DifferenTT.wall_vorticity_operator(ScalarField(sites, arch, BoundaryConditions(periodic, periodic)), grid))

        # The lid source is only defined in 2D
        arch_3d = QTTArchitecture((2, 2, 2))
        field_3d = ScalarField(siteinds(arch_3d), arch_3d, BoundaryConditions(wall, wall, wall))
        @test_throws ArgumentError DifferenTT.lid_vorticity_source(field_3d, interior_grid(arch_3d, (1.0, 1.0, 1.0)), 1.0)
    end

    @testset "Manufactured stream function" begin
        errors = Float64[]
        u_errors = Float64[]
        for K in (16, 32)
            arch = TTArchitecture((K, K))
            experiment = cavity_experiment(arch)
            cache = VorticityCache(experiment)
            h = experiment.grid.dx[1]
            x = h .* (1:K) # physical coordinates of the interior nodes
            s = experiment.psi_field.usites

            Ψ = [sin(π * xi) * sin(π * yj) for xi in x, yj in x]
            psi = MPS(ITensor(Ψ, s[1], s[2]), s)

            AΨ = Array(prod(apply(cache.poisson_op, psi)), s[1], s[2])
            u, v = streamfunction_velocity(cache, psi)
            U = Array(prod(u), s[1], s[2])
            V = Array(prod(v), s[1], s[2])

            push!(errors, relerr(AΨ, 2π^2 .* Ψ))
            push!(u_errors, relerr(U, [π * sin(π * xi) * cos(π * yj) for xi in x, yj in x]))
            # v = -∂ₓψ is the transpose of u = ∂ᵧψ for this symmetric ψ
            @test relerr(V, [-π * cos(π * xi) * sin(π * yj) for xi in x, yj in x]) ≈ u_errors[end]
            # Leading truncation error of the central first derivative: (πh)²/6
            @test u_errors[end] < 1.1 * (π * h)^2 / 6
        end
        @test errors[2] < 5e-3
        @test 3.5 < errors[1] / errors[2] < 4.5 # second order
        @test 3.5 < u_errors[1] / u_errors[2] < 4.5
    end

    @testset "Discrete divergence of the velocity ($(arch_label(arch)))" for arch in archs
        experiment = cavity_experiment(arch)
        cache = VorticityCache(experiment)
        Random.seed!(2)
        psi = random_mps(experiment.psi_field.usites; linkdims=4)
        u, v = streamfunction_velocity(cache, psi)
        div = +(apply(cache.diff_ops.d_ops[1], u), apply(cache.diff_ops.d_ops[2], v); cutoff=1e-15)
        @test norm(div) < 1e-9 * norm(u)
    end

    @testset "Wall vorticity source and transport RHS ($(arch_label(arch)), $formula)" for arch in archs, formula in (:jensen, :thom)
        Re, U = 100.0, 1.3
        experiment = cavity_experiment(arch; Re=Re, U=U, formula=formula)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        hx, hy = grid.dx

        Random.seed!(3)
        psi = random_mps(experiment.psi_field.usites; linkdims=4)
        omega = random_mps(experiment.psi_field.usites; linkdims=4)
        Ψ = dense_field(psi, grid, arch)
        Ω = dense_field(omega, grid, arch)

        # Δω completed with the wall vorticity = full 5-point Laplacian with the ghost walls filled by the formula
        source = DifferenTT.wall_vorticity_source(cache, psi)
        full_laplacian = +(apply(cache.laplacian, omega), source; cutoff=1e-15)
        @test relerr(dense_field(full_laplacian, grid, arch), dense_full_laplacian(Ω, Ψ, hx, hy, U; formula=formula)) < 1e-10

        u, v = streamfunction_velocity(cache, psi)
        rhs = VorticityTransportRHS(experiment, cache, u, v, source)
        tt_rhs = only(rhs((omega,), 0.0, 1))
        @test relerr(dense_field(tt_rhs, grid, arch), dense_transport_rhs(Ω, Ψ, hx, hy, U, Re; formula=formula)) < 1e-8
    end

    @testset "Zero vorticity skips the solve" begin
        experiment = cavity_experiment(QTTArchitecture((4, 4)))
        cache = VorticityCache(experiment)
        psi, residual = poisson_streamfunction(experiment, cache, zeros_mps(experiment.psi_field.usites), exact_solver())
        @test DifferenTT.iszero_tt(psi)
        @test residual == 0.0
    end

    @testset "poisson_streamfunction" begin
        arch = QTTArchitecture((4, 4))
        K1, K2 = variable_dims(arch)
        experiment = cavity_experiment(arch)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        Random.seed!(4)
        omega = random_mps(experiment.psi_field.usites; linkdims=4)
        Ω = dense_field(omega, grid, arch)
        A_dense = dense_poisson_matrix(K1, K2, grid.dx...)

        # Solution of -Δψ = ω, up to the truncation floor of the TT arithmetic (cutoff is a squared relative error)
        psi, residual = poisson_streamfunction(experiment, cache, omega, exact_solver())
        @test relerr(dense_field(psi, grid, arch), reshape(A_dense \ vec(Ω), K1, K2)) < 1e-6
        @test residual < 1e-6 * norm(omega)

        # The returned residual is ||Aψ - ω||: check it on an unconverged solve, where it is large
        one_iteration = GlobalCG(updater_kwargs=GlobalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-13, maxiter=1, maxdim=256, cutoff=1e-15, orth_maxdim=256, orth_cutoff=1e-15))
        rough_psi, rough_residual = poisson_streamfunction(experiment, cache, omega, one_iteration)
        @test rough_residual > 1e-3 * norm(omega)
        @test rough_residual ≈ norm(+(apply(cache.poisson_op, rough_psi; cutoff=1e-15), -1.0 * omega; cutoff=1e-15)) rtol=1e-6

        # The last stream function of the experiment is the initial guess: from the exact solution, CG returns at once
        numiter_store = Ref{Int}(-1)
        counting_solver = GlobalCG(updater_kwargs=GlobalUpdaterArgs(ishermitian=true, isposdef=true, rtol=1e-6, maxiter=2000, maxdim=256, cutoff=1e-15, orth_maxdim=256, orth_cutoff=1e-15, numiter_store=numiter_store))
        poisson_streamfunction(experiment, cache, omega, counting_solver)
        @test numiter_store[] > 0
        push!(experiment.psi_field, psi)
        poisson_streamfunction(experiment, cache, omega, counting_solver)
        @test numiter_store[] == 0
    end

    @testset "vorticity_transport from rest (priming step) ($(arch_label(arch)), $scheme)" for arch in (QTTArchitecture((4, 4)), TTArchitecture((16, 16))), scheme in (:euler, :rk4)
        Re, U, dt = 100.0, 1.0, 0.01
        experiment = cavity_experiment(arch; Re=Re, U=U)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        zero_field = zeros_mps(experiment.psi_field.usites)
        evol_args = TimeEvolArgs(0.0, dt, 1)
        stepper = scheme == :rk4 ? RK4(; time_evol_kwargs=evol_args, cutoff=1e-14, maxdim=256) : ExplicitEuler(; time_evol_kwargs=evol_args, cutoff=1e-14, maxdim=256)

        # At rest u = v = 0 and ψ = 0: only the lid drives the flow, and the zero products must not break the step
        omega_1 = DifferenTT.vorticity_transport(experiment, cache, zero_field, zero_field, stepper, 0.0)
        Ω1 = dense_field(omega_1, grid, arch)
        @test all(isfinite, Ω1)
        if scheme == :euler
            @test relerr(Ω1, (dt / Re) .* dense_field(cache.lid_source, grid, arch)) < 1e-12
        end
        # Truncation level of the RK4 stage sums
        Ω_ref, _ = dense_evolve(zeros(variable_dims(arch)...), 1, dt, grid.dx..., U, Re; formula=:jensen, scheme=scheme)
        @test relerr(Ω1, Ω_ref) < 1e-6
    end

    @testset "vorticity_step" begin
        arch = TTArchitecture((16, 16))
        experiment = cavity_experiment(arch)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        Random.seed!(5)
        omega = random_mps(experiment.psi_field.usites; linkdims=4)
        stepper = ExplicitEuler(time_evol_kwargs=TimeEvolArgs(0.0, 0.001, 1), cutoff=1e-14, maxdim=256)

        omega_next, psi, residual = vorticity_step(experiment, cache, omega, tt_direct_solver(1e-13), stepper, 0.0)

        # One Poisson solve followed by the transport with that stream function
        psi_ref, _ = poisson_streamfunction(experiment, cache, omega, tt_direct_solver(1e-13))
        omega_ref = DifferenTT.vorticity_transport(experiment, cache, omega, psi_ref, stepper, 0.0)
        @test relerr(dense_field(psi, grid, arch), dense_field(psi_ref, grid, arch)) < 1e-10
        @test relerr(dense_field(omega_next, grid, arch), dense_field(omega_ref, grid, arch)) < 1e-8
        @test residual < 1e-6 * norm(omega)

        # The experiment is not modified
        @test length(experiment.psi_field) == 0
        @test length(experiment.omega_field) == 1
    end

    @testset "End-to-end against the dense scheme ($(arch_label(arch)), $scheme)" for (arch, scheme) in ((QTTArchitecture((4, 4)), :rk4), (TTArchitecture((16, 16)), :euler))
        Re, U, dt, nsteps = 100.0, 1.0, 0.01, 4
        experiment = cavity_experiment(arch; Re=Re, U=U)
        kwargs = (; cutoff=1e-14, maxdim=256)
        evol_args = TimeEvolArgs(0.0, dt * nsteps, nsteps)
        stepper = scheme == :rk4 ? RK4(; time_evol_kwargs=evol_args, kwargs...) : ExplicitEuler(; time_evol_kwargs=evol_args, kwargs...)

        solver = arch isa TTArchitecture ? tt_direct_solver(1e-13) : exact_solver()
        vorticity_evolve!(experiment, solver, stepper)
        @test length(experiment.omega_field) == nsteps + 1
        @test length(experiment.psi_field) == nsteps

        grid = experiment.grid
        Ω, Ψ = dense_evolve(zeros(variable_dims(arch)...), nsteps, dt, grid.dx..., U, Re; formula=:jensen, scheme=scheme)
        # Truncation level of the randomized tt_mul; a sign or scaling error would be O(1)
        @test relerr(dense_field(last(experiment.omega_field), grid, arch), Ω) < 1e-6
        @test relerr(dense_field(last(experiment.psi_field), grid, arch), Ψ) < 1e-6
    end

    @testset "vorticity_evolve! smoke run" begin
        arch = TTArchitecture((16, 16))
        experiment = cavity_experiment(arch; Re=10.0, maxdim=32, cutoff=1e-12)
        stepper = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.3, 30), cutoff=1e-12, maxdim=32)
        @test isnothing(check_cfl(experiment, stepper; vmax=experiment.U_lid))

        vorticity_evolve!(experiment, tt_direct_solver(1e-10), stepper, 10)
        # Checkpoints at steps 1, 11, 21 and 30
        @test length(experiment.omega_field) == 5
        @test length(experiment.psi_field) == 4
        @test all(m -> all(isfinite, dense_field(m, experiment.grid, arch)), experiment.omega_field.u)

        # Clockwise vortex below the lid
        Ψ = dense_field(last(experiment.psi_field), experiment.grid, arch)
        @test minimum(Ψ) < 0
        @test maximum(Ψ) <= 0.01 * abs(minimum(Ψ))
        @test argmin(Ψ)[2] > 8
    end

    @testset "vorticity_evolve! argument checks and logging" begin
        arch = TTArchitecture((16, 16))
        sites = siteinds(arch)
        stepper = ExplicitEuler(time_evol_kwargs=TimeEvolArgs(0.0, 0.002, 2), cutoff=1e-12, maxdim=32)
        solver = tt_direct_solver(1e-10)

        # The vorticity must be initialized
        uninitialized = INSVorticityExperiment(interior_grid(arch, (1.0, 1.0)), ScalarField(sites, arch, walls), ScalarField(sites, arch, walls), 100.0, 1.0)
        @test_throws ArgumentError vorticity_evolve!(uninitialized, solver, stepper)

        # The stream function must be empty
        experiment = cavity_experiment(arch)
        push!(experiment.psi_field, zeros_mps(experiment.psi_field.usites))
        @test_throws ArgumentError vorticity_evolve!(experiment, solver, stepper)

        # The logger receives both fields and one residual per checkpoint
        experiment = cavity_experiment(arch)
        mktempdir() do dir
            vorticity_evolve!(experiment, solver, stepper, 1, init_logger(dir), "cavity")
            path = joinpath(dir, "cavity.jld2")
            @test isfile(path)
            jldopen(path, "r") do file
                @test haskey(file, "omega_field")
                @test haskey(file, "psi_field")
                errorlog = file["linsolve_errorlog"]
                @test length(errorlog) == 2
                @test errorlog[1] == 0.0 # The first step starts from rest: no solve
                @test errorlog[2] < 1e-6
            end
        end
    end

    @testset "CFL and Kolmogorov checks" begin
        Re = 100.0
        experiment = cavity_experiment(TTArchitecture((16, 16)); Re=Re)
        h = experiment.grid.dx[1]

        # Second-order stencils: Σ|c₁| = 1 and Σ|c₂| = 4, summed over both axes
        adv_eig, diff_eig = DifferenTT.cfl_spectral_radii(experiment; vmax=2.0)
        @test adv_eig ≈ 2 * 2.0 / h
        @test diff_eig ≈ 2 * 4 / (Re * h^2)

        stable = TimeEvolArgs(0.0, 0.005, 1)
        unstable = TimeEvolArgs(0.0, 1.0, 1)
        @test_logs min_level=Base.CoreLogging.Warn check_cfl(experiment, RK4(time_evol_kwargs=stable))
        @test_logs min_level=Base.CoreLogging.Warn check_cfl(experiment, ExplicitEuler(time_evol_kwargs=stable))
        @test_logs (:warn,) match_mode=:any check_cfl(experiment, RK4(time_evol_kwargs=unstable))
        @test_logs (:warn,) match_mode=:any check_cfl(experiment, ExplicitEuler(time_evol_kwargs=unstable))

        # h = 1/17 does not resolve Re^(-3/4) at Re = 100, but does at Re = 10
        @test_logs (:warn,) match_mode=:any check_kolmogorov_scale(experiment)
        resolved_experiment = cavity_experiment(TTArchitecture((16, 16)); Re=10.0)
        @test_logs (:info, "Kolmogorov scale is resolved") check_kolmogorov_scale(resolved_experiment)
    end
end
