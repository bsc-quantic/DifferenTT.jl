using ITensors
using ITensorMPS
using DifferenTT
using Test
using LinearAlgebra
using Random
using JLD2

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# The dense references contract operators with more indices than the ITensors order-warning threshold
function dense_mpo(A::MPO)
    previous = ITensors.disable_warn_order()
    try
        return contract_mpo(A)
    finally
        ITensors.set_warn_order(previous)
    end
end

# -------------------------------------------------------------------------------------------------
# Dense references of the vorticity-streamfunction scheme with general faces. Fields are stored as F[x, y]
# on the K₁ × K₂ stored nodes. `pad` adds one ghost row per face: a value (number or profile along the face),
# `:mirror` (Neumann, f₀ = f₂) or `:periodic`. Without ghosts, every ghost value is 0.
# -------------------------------------------------------------------------------------------------

function dense_field(mps::MPS, grid::Grid{2}, arch)
    K1, K2 = variable_dims(arch)
    return [DifferenTT.sample(mps, ((i - 1) * grid.dx[1], (j - 1) * grid.dx[2]), grid, arch) for i in 1:K1, j in 1:K2]
end

ghost_row(g, mirror, periodic) = g === :mirror ? mirror : (g === :periodic ? periodic : g)

function pad(F::AbstractMatrix, ghosts=((0.0, 0.0), (0.0, 0.0)))
    P = zeros(size(F, 1) + 2, size(F, 2) + 2)
    P[2:(end-1), 2:(end-1)] .= F
    P[1, 2:(end-1)] .= ghost_row(ghosts[1][1], F[2, :], F[end, :])
    P[end, 2:(end-1)] .= ghost_row(ghosts[1][2], F[end-1, :], F[1, :])
    P[2:(end-1), 1] .= ghost_row(ghosts[2][1], F[:, 2], F[:, end])
    P[2:(end-1), end] .= ghost_row(ghosts[2][2], F[:, end-1], F[:, 1])
    return P
end

dense_dx(P, hx) = (P[3:end, 2:(end-1)] .- P[1:(end-2), 2:(end-1)]) ./ (2hx)
dense_dy(P, hy) = (P[2:(end-1), 3:end] .- P[2:(end-1), 1:(end-2)]) ./ (2hy)

function dense_laplacian(P, hx, hy)
    C = P[2:(end-1), 2:(end-1)]
    return (P[1:(end-2), 2:(end-1)] .+ P[3:end, 2:(end-1)] .- 2C) ./ hx^2 .+
           (P[2:(end-1), 1:(end-2)] .+ P[2:(end-1), 3:end] .- 2C) ./ hy^2
end

face_ghosts(faces, ghost) = ntuple(axis -> (ghost(faces[axis][1], axis, :left), ghost(faces[axis][2], axis, :right)), 2)
closure_ghost(face) = face isa OutflowFace ? :mirror : (face isa PeriodicFace ? :periodic : nothing)

psi_ghost(face, axis, side) = isnothing(closure_ghost(face)) ? face.psi : closure_ghost(face)
homogeneous_ghost(face, axis, side) = something(closure_ghost(face), 0.0)
function flux_ghost(face, axis, side)
    something(closure_ghost(face), face isa InflowFace ? face.normal_velocity .* face.omega : 0.0)
end

# Ghost vorticity of each face. On a no-slip wall: (c₁ψ₁ + c₂ψ₂ - (c₁ + c₂)ψ_w)/h² + s c_U U_t/h, with the sign s
# from ω = ∂ₓv - ∂ᵧu (+1 on the top and left walls)
function omega_ghosts(faces, Ψ, hx, hy; formula)
    c1, c2, cU = DifferenTT.wall_vorticity_coefficients(formula)
    function ghost(face, axis, side)
        isnothing(closure_ghost(face)) || return closure_ghost(face)
        face isa SlipWall && return 0.0
        face isa InflowFace && return face.omega
        h = axis == 1 ? hx : hy
        ψ1, ψ2 = if axis == 1
            (side == :left ? (Ψ[1, :], Ψ[2, :]) : (Ψ[end, :], Ψ[end-1, :]))
        else
            (side == :left ? (Ψ[:, 1], Ψ[:, 2]) : (Ψ[:, end], Ψ[:, end-1]))
        end
        s = ((axis == 2) == (side == :right)) ? 1.0 : -1.0
        return (c1 .* ψ1 .+ c2 .* ψ2 .- (c1 + c2) * face.psi) ./ h^2 .+ s * cU .* face.velocity ./ h
    end
    return face_ghosts(faces, ghost)
end

function dense_full_laplacian(Ω, Ψ, hx, hy, faces::Tuple; formula)
    dense_laplacian(pad(Ω, omega_ghosts(faces, Ψ, hx, hy; formula=formula)), hx, hy)
end
function dense_full_laplacian(Ω, Ψ, hx, hy, U::Real; formula)
    dense_full_laplacian(Ω, Ψ, hx, hy, cavity_faces(U); formula=formula)
end

dense_velocity(Ψ, hx, hy, faces) = (P=pad(Ψ, face_ghosts(faces, psi_ghost)); (dense_dy(P, hy), -dense_dx(P, hx)))

# Conservative advection: ghost fluxes vanish on walls, carry the inflow flux and are first-order upwind on outflows
function dense_advection(u, v, Ω, hx, hy, faces)
    X, Y = u .* Ω, v .* Ω
    ghosts = face_ghosts(faces, flux_ghost)
    ax, ay = dense_dx(pad(X, ghosts), hx), dense_dy(pad(Y, ghosts), hy)
    faces[1][1] isa OutflowFace && (ax[1, :] .= (X[2, :] .- X[1, :]) ./ hx)
    faces[1][2] isa OutflowFace && (ax[end, :] .= (X[end, :] .- X[end-1, :]) ./ hx)
    faces[2][1] isa OutflowFace && (ay[:, 1] .= (Y[:, 2] .- Y[:, 1]) ./ hy)
    faces[2][2] isa OutflowFace && (ay[:, end] .= (Y[:, end] .- Y[:, end-1]) ./ hy)
    return ax .+ ay
end

function dense_transport_rhs(Ω, Ψ, hx, hy, faces::Tuple, Re; formula)
    u, v = dense_velocity(Ψ, hx, hy, faces)
    return -dense_advection(u, v, Ω, hx, hy, faces) .+ dense_full_laplacian(Ω, Ψ, hx, hy, faces; formula=formula) ./ Re
end
function dense_transport_rhs(Ω, Ψ, hx, hy, U::Real, Re; formula)
    dense_transport_rhs(Ω, Ψ, hx, hy, cavity_faces(U), Re; formula=formula)
end

# -Δ acting on vec(F) with Dirichlet ghost walls of value 0, with x the fastest index
function dense_poisson_matrix(K1, K2, hx, hy)
    second_derivative(n, h) = Matrix(SymTridiagonal(fill(-2.0, n), ones(n - 1))) ./ h^2
    return -(
        kron(Matrix(1.0I, K2, K2), second_derivative(K1, hx)) + kron(second_derivative(K2, hy), Matrix(1.0I, K1, K1))
    )
end

# Solution of -Δψ = ω with the face values of ψ (mirrored, unweighted outflow rows; not for singular systems)
function dense_poisson_solve(Ω, hx, hy, faces)
    K1, K2 = size(Ω)
    homogeneous = face_ghosts(faces, homogeneous_ghost)
    A = zeros(K1 * K2, K1 * K2)
    for k in 1:(K1*K2)
        e = zeros(K1 * K2)
        e[k] = 1.0
        A[:, k] .= -vec(dense_laplacian(pad(reshape(e, K1, K2), homogeneous), hx, hy))
    end
    lift = vec(dense_laplacian(pad(zeros(K1, K2), face_ghosts(faces, psi_ghost)), hx, hy))
    return reshape(A \ (vec(Ω) .+ lift), K1, K2)
end

# One-solve-per-step scheme: velocity and wall vorticity frozen over the stages
function dense_evolve(Ω0, nsteps, dt, hx, hy, faces::Tuple, Re; formula, scheme)
    Ω = copy(Ω0)
    Ψ = zero(Ω0)
    for _ in 1:nsteps
        Ψ = dense_poisson_solve(Ω, hx, hy, faces)
        R(W) = dense_transport_rhs(W, Ψ, hx, hy, faces, Re; formula=formula)
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
function dense_evolve(Ω0, nsteps, dt, hx, hy, U::Real, Re; formula, scheme)
    dense_evolve(Ω0, nsteps, dt, hx, hy, cavity_faces(U), Re; formula=formula, scheme=scheme)
end

relerr(a, b) = norm(a .- b) / norm(b)

arch_label(arch::QTTArchitecture) = "QTT $(arch.var_nsites)$(arch.is_interleaved ? " interleaved" : "")"
arch_label(arch::TTArchitecture) = "TT $(arch.var_dims)"

# MPS of the product g(x) f(y) on the stored nodes, for any architecture
function product_mps(field::ScalarField{2}, gx::AbstractVector, fy)
    profile = DifferenTT.face_profile_mps(field, 1, fy)
    diagonal = DifferenTT._onedim_diagonal_operator(field.arch, 1, gx, Array)
    return apply(DifferenTT.ndim_operator_mpo(diagonal, 1, field.usites, field.arch), profile; cutoff=1e-15)
end

# Rotation by 90° counter-clockwise of a field on a square grid: G(x', y') = F(x, y) with (x', y') = (L - y, x)
rotate_ccw(F) = [F[j, size(F, 1)+1-i] for i in axes(F, 1), j in axes(F, 2)]

# Plane Poiseuille flow of a channel of height H with a left inflow: walls at the ghost rows y = 0 and y = H
poiseuille_psi(y, U, H) = U * (2y^2 / H - 4y^3 / (3H^2))
poiseuille_omega(y, U, H) = -4U * (H - 2y) / H^2

function poiseuille_faces(arch; U=1.0, H=1.0)
    K2 = variable_dims(arch)[2]
    h = H / (K2 + 1)
    y = h .* (1:K2)
    ψ = poiseuille_psi.(y, U, H)
    Q = poiseuille_psi(H, U, H)
    # Inflow velocity of the discrete profile (central difference with the wall values), so that the inflow flux
    # matches the advective flux of the interior velocity exactly
    u = (vcat(ψ[2:end], Q) .- vcat(0.0, ψ[1:(end-1)])) ./ (2h)
    inflow = InflowFace(; psi=ψ, omega=poiseuille_omega.(y, U, H), normal_velocity=u)
    return ((inflow, OutflowFace()), (NoSlipWall(; psi=0.0), NoSlipWall(; psi=Q)))
end

@testset "Vorticity-streamfunction" begin
    # The wall conditions shared by the cavity experiments
    wall = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
    walls = BoundaryConditions(wall, wall)
    periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())

    function cavity_experiment(arch; Re=100.0, U=1.0, formula=:jensen, maxdim=256, cutoff=1e-14, bcs=walls)
        sites = siteinds(arch)
        grid = interior_grid(arch, (1.0, 1.0))
        omega_field = ScalarField(zeros_mps(sites), arch, bcs)
        psi_field = ScalarField(sites, arch, bcs)
        return INSVorticityExperiment(
            grid, omega_field, psi_field, Re, U; wall_formula=formula, maxdim=maxdim, cutoff=cutoff
        )
    end

    # Experiment with general faces; `omega(psi_field, grid)` builds the initial vorticity
    function face_experiment(
        arch,
        faces;
        domain=(1.0, 1.0),
        Re=100.0,
        formula=:jensen,
        omega=nothing,
        vmax=nothing,
        maxdim=256,
        cutoff=1e-14,
        bcs=vorticity_boundary_conditions(faces),
    )
        sites = siteinds(arch)
        grid = vorticity_grid(arch, domain, faces)
        psi_field = ScalarField(sites, arch, bcs)
        omega_field = ScalarField(isnothing(omega) ? zeros_mps(sites) : omega(psi_field, grid), arch, bcs)
        return INSVorticityExperiment(
            grid, omega_field, psi_field, Re, faces; wall_formula=formula, maxdim=maxdim, cutoff=cutoff, vmax=vmax
        )
    end
    periodic_faces = ((PeriodicFace(), PeriodicFace()), (PeriodicFace(), PeriodicFace()))

    exact_solver() = GlobalCG(
        updater_kwargs=GlobalUpdaterArgs(
            ishermitian=true,
            isposdef=true,
            rtol=1e-13,
            maxiter=2000,
            maxdim=256,
            cutoff=1e-15,
            orth_maxdim=256,
            orth_cutoff=1e-15,
        ),
    )
    # On a 2-site TT, a single 2-site ALS update is a direct CG solve of the full system
    tt_direct_solver(rtol) = ALS(;
        sweep_kwargs=SweepingArgs(; nsweeps=1, maxdim=256, cutoff=1e-15),
        updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=rtol, maxiter=1000, krylovdim=1),
    )

    qtt_archs = (QTTArchitecture((4, 4)), QTTArchitecture((4, -4)), QTTArchitecture((4, 4); interleaved=true))
    archs = (qtt_archs..., TTArchitecture((16, 16)))

    # interior_grid, entry_mpo and the 1D wall vorticity corrections are tested with their modules
    # (test_grid.jl, test_qttconstructors.jl, test_bcoperators.jl)

    @testset "Constructor checks" begin
        arch = QTTArchitecture((4, 4))
        @test cavity_experiment(arch) isa INSVorticityExperiment{2}
        @test_throws ArgumentError cavity_experiment(arch; formula=:unknown)
        @test_throws ArgumentError cavity_experiment(arch; bcs=BoundaryConditions(periodic, wall))
        @test_throws ArgumentError cavity_experiment(
            arch; bcs=BoundaryConditions(wall, BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0)))
        )

        # Stored settings; an integer lid velocity is stored as Float64
        experiment = cavity_experiment(arch; Re=250, U=2, formula=:thom, maxdim=17, cutoff=1e-9)
        @test experiment.faces == cavity_faces(2)
        @test experiment.vmax === 2.0
        @test experiment.Re == 250
        @test experiment.wall_formula == :thom
        @test experiment.stencil_order == 2
        @test experiment.maxdim == 17
        @test experiment.cutoff == 1e-9

        # Only 2D is supported
        arch_3d = QTTArchitecture((2, 2, 2))
        sites_3d = siteinds(arch_3d)
        walls_3d = BoundaryConditions(wall, wall, wall)
        @test_throws ArgumentError INSVorticityExperiment(
            interior_grid(arch_3d, (1.0, 1.0, 1.0)),
            ScalarField(zeros_mps(sites_3d), arch_3d, walls_3d),
            ScalarField(sites_3d, arch_3d, walls_3d),
            100.0,
            1.0,
        )

        sites = siteinds(arch)
        grid = interior_grid(arch, (1.0, 1.0))
        omega_field = ScalarField(zeros_mps(sites), arch, walls)
        psi_field = ScalarField(sites, arch, walls)
        @test_throws ArgumentError INSVorticityExperiment(grid, omega_field, psi_field, 100.0, 1.0; stencil_order=8)
        @test_throws ArgumentError INSVorticityExperiment(
            grid, omega_field, ScalarField(siteinds(arch), arch, walls), 100.0, 1.0
        )

        # General faces: both fields must have the boundary conditions of the faces
        K1, K2 = variable_dims(arch)
        faces = poiseuille_faces(arch)
        @test face_experiment(arch, faces).faces == faces
        @test_throws ArgumentError face_experiment(arch, faces; bcs=walls) # Dirichlet on the outflow face
        @test_throws ArgumentError face_experiment(arch, cavity_faces(1.0); bcs=vorticity_boundary_conditions(faces))
        mixed_sites = siteinds(arch)
        @test_throws ArgumentError INSVorticityExperiment(
            vorticity_grid(arch, (1.0, 1.0), faces),
            ScalarField(zeros_mps(mixed_sites), arch, walls),
            ScalarField(mixed_sites, arch, vorticity_boundary_conditions(faces)),
            100.0,
            faces,
        )
        # Periodic faces come in pairs
        @test_throws ArgumentError face_experiment(
            arch, ((PeriodicFace(), NoSlipWall()), (NoSlipWall(), NoSlipWall())); bcs=BoundaryConditions(periodic, wall)
        )
        # Profiles have one value per node along the face
        @test_throws ArgumentError face_experiment(
            arch, ((InflowFace(psi=zeros(K2 + 1), normal_velocity=1.0), OutflowFace()), (NoSlipWall(), NoSlipWall()))
        )
        @test_throws ArgumentError face_experiment(
            arch, ((NoSlipWall(), NoSlipWall()), (NoSlipWall(), NoSlipWall(velocity=ones(K2)[1:3])))
        )

        # vmax: the largest wall or inflow speed, 1 (with an @info) if every face is at rest, or the given value
        @test face_experiment(
            arch, ((NoSlipWall(), NoSlipWall()), (NoSlipWall(velocity=-2), NoSlipWall(velocity=1)))
        ).vmax === 2.0
        @test face_experiment(arch, faces).vmax ≈ maximum(faces[1][1].normal_velocity)
        @test_logs (:info, r"vmax") match_mode=:any face_experiment(arch, periodic_faces)
        @test face_experiment(arch, periodic_faces).vmax === 1.0
        @test face_experiment(arch, periodic_faces; vmax=0.5).vmax === 0.5
    end

    @testset "Faces, boundary conditions and grid" begin
        arch = TTArchitecture((16, 16))
        K1, K2 = variable_dims(arch)
        faces = ((InflowFace(psi=0.0, normal_velocity=1.0), OutflowFace()), (PeriodicFace(), PeriodicFace()))
        bcs = vorticity_boundary_conditions(faces)
        @test bcs[1].left isa DirichletBC
        @test bcs[1].right isa NeumannBC
        @test bcs[2].left isa PeriodicBC && bcs[2].right isa PeriodicBC
        slip = vorticity_boundary_conditions(((SlipWall(), NoSlipWall()), (NoSlipWall(), SlipWall(psi=1.0))))
        @test all(i -> slip[i].left isa DirichletBC && slip[i].right isa DirichletBC, 1:2)

        @test cavity_faces(1.5) == ((NoSlipWall(), NoSlipWall()), (NoSlipWall(), NoSlipWall(velocity=1.5)))
        @test cavity_faces(2)[2][2].velocity === 2.0
        @test NoSlipWall(velocity=[1, 2]) == NoSlipWall(velocity=[1.0, 2.0])
        @test NoSlipWall(velocity=1.0) != NoSlipWall(velocity=2.0)

        # One ghost side for inflow + outflow and for periodic axes, two for walls, none for two outflows
        @test all(vorticity_grid(arch, (2.0, 1.0), faces).dx .≈ (2.0 / K1, 1.0 / K2))
        @test vorticity_grid(arch, (1.0, 1.0), cavity_faces(1.0)).dx == interior_grid(arch, (1.0, 1.0)).dx
        @test vorticity_grid(arch, (1.0, 1.0), ((OutflowFace(), OutflowFace()), (SlipWall(), NoSlipWall()))).dx[1] ≈
            1 / (K1 - 1)
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

        A = dense_mpo(cache.poisson_op)
        @test A ≈ A'
        @test eigmin(Symmetric(A)) > 0
        @test A ≈ -dense_mpo(cache.laplacian)

        # The right-hand side of -Δψ = ω is the vorticity itself
        @test DifferenTT.poisson_rhs(experiment, cache, psi) === psi
    end

    @testset "Wall vorticity operator and static wall source" begin
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
            lid = VorticityCache(cavity_experiment(arch; U=1.3, formula=formula)).wall_source
            expected_lid = zeros(K1, K2)
            expected_lid[:, K2] .= cU * 1.3 / hy^3
            @test dense_field(lid, grid, arch) ≈ expected_lid
        end

        # A cavity at rest has no static wall source
        @test isnothing(VorticityCache(cavity_experiment(arch; U=0.0)).wall_source)

        # No wall at all: no operator
        @test isnothing(
            DifferenTT.wall_vorticity_operator(ScalarField(sites, arch, BoundaryConditions(periodic, periodic)), grid)
        )
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

    @testset "Wall vorticity source and transport RHS ($(arch_label(arch)), $formula)" for arch in archs,
        formula in (:jensen, :thom)

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
        full_laplacian = +(apply(cache.omega_laplacian, omega), source; cutoff=1e-15)
        @test relerr(dense_field(full_laplacian, grid, arch), dense_full_laplacian(Ω, Ψ, hx, hy, U; formula=formula)) <
            1e-10

        u, v = streamfunction_velocity(cache, psi)
        rhs = VorticityTransportRHS(experiment, cache, u, v, source)
        tt_rhs = only(rhs((omega,), 0.0, 1))
        @test relerr(dense_field(tt_rhs, grid, arch), dense_transport_rhs(Ω, Ψ, hx, hy, U, Re; formula=formula)) < 1e-8
    end

    @testset "Zero vorticity skips the solve" begin
        experiment = cavity_experiment(QTTArchitecture((4, 4)))
        cache = VorticityCache(experiment)
        psi, residual = poisson_streamfunction(
            experiment, cache, zeros_mps(experiment.psi_field.usites), exact_solver()
        )
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
        one_iteration = GlobalCG(
            updater_kwargs=GlobalUpdaterArgs(
                ishermitian=true,
                isposdef=true,
                rtol=1e-13,
                maxiter=1,
                maxdim=256,
                cutoff=1e-15,
                orth_maxdim=256,
                orth_cutoff=1e-15,
            ),
        )
        rough_psi, rough_residual = poisson_streamfunction(experiment, cache, omega, one_iteration)
        @test rough_residual > 1e-3 * norm(omega)
        @test rough_residual ≈ norm(+(apply(cache.poisson_op, rough_psi; cutoff=1e-15), -1.0 * omega; cutoff=1e-15)) rtol=1e-6

        # The last stream function of the experiment is the initial guess: from the exact solution, CG returns at once
        numiter_store = Ref{Int}(-1)
        counting_solver = GlobalCG(
            updater_kwargs=GlobalUpdaterArgs(
                ishermitian=true,
                isposdef=true,
                rtol=1e-6,
                maxiter=2000,
                maxdim=256,
                cutoff=1e-15,
                orth_maxdim=256,
                orth_cutoff=1e-15,
                numiter_store=numiter_store,
            ),
        )
        poisson_streamfunction(experiment, cache, omega, counting_solver)
        @test numiter_store[] > 0
        push!(experiment.psi_field, psi)
        poisson_streamfunction(experiment, cache, omega, counting_solver)
        @test numiter_store[] == 0
    end

    @testset "vorticity_transport from rest (priming step) ($(arch_label(arch)), $scheme)" for arch in (
            QTTArchitecture((4, 4)), TTArchitecture((16, 16))
        ),
        scheme in (:euler, :rk4)

        Re, U, dt = 100.0, 1.0, 0.01
        experiment = cavity_experiment(arch; Re=Re, U=U)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        zero_field = zeros_mps(experiment.psi_field.usites)
        evol_args = TimeEvolArgs(0.0, dt, 1)
        stepper = if scheme == :rk4
            RK4(; time_evol_kwargs=evol_args, cutoff=1e-14, maxdim=256)
        else
            ExplicitEuler(; time_evol_kwargs=evol_args, cutoff=1e-14, maxdim=256)
        end

        # At rest u = v = 0 and ψ = 0: only the lid drives the flow, and the zero products must not break the step
        omega_1 = DifferenTT.vorticity_transport(experiment, cache, zero_field, zero_field, stepper, 0.0)
        Ω1 = dense_field(omega_1, grid, arch)
        @test all(isfinite, Ω1)
        if scheme == :euler
            @test relerr(Ω1, (dt / Re) .* dense_field(cache.wall_source, grid, arch)) < 1e-12
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

    @testset "End-to-end against the dense scheme ($(arch_label(arch)), $scheme)" for (arch, scheme) in (
        (QTTArchitecture((4, 4)), :euler), (TTArchitecture((16, 16)), :rk4)
    )
        Re, U, dt, nsteps = 100.0, 1.0, 0.01, 3
        experiment = cavity_experiment(arch; Re=Re, U=U)
        kwargs = (; cutoff=1e-14, maxdim=256)
        evol_args = TimeEvolArgs(0.0, dt * nsteps, nsteps)
        stepper = if scheme == :rk4
            RK4(; time_evol_kwargs=evol_args, kwargs...)
        else
            ExplicitEuler(; time_evol_kwargs=evol_args, kwargs...)
        end

        solver = arch isa TTArchitecture ? tt_direct_solver(1e-13) : exact_solver()
        vorticity_evolve!(experiment, solver, stepper)
        @test length(experiment.omega_field) == nsteps + 1
        @test length(experiment.psi_field) == nsteps

        grid = experiment.grid
        Ω, Ψ = dense_evolve(
            zeros(variable_dims(arch)...), nsteps, dt, grid.dx..., U, Re; formula=:jensen, scheme=scheme
        )
        # Truncation level of the randomized tt_mul; a sign or scaling error would be O(1)
        @test relerr(dense_field(last(experiment.omega_field), grid, arch), Ω) < 1e-6
        @test relerr(dense_field(last(experiment.psi_field), grid, arch), Ψ) < 1e-6
    end

    @testset "vorticity_evolve! smoke run" begin
        arch = TTArchitecture((16, 16))
        experiment = cavity_experiment(arch; Re=10.0, maxdim=32, cutoff=1e-12)
        stepper = RK4(time_evol_kwargs=TimeEvolArgs(0.0, 0.3, 30), cutoff=1e-12, maxdim=32)
        @test isnothing(check_cfl(experiment, stepper; vmax=experiment.vmax))

        vorticity_evolve!(experiment, tt_direct_solver(1e-10), stepper, 10)
        # Stored: the initial vorticity and the steps 10, 20 and 30 (the stream function of the same steps)
        @test length(experiment.omega_field) == 4
        @test length(experiment.psi_field) == 3
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
        uninitialized = INSVorticityExperiment(
            interior_grid(arch, (1.0, 1.0)),
            ScalarField(sites, arch, walls),
            ScalarField(sites, arch, walls),
            100.0,
            1.0,
        )
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

    # Regression test: with nsteps - 1 not a multiple of save_every, the evolution also stored the state of step
    # nsteps - 1, and the k-th residual of the log was that of the step of the k-th push, not of the k-th stored
    # stream function. The stored states and residuals must be those of a save_every = 1 run at the steps
    # save_every, 2 save_every, … and nsteps.
    @testset "vorticity_evolve! stores every save_every-th step and the last one" begin
        arch = TTArchitecture((8, 8))
        stepper = ExplicitEuler(time_evol_kwargs=TimeEvolArgs(0.0, 0.005, 5), cutoff=1e-12, maxdim=32)
        function run(save_every)
            experiment = cavity_experiment(arch; Re=10.0, maxdim=32, cutoff=1e-12)
            errorlog = mktempdir() do dir
                Random.seed!(1234) # The advection products are randomized (RSI)
                vorticity_evolve!(experiment, tt_direct_solver(1e-12), stepper, save_every, init_logger(dir), "run")
                jldopen(file -> file["linsolve_errorlog"], joinpath(dir, "run.jld2"))
            end
            return experiment, errorlog
        end
        every, every_log = run(1)
        sparse, sparse_log = run(3)
        @test length(sparse.omega_field) == 3 # steps 0, 3 and 5
        @test length(sparse.psi_field) == length(sparse_log) == 2 # steps 3 and 5
        @test sparse_log ≈ every_log[[3, 5]]
        dense(m) = dense_field(m, every.grid, arch) # The two experiments have different site indices
        for (k, step) in enumerate((3, 5))
            @test dense(sparse.psi_field[k]) ≈ dense(every.psi_field[step]) rtol = 1e-10
        end
        for (k, step) in enumerate((0, 3, 5))
            @test dense(sparse.omega_field[k]) ≈ dense(every.omega_field[step+1]) rtol = 1e-10
        end
        @test_throws ArgumentError vorticity_evolve!(cavity_experiment(arch), tt_direct_solver(1e-12), stepper, 0)
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

    # ---------------------------------------------------------------------------------------------
    # General faces
    # ---------------------------------------------------------------------------------------------

    @testset "Outflow advection closure ($(arch_label(arch)))" for arch in (
        QTTArchitecture((4, 4)), QTTArchitecture((4, -4)), TTArchitecture((16, 16))
    )
        faces = ((OutflowFace(), OutflowFace()), (NoSlipWall(), OutflowFace()))
        grid = vorticity_grid(arch, (1.0, 1.0), faces)
        hx, hy = grid.dx
        field = ScalarField(siteinds(arch), arch, vorticity_boundary_conditions(faces))
        plain = DifferentialOpsScalar(grid, field; stencil_order=2)
        ops = DifferenTT.outflow_advection_closure(plain, field, grid, faces)

        Random.seed!(11)
        f = random_mps(field.usites; linkdims=4)
        F = dense_field(f, grid, arch)
        ghosts = ((:mirror, :mirror), (0.0, :mirror))

        # The mirrored closure zeroes the outflow rows of the first derivative
        Dx_plain = dense_field(apply(plain.d_ops[1], f), grid, arch)
        @test norm(Dx_plain[1, :]) + norm(Dx_plain[end, :]) < 1e-10 * norm(Dx_plain)

        # The closure replaces them with the one-sided difference and leaves the other rows central
        expected_x = dense_dx(pad(F, ghosts), hx)
        expected_x[1, :] .= (F[2, :] .- F[1, :]) ./ hx
        expected_x[end, :] .= (F[end, :] .- F[end-1, :]) ./ hx
        expected_y = dense_dy(pad(F, ghosts), hy)
        expected_y[:, end] .= (F[:, end] .- F[:, end-1]) ./ hy
        @test relerr(dense_field(apply(ops.d_ops[1], f), grid, arch), expected_x) < 1e-10
        @test relerr(dense_field(apply(ops.d_ops[2], f), grid, arch), expected_y) < 1e-10
        @test ops.d2_ops === plain.d2_ops

        # Without outflow faces the operators are returned as they are
        wall_field = ScalarField(siteinds(arch), arch, walls)
        wall_ops = DifferentialOpsScalar(grid, wall_field; stencil_order=2)
        @test DifferenTT.outflow_advection_closure(wall_ops, wall_field, grid, cavity_faces(1.0)) === wall_ops
    end

    # Every face type on both axes, with profiles, moving walls and ψ ≠ 0 on the faces
    function mixed_faces(arch, case)
        K1, K2 = variable_dims(arch)
        Random.seed!(12)
        if case == 1
            return (
                (InflowFace(psi=rand(K2), omega=rand(K2), normal_velocity=1 .+ rand(K2)), OutflowFace()),
                (NoSlipWall(velocity=-0.7, psi=0.3), SlipWall(psi=1.1)),
            )
        elseif case == 2
            return (
                (NoSlipWall(velocity=rand(K2), psi=0.2), NoSlipWall(velocity=-0.5, psi=-0.4)),
                (InflowFace(psi=rand(K1), omega=rand(K1), normal_velocity=1.0), OutflowFace()),
            )
        else
            return (periodic_faces[1], (NoSlipWall(velocity=0.4, psi=-0.2), NoSlipWall(velocity=1.0, psi=0.5)))
        end
    end

    @testset "Boundary terms and transport RHS with general faces ($(arch_label(arch)), case $case, $formula)" for arch in
                                                                                                                   archs,
        case in 1:3,
        formula in (:jensen, :thom)

        Re = 100.0
        faces = mixed_faces(arch, case)
        experiment = face_experiment(arch, faces; Re=Re, formula=formula, domain=(1.0, 1.3))
        cache = VorticityCache(experiment)
        @test !cache.regularize
        grid = experiment.grid
        hx, hy = grid.dx

        Random.seed!(13)
        psi = random_mps(experiment.psi_field.usites; linkdims=4)
        omega = random_mps(experiment.psi_field.usites; linkdims=4)
        Ψ = dense_field(psi, grid, arch)
        Ω = dense_field(omega, grid, arch)

        # Velocity completed with the face values of ψ
        u, v = streamfunction_velocity(cache, psi)
        U, V = dense_velocity(Ψ, hx, hy, faces)
        @test relerr(dense_field(u, grid, arch), U) < 1e-10
        @test relerr(dense_field(v, grid, arch), V) < 1e-10

        # Poisson system: A ψ - b = W (-Δψ - ω), with the face values of ψ and the half weight on outflow rows
        W = ones(size(Ψ))
        faces[1][2] isa OutflowFace && (W[end, :] .*= 0.5)
        faces[2][2] isa OutflowFace && (W[:, end] .*= 0.5)
        residual = +(
            apply(cache.poisson_op, psi), -1.0 * DifferenTT.poisson_rhs(experiment, cache, omega); cutoff=1e-15
        )
        expected_residual = W .* (-dense_laplacian(pad(Ψ, face_ghosts(faces, psi_ghost)), hx, hy) .- Ω)
        # Truncation floor of the sum ω + ℓ at experiment.cutoff (a squared relative error)
        @test relerr(dense_field(residual, grid, arch), expected_residual) < 1e-6

        # Δω completed with the face vorticity, and the whole transport RHS
        source = DifferenTT.wall_vorticity_source(cache, psi)
        full_laplacian = DifferenTT._nothing_sum_handler(apply(cache.omega_laplacian, omega), source; cutoff=1e-15)
        @test relerr(
            dense_field(full_laplacian, grid, arch), dense_full_laplacian(Ω, Ψ, hx, hy, faces; formula=formula)
        ) < 1e-10
        tt_rhs = only(VorticityTransportRHS(experiment, cache, u, v, source)((omega,), 0.0, 1))
        @test relerr(dense_field(tt_rhs, grid, arch), dense_transport_rhs(Ω, Ψ, hx, hy, faces, Re; formula=formula)) <
            1e-8
    end

    @testset "Weighted Poisson operator with an outflow ($(arch_label(arch)))" for arch in (
        QTTArchitecture((4, 4)), QTTArchitecture((4, -4)), TTArchitecture((16, 16))
    )
        cache = VorticityCache(face_experiment(arch, poiseuille_faces(arch); domain=(2.0, 1.0)))
        A = dense_mpo(cache.poisson_op)
        @test norm(A - A') < 1e-10 * norm(A)
        @test eigmin(Symmetric(A)) > 0
        # The diffusion of ω is not weighted
        @test !(dense_mpo(cache.omega_laplacian) ≈ dense_mpo(cache.laplacian))
    end

    @testset "Regularized Poisson system, doubly periodic ($(arch_label(arch)))" for arch in (
        QTTArchitecture((4, 4)), TTArchitecture((16, 16))
    )
        experiment = face_experiment(arch, periodic_faces; vmax=1.0)
        cache = VorticityCache(experiment)
        @test cache.regularize
        grid = experiment.grid
        sites = experiment.psi_field.usites

        A = dense_mpo(cache.poisson_op)
        @test A ≈ A'
        @test eigmin(Symmetric(A)) > 0
        # The pin c 𝟏𝟏ᵀ with c = 1/(number of nodes) gives A 𝟏 = 𝟏
        @test dense_field(apply(cache.poisson_op, ones_mps(sites, 1)), grid, arch) ≈ ones(variable_dims(arch)...)
        # The mean of the right-hand side is removed
        @test abs(DifferenTT.integrate(DifferenTT.poisson_rhs(experiment, cache, ones_mps(sites, 1)))) < 1e-12

        # Discrete eigenfunction: h = L/n, so the samples are exactly periodic
        x, y = grid.points
        λ_d = sum((2 - 2cos(2π * h)) / h^2 for h in grid.dx)
        psi = product_mps(experiment.psi_field, sin.(2π .* x), sin.(2π .* y))
        solver = arch isa TTArchitecture ? tt_direct_solver(1e-13) : exact_solver()
        solution, _ = poisson_streamfunction(experiment, cache, λ_d * psi, solver)
        @test relerr(dense_field(solution, grid, arch), dense_field(psi, grid, arch)) < 1e-8
    end

    @testset "Rotation equivariance of the moving walls" begin
        arch = TTArchitecture((16, 16))
        Re, U, dt, nsteps = 100.0, 1.0, 0.01, 3
        lid(axis, side, velocity) = ntuple(
            a -> ntuple(
                k -> (a == axis && k == (side == :left ? 1 : 2)) ? NoSlipWall(velocity=velocity) : NoSlipWall(), 2
            ),
            2,
        )
        # Each configuration is the previous one rotated by 90° counter-clockwise
        configurations = (lid(2, :right, U), lid(1, :left, U), lid(2, :left, -U), lid(1, :right, -U))
        @test configurations[1] == cavity_faces(U)

        results = map(configurations) do faces
            experiment = face_experiment(arch, faces; Re=Re)
            stepper = ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, dt * nsteps, nsteps), cutoff=1e-14, maxdim=256)
            vorticity_evolve!(experiment, tt_direct_solver(1e-12), stepper)
            return dense_field(last(experiment.omega_field), experiment.grid, arch),
            dense_field(last(experiment.psi_field), experiment.grid, arch)
        end
        # Truncation level of the TT arithmetic; a wrong sign of a moving wall would be O(1)
        for k in 2:length(results)
            @test relerr(results[k][1], rotate_ccw(results[k-1][1])) < 1e-6
            @test relerr(results[k][2], rotate_ccw(results[k-1][2])) < 1e-6
        end
    end

    @testset "Doubly-driven cavity" begin
        arch = TTArchitecture((16, 16))
        Re, U, dt, nsteps = 100.0, 1.0, 0.01, 3
        faces = ((NoSlipWall(), NoSlipWall()), (NoSlipWall(velocity=(-U)), NoSlipWall(velocity=U)))
        experiment = face_experiment(arch, faces; Re=Re)
        stepper = ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, dt * nsteps, nsteps), cutoff=1e-14, maxdim=256)
        vorticity_evolve!(experiment, tt_direct_solver(1e-12), stepper)
        grid = experiment.grid
        Ω = dense_field(last(experiment.omega_field), grid, arch)
        Ψ = dense_field(last(experiment.psi_field), grid, arch)

        Ω_ref, Ψ_ref = dense_evolve(
            zeros(variable_dims(arch)...), nsteps, dt, grid.dx..., faces, Re; formula=:jensen, scheme=:euler
        )
        @test relerr(Ω, Ω_ref) < 1e-6
        @test relerr(Ψ, Ψ_ref) < 1e-6
        # Symmetric under a rotation by 180°
        @test relerr(Ω, rotate_ccw(rotate_ccw(Ω))) < 1e-6
        @test relerr(Ψ, rotate_ccw(rotate_ccw(Ψ))) < 1e-6
    end

    @testset "Poiseuille fixed point ($(arch_label(arch)))" for arch in
                                                                (QTTArchitecture((4, 4)), TTArchitecture((16, 16)))
        U, H, Re = 1.0, 1.0, 100.0
        faces = poiseuille_faces(arch; U=U, H=H)
        experiment = face_experiment(arch, faces; domain=(2.0, H), Re=Re)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        y = grid.dx[2] .* (1:variable_dims(arch)[2])
        psi_exact = DifferenTT.face_profile_mps(experiment.psi_field, 1, poiseuille_psi.(y, U, H))
        omega_exact = DifferenTT.face_profile_mps(experiment.psi_field, 1, poiseuille_omega.(y, U, H))

        # The 3-point stencil is exact on the cubic ψ, and the inflow, wall and (weighted) outflow rows are exact too
        solver = arch isa TTArchitecture ? tt_direct_solver(1e-13) : exact_solver()
        psi, _ = poisson_streamfunction(experiment, cache, omega_exact, solver)
        # Truncation floor of the right-hand side; the discretization error would be O(h²) ≈ 1e-3
        @test relerr(dense_field(psi, grid, arch), dense_field(psi_exact, grid, arch)) < 1e-6

        # Jensen's formula is exact on the cubic ψ, v vanishes and the inflow flux cancels the advection of the uniform uω
        u, v = streamfunction_velocity(cache, psi_exact)
        @test norm(v) < 1e-12 * norm(u)
        source = DifferenTT.wall_vorticity_source(cache, psi_exact)
        rhs = only(VorticityTransportRHS(experiment, cache, u, v, source)((omega_exact,), 0.0, 1))
        @test norm(rhs) < 1e-10 * norm(apply(cache.omega_laplacian, omega_exact)) / Re
    end

    @testset "Couette fixed point ($formula)" for formula in (:jensen, :thom)
        arch = TTArchitecture((16, 16))
        U, H, Re = 1.0, 1.0, 100.0
        faces = (periodic_faces[1], (NoSlipWall(psi=0.0), NoSlipWall(velocity=U, psi=U * H / 2)))
        experiment = face_experiment(arch, faces; domain=(1.0, H), Re=Re, formula=formula)
        cache = VorticityCache(experiment)
        grid = experiment.grid
        y = grid.dx[2] .* (1:variable_dims(arch)[2])
        psi_exact = DifferenTT.face_profile_mps(experiment.psi_field, 1, U .* y .^ 2 ./ (2H))
        omega_exact = DifferenTT.face_profile_mps(experiment.psi_field, 1, -U / H)

        psi, _ = poisson_streamfunction(experiment, cache, omega_exact, tt_direct_solver(1e-13))
        @test relerr(dense_field(psi, grid, arch), dense_field(psi_exact, grid, arch)) < 1e-8
        u, v = streamfunction_velocity(cache, psi_exact)
        @test dense_field(u, grid, arch) ≈ repeat((U / H .* y)', variable_dims(arch)[1])
        # Both wall formulas are exact on the quadratic ψ, with the moving wall and ψ ≠ 0 on the walls
        source = DifferenTT.wall_vorticity_source(cache, psi_exact)
        rhs = only(VorticityTransportRHS(experiment, cache, u, v, source)((omega_exact,), 0.0, 1))
        @test norm(rhs) < 1e-10 * norm(apply(cache.omega_laplacian, omega_exact)) / Re
    end

    @testset "Slip channel" begin
        arch = TTArchitecture((16, 16))
        U, H = 1.0, 1.0
        faces = (periodic_faces[1], (SlipWall(psi=0.0), SlipWall(psi=U * H)))
        experiment = face_experiment(arch, faces; domain=(1.0, H))
        cache = VorticityCache(experiment)
        grid = experiment.grid
        @test isnothing(cache.wall_op)
        @test isnothing(cache.wall_source)

        # From rest, the Poisson solve gives the uniform flow ψ = U y
        zero_field = zeros_mps(experiment.psi_field.usites)
        psi, _ = poisson_streamfunction(experiment, cache, zero_field, tt_direct_solver(1e-13))
        y = grid.dx[2] .* (1:variable_dims(arch)[2])
        @test relerr(dense_field(psi, grid, arch), repeat((U .* y)', variable_dims(arch)[1])) < 1e-8
        u, v = streamfunction_velocity(cache, psi)
        @test dense_field(u, grid, arch) ≈ fill(U, variable_dims(arch)...)
        @test norm(v) < 1e-10

        # ω = 0 is a fixed point
        rhs = only(
            VorticityTransportRHS(experiment, cache, u, v, DifferenTT.wall_vorticity_source(cache, psi))(
                (zero_field,), 0.0, 1
            ),
        )
        @test norm(rhs) < 1e-12
    end

    @testset "Taylor-Green decay (doubly periodic)" begin
        arch = TTArchitecture((32, 32))
        Re, dt, nsteps = 10.0, 0.002, 5
        omega0(psi_field, grid) = product_mps(psi_field, sin.(2π .* grid.points[1]), sin.(2π .* grid.points[2]))
        experiment = face_experiment(arch, periodic_faces; Re=Re, omega=omega0, vmax=1.0)
        grid = experiment.grid
        # Discrete eigenfunction: with hₓ = hᵧ the central conservative advection of this mode vanishes exactly,
        # so each explicit Euler step only multiplies it by 1 - dt λ_d / Re
        λ_d = sum((2 - 2cos(2π * h)) / h^2 for h in grid.dx)
        Ω0 = dense_field(last(experiment.omega_field), grid, arch)

        stepper = ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, dt * nsteps, nsteps), cutoff=1e-14, maxdim=256)
        vorticity_evolve!(experiment, tt_direct_solver(1e-13), stepper)
        decay = 1 - dt * λ_d / Re
        @test relerr(dense_field(last(experiment.omega_field), grid, arch), decay^nsteps .* Ω0) < 1e-6
        # ψ of the last step solves -Δψ = ω at its start
        @test relerr(dense_field(last(experiment.psi_field), grid, arch), (decay^(nsteps - 1) / λ_d) .* Ω0) < 1e-6
    end

    @testset "Poiseuille end-to-end against the dense scheme" begin
        arch = TTArchitecture((16, 16))
        Re, dt, nsteps = 100.0, 0.01, 3
        faces = poiseuille_faces(arch)
        experiment = face_experiment(arch, faces; domain=(2.0, 1.0), Re=Re)
        stepper = ExplicitEuler(; time_evol_kwargs=TimeEvolArgs(0.0, dt * nsteps, nsteps), cutoff=1e-14, maxdim=256)
        vorticity_evolve!(experiment, tt_direct_solver(1e-13), stepper)
        @test length(experiment.omega_field) == nsteps + 1
        @test length(experiment.psi_field) == nsteps

        # From rest, the first solve gives the potential flow of the face values
        grid = experiment.grid
        Ω, Ψ = dense_evolve(
            zeros(variable_dims(arch)...), nsteps, dt, grid.dx..., faces, Re; formula=:jensen, scheme=:euler
        )
        @test relerr(
            dense_field(experiment.psi_field[1], grid, arch),
            dense_poisson_solve(zeros(variable_dims(arch)...), grid.dx..., faces),
        ) < 1e-8
        @test relerr(dense_field(last(experiment.omega_field), grid, arch), Ω) < 1e-6
        @test relerr(dense_field(last(experiment.psi_field), grid, arch), Ψ) < 1e-6
    end
end
