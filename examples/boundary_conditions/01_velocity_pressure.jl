# # [Velocity–pressure formulation](@id bc_velocity_pressure)
#
# In this tutorial we explain how every boundary condition and every boundary correction of the velocity–pressure
# (primitive-variable) formulation is implemented: the closures of the finite-difference operators of
# [`DifferentialOpsScalar`](@ref) and [`DifferentialOps`](@ref), the Laplacians [`kronecker_laplacian`](@ref) and
# [`symmetrized_laplacian`](@ref), and the pressure system of the Chorin projection ([`ChorinCache`](@ref),
# [`poisson_rhs`](@ref)).
#
# The whole construction relies on a single idea. Every operator of DifferenTT is the interior central stencil of order
# ``q``, completed by values at nodes that are not stored, the ghost nodes. A boundary condition is a rule that gives
# those ghost values. When the rule involves the unknowns, substituting it changes the operator, i.e. a few of its rows
# next to the boundary; when it involves only data, it adds a known vector, a lift, to the right-hand side. Each section
# below is one instance of this idea, and says which of the two cases it is.
#
# We proceed as follows. We first fix the grid and the stencils (Sec. 1), and then treat the closures one by one:
# periodic wrapping (Sec. 2), truncation at Dirichlet and open sides (Sec. 3), the lift of Dirichlet data (Sec. 4), the
# boundary rows of the divergence (Sec. 5), the Neumann mirror (Sec. 6) and the weight that makes it symmetric (Sec. 7).
# We then turn to singular systems (Sec. 8), and finally assemble the pressure system of the channel of Sec. 5.2 of
# Torrente et al., *Factorized approximate inverse preconditioners in the tensor train format*, term by term (Sec. 9).
# Each closure is printed on a tiny grid and checked against a dense matrix built by hand from the ghost-node rule; all
# the checks run in this tutorial, and their results are printed.
#
# Several of the functions we call are internal (they are not exported, and some are not even `public`), such as
# `onedim_differential_operator`, `neumann_correction_matrix` or `∇p_correction`. We call them qualified,
# `DifferenTT.name`, and show them only to expose the construction; a user builds the operators of a field with the
# exported functions named in each section.

using DifferenTT
using ITensors, ITensorMPS
using LinearAlgebra
using Plots
using Printf
using StableRNGs

rng = StableRNG(1729);

# The dense checks contract MPOs of up to 8 sites into matrices, i.e. ITensors with 16 indices, for which ITensors warns
# by default; we raise the threshold.

ITensors.set_warn_order(18);

# Every dense check compares the library with a reference that is written term by term from the ghost-node rules,
# without calling the code under test. For one dimension, `ghost_operator` places the coefficient ``c_k`` of row ``i``
# on column ``i + k``, and when that column is a ghost node it drops the coefficient (`:truncate`), moves it to the
# mirror node (`:mirror`) or wraps it around (`:wrap`).

function ghost_operator(order, n, left, right; q)
    D = zeros(n, n)
    for i in 1:n, (k, c) in DifferenTT.derivative_coefficients(order; stencil_order=q)
        j = i + k
        side = j < 1 ? left : (j > n ? right : :stored)
        if side == :stored
            D[i, j] += c
        elseif side == :mirror
            D[i, j < 1 ? 2 - j : 2n - j] += c
        elseif side == :wrap
            D[i, mod1(j, n)] += c
        end
    end
    return D
end;

# The eighth-order coefficients are fractions with the common denominators 840 (first derivative) and 5040 (second
# derivative), so we print the operators multiplied by these factors, as integers.

scale(order, q) = Dict((1, 2) => 2, (1, 8) => 840, (2, 2) => 1, (2, 8) => 5040)[(order, q)]
as_integers(D, order, q) = round.(Int, scale(order, q) .* Matrix(D));

# ## 1. Setting: the grid, the stencils and the interior operator
#
# The velocity–pressure formulation stores the boundary nodes. The [`Grid`](@ref) of an axis of length ``L`` with ``n``
# nodes therefore places one node at each end, with the spacing ``h = L/(n - 1)``, and we number its nodes from 0, so
# that node ``u_j`` sits at ``x_j = j h``, ``j = 0, \dots, n - 1``. The ghost nodes are ``u_{-1}, u_{-2}, \dots`` beyond
# the left end and ``u_n, u_{n+1}, \dots`` beyond the right end, at the same spacing. (The matrices that Julia prints
# below are indexed from 1, so their row ``i`` is node ``u_{i-1}``.)

grid16 = Grid(QTTArchitecture((4,)), 1.0)
grid16.dx[1], first(grid16.points[1]), last(grid16.points[1])

# The derivatives are central finite differences of order ``q``: [`derivative_coefficients`](@ref) returns the
# coefficients ``c_k``, ``|k| \le q/2``, such that
#
# ```math
# h^m \, u^{(m)}(x_j) \approx \sum_{k=-q/2}^{q/2} c_k \, u_{j+k},
# ```
#
# where ``m`` is the derivative order. The library provides ``q = 2`` and ``q = 8``, and returns the coefficients up to
# ``|k| = 4`` in both cases (zero beyond ``q/2``). We print them for ``k = -4, \dots, 4``, the first derivative in the
# first two rows and the second derivative in the last two, as exact fractions.

coefficient_table = [
    rationalize(DifferenTT.derivative_coefficients(m; stencil_order=q)[k]) for
    (m, q) in ((1, 2), (1, 8), (2, 2), (2, 8)), k in -4:4
]

# Away from the boundaries, the operator is the band Toeplitz matrix ``D_{ij} = c_{j-i}``. In the QTT format, a shift by
# a fixed offset has an exact MPO of bond dimension 2, since adding the offset to a binary index carries one bit from
# each site to the next (Kazeev & Khoromskij, 2012). [`toeplitz_mpo`](@ref) builds the whole band at once, with bond
# dimension 3, except for the last bond, which has dimension 9 when the coefficients reach the offsets ``\pm 4``. Since
# `derivative_coefficients` returns the offsets up to ``\pm 4`` for both stencil orders, the operators of the library
# have this last bond of 9 also for ``q = 2`` (with zero coefficients beyond ``\pm 1``), whatever the number of sites.
# We list the bond dimensions of the second-derivative band on ``2^4``, ``2^7`` and ``2^{10}`` nodes.

truncate_bc = BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0))
function band_bonds(N, q)
    linkdims(
        DifferenTT.toeplitz_mpo(siteinds(2, N), DifferenTT.derivative_coefficients(2; stencil_order=q), truncate_bc)
    )
end
(q2=[band_bonds(N, 2) for N in (4, 7, 10)], q8=[band_bonds(N, 8) for N in (4, 7, 10)])

# As a first dense check, the band of the MPO on ``2^4`` nodes, contracted to a matrix, is the Toeplitz matrix with its
# entries outside the matrix dropped, which `ghost_operator` builds with `:truncate` on both sides.

toeplitz_check = [
    maximum(
        abs,
        contract_mpo(
            DifferenTT.toeplitz_mpo(siteinds(2, 4), DifferenTT.derivative_coefficients(2; stencil_order=q), truncate_bc)
        ) - ghost_operator(2, 16, :truncate, :truncate; q),
    ) for q in (2, 8)
]

# All the closures of this tutorial complete this band on its first and last ``q/2`` rows, whose stencils reach the
# ghost nodes. The figure below shows the ghost nodes of an axis of 8 nodes for ``q = 8``, and what each boundary
# condition puts on them; the next sections derive each row of the figure.

subscript(j) = (j < 0 ? "₋" : "") * join(Char(0x2080 + d) for d in reverse(digits(abs(j))))

function ghost_diagram(n)
    stored, ghosts = 0:(n-1), [-4:-1; n:(n+3)]
    fig = plot(;
        size=(1000, 400),
        xlims=(-5, n + 4),
        ylims=(0.3, 3.9),
        yticks=([1, 2, 3], ["Neumann", "Dirichlet, open", "periodic"]),
        xticks=(-4:(n+3), string.(-4:(n+3))),
        xlabel="x / h",
        legend=:none,
        grid=false,
        bottom_margin=6Plots.mm,
        title="Ghost nodes (open circles) of an axis of $n stored nodes, q = 8",
    )
    marker(x, row; kwargs...) = plot!(fig, [x, x], [row - 0.2, row + 0.2]; linewidth=1.5, kwargs...)
    for row in 1:3
        scatter!(fig, stored, fill(row, n); color=:black, markersize=6)
        scatter!(fig, ghosts, fill(row, 8); color=:white, markerstrokecolor=:gray, markersize=6)
        for j in stored
            annotate!(fig, j, row + 0.28, text("u" * subscript(j), 8, :black))
        end
    end
    ## Periodic: the ghost node is the image node; with the periodic flag, node n is the image of node 0
    for j in ghosts
        annotate!(fig, j, 3.28, text("u" * subscript(mod(j, n)), 8, :royalblue))
    end
    marker(n, 3; color=:royalblue, linestyle=:dot)
    annotate!(fig, n, 2.68, text("L = nh (period)", 8, :royalblue))
    ## Dirichlet and open: the operator drops the ghost values (zero); a Dirichlet lift fills them with g
    for j in ghosts
        annotate!(fig, j, 2.28, text("g", 9, :firebrick))
    end
    scatter!(fig, [-1, n], [2, 2]; markershape=:diamond, color=:firebrick, markersize=8)
    annotate!(fig, -1, 1.68, text("g at x = -h", 8, :firebrick))
    annotate!(fig, n, 1.68, text("g at x = L + h", 8, :firebrick))
    annotate!(fig, (n - 1) / 2, 1.68, text("operator: 0 on every ghost; lift: g on every ghost", 8, :firebrick))
    ## Neumann: the ghost values mirror the stored ones about the boundary node, u₋ₖ = uₖ
    for k in 1:4
        annotate!(fig, -k, 1.28, text("u" * subscript(k), 8, :darkgreen))
        annotate!(fig, n - 1 + k, 1.28, text("u" * subscript(n - 1 - k), 8, :darkgreen))
    end
    marker(0, 1; color=:darkgreen, linestyle=:dash)
    marker(n - 1, 1; color=:darkgreen, linestyle=:dash)
    annotate!(fig, 0, 0.68, text("boundary at x = 0", 8, :darkgreen))
    annotate!(fig, n - 1, 0.68, text("boundary at x = L", 8, :darkgreen))
    return fig
end

ghost_diagram(8)

# The diamonds mark where a Dirichlet value is located: on the first ghost node, one spacing beyond the last stored node
# (Sec. 4), i.e. at ``x = -h`` and ``x = L + h`` on the [`Grid`](@ref), whose stored nodes span ``[0, L]`` with
# ``L = (n - 1) h``. A Neumann boundary, in contrast, lies on the stored node itself (dashed lines, Sec. 6), and the
# period of a periodic axis is ``n h`` (dotted line, Sec. 2), which is why a periodic axis needs a different grid.
#
# ## 2. Periodic sides: the ghost node is the image node
#
# On a periodic axis the field repeats with the period of the axis, so no information is missing: a ghost node is the
# image of a stored node, ``u_{j} = u_{j \bmod n}``. Substituting this rule involves only the unknowns, so it changes
# the operator and adds nothing to the right-hand side: the band wraps around and the operator becomes the circulant
# matrix ``D_{ij} = \sum_{k \equiv j - i \ (\mathrm{mod}\ n)} c_k``. The sum matters on short axes: on 8 nodes with
# ``q = 8``, the offsets ``-4`` and ``4`` land on the same node and their coefficients add up. We print the
# second-derivative operator of 8 nodes for ``q = 2`` and ``q = 8``, scaled as integers.

periodic = BoundaryCond1d(PeriodicBC(), PeriodicBC())
arch8 = TTArchitecture((8,))
as_integers(DifferenTT.onedim_differential_operator(2, arch8, 1, periodic; stencil_order=2), 2, 2)

#-

as_integers(DifferenTT.onedim_differential_operator(2, arch8, 1, periodic; stencil_order=8), 2, 8)

# In the corners of the second matrix, ``-18 = 2 \times (-9)`` is the sum of ``c_{-4}`` and ``c_4``. The function
# [`onedim_differential_operator`](@ref) returns a dense matrix for a [`TTArchitecture`](@ref) and an MPO for a
# [`QTTArchitecture`](@ref); in QTT, [`toeplitz_mpo`](@ref) builds the circulant directly, by replacing the shift of the
# first site with its cyclic version, so the bond dimensions are those of Sec. 1. The QTT operator needs at least 4
# sites, so we check it on ``2^4`` nodes. Here, and in all the following sections, the dense check runs over both
# derivative orders and both stencil orders, and prints the largest entry of the difference with the hand reference.

arch12, qtt16 = TTArchitecture((12,)), QTTArchitecture((4,))
function closure_errors(bc, left, right)
    return maximum(
        max(
            maximum(
                abs,
                DifferenTT.onedim_differential_operator(m, arch12, 1, bc; stencil_order=q) -
                ghost_operator(m, 12, left, right; q),
            ),
            maximum(
                abs,
                contract_mpo(DifferenTT.onedim_differential_operator(m, qtt16, 1, bc; stencil_order=q)) -
                ghost_operator(m, 16, left, right; q),
            ),
        ) for m in (1, 2), q in (2, 8)
    )
end
closure_errors(periodic, :wrap, :wrap)

# The circulant has the period ``n h``, which equals the length ``L`` of the axis only if the node at ``L`` is not
# stored, since it is the image of the node at 0. The plain `Grid` stores it, ``h = L/(n - 1)``, so the wrapped stencil
# has the period ``L + h`` and is inconsistent at the seam. A periodic axis therefore needs `Grid(arch, L; periodic)`,
# whose spacing is ``h = L/n``. The flags can be taken from the boundary conditions with `DifferenTT.periodic_axes` (not
# exported in this version). We apply the wrapped second derivative of ``q = 8`` on ``2^6`` nodes to
# ``u(x) = \sin(2\pi x / L)`` on both grids, and print the spacing and the largest relative error against
# ``u'' = -(2\pi/L)^2 u``.

L = 1.0
arch64 = QTTArchitecture((6,))
D2_wrapped = contract_mpo(DifferenTT.onedim_differential_operator(2, arch64, 1, periodic; stencil_order=8))
function wrapped_error(grid)
    u = sin.(2π .* grid.points[1] ./ L)
    return maximum(abs, D2_wrapped * u ./ grid.dx[1]^2 .+ (2π / L)^2 .* u) / (2π / L)^2
end
flagged = Grid(arch64, (L,); periodic=DifferenTT.periodic_axes(BoundaryConditions(periodic)))
plain = Grid(arch64, L)
(flagged=(flagged.dx[1], wrapped_error(flagged)), plain=(plain.dx[1], wrapped_error(plain)))

# With the flag, the error is that of the eighth-order stencil, ``3 \cdot 10^{-12}``; without it, the stencil is
# inconsistent near the seam, and the relative error is 14. The Chorin runs of the paper used the plain grid, i.e. the
# period ``L + h`` (a known convention of v0.1, kept so that those runs can be reproduced); in this tutorial every
# periodic axis is built with the flag.
#
# ## 3. Dirichlet and open sides: truncation
#
# At a non-periodic side, the ghost values are not given by the unknowns, and the operators of the library do not
# extrapolate them: they truncate the stencil, i.e. they drop the entries that fall outside the grid, which amounts to
# taking the ghost values as zero. The rows that change are those within ``q/2`` of the side, which
# `DifferenTT.stencil_rows` returns for the left side (the right side has their mirror images).

DifferenTT.stencil_rows(2), DifferenTT.stencil_rows(8)

# We print the truncated second derivative on 12 nodes for ``q = 8``, scaled as integers: its first and last four rows
# miss the coefficients of the ghost nodes they reach, and every other row is the full stencil.

as_integers(DifferenTT.onedim_differential_operator(2, arch12, 1, truncate_bc; stencil_order=8), 2, 8)

# Both [`DirichletBC`](@ref) and [`OpenBC`](@ref) give this truncated operator, and the boundary value of a
# `DirichletBC` does not enter it. The dense check covers both, against `ghost_operator` with `:truncate`.

open_bc = BoundaryCond1d(OpenBC(), OpenBC())
(dirichlet=closure_errors(truncate_bc, :truncate, :truncate), open=closure_errors(open_bc, :truncate, :truncate))

# Since a zero ghost value is not a consistent extrapolation of a smooth field, the truncated rows are not consistent
# approximations of the derivatives, and the two boundary types differ in what is done about it. On a `DirichletBC`
# side, the lift of Sec. 4 restores the dropped coefficients with the boundary value on the right-hand side of the
# implicit systems, and the projector of Sec. 5 removes the truncated rows from the explicit time steps and from the
# divergence. On an `OpenBC` side nothing else acts: no lift is added, and its rows are evolved by the explicit steps
# with the truncated stencils, so an `OpenBC` side imposes in practice zero ghost values on every operator.
# `DifferenTT.onedim_boundary_nullifier`, the projector of Sec. 5, returns `nothing` for it.

isnothing(DifferenTT.onedim_boundary_nullifier(arch12, 1, open_bc; stencil_order=8))

# ## 4. Non-homogeneous Dirichlet data: the lift
#
# The truncation of Sec. 3 took the ghost values as zero. When the ghost nodes beyond a Dirichlet side hold a known
# value ``g`` instead, the full stencil differs from the truncated one by the dropped coefficients times ``g``, and
# since this difference involves only data, it moves to the right-hand side as a lift. On the left side, row ``i``
# (counted from 1) reaches the ghost nodes ``u_{-1}, \dots, u_{-(q/2 - i + 1)}`` with the coefficients
# ``c_{-i}, \dots, c_{-q/2}``, so that, when every ghost node holds ``g``, the missing contribution is
#
# ```math
# \ell_i = \frac{g}{h^m} \sum_{k=-q/2}^{-i} c_k, \qquad i = 1, \dots, q/2,
# ```
#
# where ``m`` is the derivative order. `DifferenTT.onedim_left_dirichlet_correction` stores these sums as the product of
# an upper triangular Toeplitz block ``C_L``, of first row ``(c_{-4}, c_{-3}, c_{-2}, c_{-1})``, with the all-ones
# vector: ``(C_L \mathbf{1})_i = \sum_{k=-4}^{-i} c_k``. We print the corner block of the second derivative for
# ``q = 2`` and ``q = 8``, scaled as integers.

left_block(q) = DifferenTT.onedim_left_dirichlet_correction(2, arch12, 1; stencil_order=q)[1:4, 1:4]
as_integers(left_block(2), 2, 2)

#-

as_integers(left_block(8), 2, 8)

# Note that the columns of ``C_L`` are not ghost nodes: the block is only meaningful applied to a vector that is
# constant along the axis, which is how the library uses it. For ``q = 2``, only row 1 receives ``c_{-1} g``. The block
# is `DifferenTT.corner_pick_matrix` of the four coefficients; in QTT, `DifferenTT.corner_pick_mpo` places it in the
# corner of an MPO of bond dimension at most 2, which on ``2^4`` nodes contracts to the zero-padded block:

left_coefficients(q) = [DifferenTT.derivative_coefficients(2; stencil_order=q)[k] for k in -4:-1]
corner_qtt = contract_mpo(DifferenTT.corner_pick_mpo(left_coefficients(8), siteinds(2, 4), :upper))
corner_dense = zeros(16, 16)
corner_dense[1:4, 1:4] = DifferenTT.corner_pick_matrix(left_coefficients(8), :upper)
maximum(abs, corner_qtt - corner_dense)

# The right side is the mirror image, `DifferenTT.onedim_right_dirichlet_correction`, a lower triangular block in the
# bottom-right corner. The dense check of the lift compares, on each row, the truncated operator applied to the stored
# values plus the lifts of both sides with the full stencil applied to the vector extended with the ghost values. The
# reference `ghost_stencil` applies the stencil along one axis of an array of any dimension, with a ghost rule for each
# side: a value (a number, or one value per node of the other axis), `:mirror` or `:wrap`.

function ghost_stencil(F, order, axis, q, h, left, right)
    out = zeros(size(F))
    n = size(F, axis)
    for idx in CartesianIndices(F), (k, c) in DifferenTT.derivative_coefficients(order; stencil_order=q)
        j = idx[axis] + k
        rule = j < 1 ? left : (j > n ? right : :stored)
        if rule isa Symbol
            jj = if rule == :stored
                j
            elseif rule == :mirror
                (j < 1 ? 2 - j : 2n - j)
            else
                mod1(j, n)
            end
            out[idx] += c * F[CartesianIndex(Base.setindex(Tuple(idx), jj, axis))]
        else
            out[idx] += c * (rule isa Number ? rule : rule[idx[3-axis]])
        end
    end
    return out ./ h^order
end;

# We take the values ``g_L = 0.7`` and ``g_R = -1.3``, a random vector of stored values, and ``h = 1``; the check runs
# over both derivative orders and stencil orders, in TT (12 nodes) and QTT (``2^4`` nodes).

g_left, g_right = 0.7, -1.3
function lift_error(arch, n, q, m)
    to_matrix(op) = op isa MPO ? contract_mpo(op) : op
    D = to_matrix(DifferenTT.onedim_differential_operator(m, arch, 1, truncate_bc; stencil_order=q))
    C_L = to_matrix(DifferenTT.onedim_left_dirichlet_correction(m, arch, 1; stencil_order=q))
    C_R = to_matrix(DifferenTT.onedim_right_dirichlet_correction(m, arch, 1; stencil_order=q))
    u = randn(rng, n)
    lifted = D * u + g_left * C_L * ones(n) + g_right * C_R * ones(n)
    return maximum(abs, lifted - ghost_stencil(u, m, 1, q, 1.0, g_left, g_right))
end
maximum(max(lift_error(arch12, 12, q, m), lift_error(qtt16, 16, q, m)) for q in (2, 8), m in (1, 2))

# The library scales these blocks by ``g/h^m`` and lifts them to ``N`` dimensions with [`ndim_operator_mpo`](@ref):
# `DifferenTT.scalar_dirichlet_correction_dim` returns, for one axis, the operator ``C`` whose product with the all-ones
# MPS is the lift, and [`scalar_dirichlet_correction`](@ref) sums it over the axes. We check the two-dimensional case on
# a ``2^4 \times 2^4`` QTT grid of the box ``[0, 1] \times [0, 1.3]``, with a different value on each face, for both
# stencil orders: the second derivatives of [`DifferentialOpsScalar`](@ref) plus the lift equal the full stencils on the
# ghost-extended array. The values of an MPS on this grid are read with `as_array`, which reshapes the dense vector of
# `contract_mps` (site 1 most significant, so ``x`` is the slower index).

arch2d = QTTArchitecture((4, 4))
sites2d = siteinds(arch2d)
grid2d = Grid(arch2d, (1.0, 1.3))
as_array(mps, arch) = permutedims(reshape(contract_mps(mps), reverse(variable_dims(arch))...))
face_values = (0.7, -1.3, 2.0, 0.5) # x = 0, x = L₁, y = 0, y = L₂
faces = BoundaryConditions(
    BoundaryCond1d(DirichletBC(face_values[1]), DirichletBC(face_values[2])),
    BoundaryCond1d(DirichletBC(face_values[3]), DirichletBC(face_values[4])),
)
dirichlet_field = ScalarField(sites2d, arch2d, faces)
p_random = random_mps(rng, sites2d; linkdims=4)
P = as_array(p_random, arch2d)
function lift_error_2d(q)
    ops = DifferentialOpsScalar(grid2d, dirichlet_field; stencil_order=q)
    C = scalar_dirichlet_correction(dirichlet_field, grid2d, 2; stencil_order=q)
    lifted = as_array(apply(ops.d2_ops[1], p_random) + apply(ops.d2_ops[2], p_random), arch2d)
    lifted .+= as_array(apply(C, ones_mps(sites2d, 1)), arch2d)
    reference = sum(ghost_stencil(P, 2, a, q, grid2d.dx[a], face_values[2a-1], face_values[2a]) for a in 1:2)
    return maximum(abs, lifted - reference) / maximum(abs, reference)
end
(q2=lift_error_2d(2), q8=lift_error_2d(8))

# A face whose value varies along it, ``g(y)``, needs the lift ``\frac{1}{h^m} (C \otimes \mathbb{I})\, \mathbf{g}``,
# where ``\mathbf{g}`` is the profile extended along the axis of the face by [`face_profile_mps`](@ref), since ``C``
# must act on a vector that is constant along its axis. [`face_dirichlet_lift`](@ref) builds this lift for one face. It
# serves the vorticity–streamfunction formulation, and its default stencil order is therefore ``q = 2``. We check it on
# the left face of ``x`` with the profile ``g(y) = \sin(\pi y / 1.3)``, for both stencil orders, against the full
# stencil whose ghost nodes hold ``g(y_j)`` on each row ``y_j``.

profile = sin.(π .* grid2d.points[2] ./ 1.3)
function profile_lift_error(q)
    ops = DifferentialOpsScalar(grid2d, dirichlet_field; stencil_order=q)
    lift = face_dirichlet_lift(dirichlet_field, grid2d, 2, 1, :left, profile; stencil_order=q)
    lifted = as_array(apply(ops.d2_ops[1], p_random), arch2d) + as_array(lift, arch2d)
    reference = ghost_stencil(P, 2, 1, q, grid2d.dx[1], profile, 0.0)
    return maximum(abs, lifted - reference) / maximum(abs, reference)
end
(q2=profile_lift_error(2), q8=profile_lift_error(8))

# This construction also fixes where the Dirichlet value is located. The lift puts ``g`` on the ghost nodes, and the
# stored boundary node ``u_0`` stays an unknown, so the value sits on the first ghost node, one spacing beyond the last
# stored node: at ``x = -h`` and ``x = L + h`` on the [`Grid`](@ref) of Sec. 1, as the diamonds of the ghost-node
# diagram show. A Dirichlet problem posed on ``[0, L]`` must therefore use a grid whose first ghost nodes are the
# boundary points, which is what [`interior_grid`](@ref) builds (see the tutorial on the Poisson equation in one
# dimension).
#
# The lift fills every ghost node with the same value ``g``, and here the two stencil orders behave differently. For
# ``q = 2`` only the first ghost node is reached, and the closure is exact for a linear field. For ``q = 8`` the nodes
# ``u_{-2}, u_{-3}, u_{-4}`` also hold ``g`` instead of the extrapolated values, which leaves an error of order ``1/h``
# on the first rows of the second derivative; the solution of a Poisson problem is then only first-order accurate. This
# is a known limitation of v0.1 (listed in `NEWS.md`), whose planned fix for v0.2 is the odd reflection about the first
# ghost node, ``u_{-1-k} = 2g - u_{-1+k}``. We show it on the linear field ``u(x) = 1 + x``, with the values ``u(-h)``
# and ``u(L + h)`` of the two first ghost nodes as Dirichlet data. Its exact second derivative is zero, and we print the
# largest entry of the lifted second derivative on ``2^4`` and ``2^5`` nodes of the [`Grid`](@ref) of ``[0, 1]``, for
# both stencil orders.

function linear_field_residual(n, q)
    arch = TTArchitecture((n,))
    h = 1 / (n - 1)
    u = 1 .+ h .* (0:(n-1))
    D = DifferenTT.onedim_differential_operator(2, arch, 1, truncate_bc; stencil_order=q)
    C_L = DifferenTT.onedim_left_dirichlet_correction(2, arch, 1; stencil_order=q)
    C_R = DifferenTT.onedim_right_dirichlet_correction(2, arch, 1; stencil_order=q)
    return maximum(abs, (D * u + (1 - h) * C_L * ones(n) + (1 + n * h) * C_R * ones(n)) / h^2)
end
[(n=n, h=1 / (n - 1), q2=linear_field_residual(n, 2), q8=linear_field_residual(n, 8)) for n in (16, 32)]

# Halving ``h`` doubles the residual of ``q = 8``, while ``q = 2`` stays at the round-off.
#
# Finally, the value of a `DirichletBC` enters only through this lift, which `scalar_dirichlet_correction_dim` builds
# and only the pressure system of the Chorin projection uses ([`poisson_rhs`](@ref), Sec. 9). The explicit time steppers
# do not impose it: as we show in Sec. 5, they hold the rows next to a Dirichlet side at the values of the initial
# field. A velocity with a Dirichlet side is therefore fixed on its stored boundary node and the next ``q/2 - 1`` nodes,
# whereas the pressure value sits on the first ghost node. We confirm that the value does not enter the operators: the
# first and second derivatives of the field above are the same if all its faces hold the value 5.

shifted_faces = BoundaryConditions(
    BoundaryCond1d(DirichletBC(5.0), DirichletBC(5.0)), BoundaryCond1d(DirichletBC(5.0), DirichletBC(5.0))
)
ops_values = DifferentialOpsScalar(grid2d, dirichlet_field; stencil_order=8)
ops_shifted = DifferentialOpsScalar(grid2d, ScalarField(sites2d, arch2d, shifted_faces); stencil_order=8)
maximum(
    maximum(abs, contract_mpo(a[i]) - contract_mpo(b[i])) for
    (a, b) in ((ops_values.d_ops, ops_shifted.d_ops), (ops_values.d2_ops, ops_shifted.d2_ops)), i in 1:2
)

# ## 5. Boundary rows of the divergence
#
# The truncated rows of Sec. 3 are not consistent approximations of the derivatives, and the velocity–pressure
# formulation removes them from the two places where they would act. The projector
#
# ```math
# Z = \mathbb{I} - \sum_{r \in R} \mathbf{e}_r \mathbf{e}_r^\top,
# ```
#
# where ``R`` holds the ``q/2`` rows next to each `DirichletBC` side, is built by `DifferenTT.onedim_boundary_nullifier`
# in one dimension; it is the identity with zeros on those rows. We print its diagonal on 12 nodes, for a Dirichlet left
# side and a Neumann right side, and check the QTT version on ``2^4`` nodes against the same mask.

inlet_outlet = BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0))
Z12 = DifferenTT.onedim_boundary_nullifier(arch12, 1, inlet_outlet; stencil_order=8)
Int.(diag(Z12))

#-

mask16 = [i <= 4 ? 0.0 : 1.0 for i in 1:16]
maximum(
    abs, contract_mpo(DifferenTT.onedim_boundary_nullifier(qtt16, 1, inlet_outlet; stencil_order=8)) - Diagonal(mask16)
)

# In ``N`` dimensions, [`BoundaryNullScalar`](@ref) multiplies the one-dimensional projectors of every axis, lifted by
# [`ndim_operator_mpo`](@ref), so it zeroes every row next to a Dirichlet side of any axis; [`BoundaryNull`](@ref)
# collects one per component of a velocity field, each built from the conditions of its own component. The time steppers
# apply it to the right-hand side of the predictor ([`NavierStokesPredictorRHS`](@ref)), so that an explicit step leaves
# the field on those rows at its initial values: this is how a Dirichlet velocity is imposed.
#
# The divergence needs a different selection. Its term ``\partial_i u_i`` is the first derivative of component ``i``
# along axis ``i``, which is truncated only within ``q/2`` rows of a Dirichlet side of component ``i`` on that axis.
# [`divergence_dirichlet_correction`](@ref) therefore multiplies, for each axis ``i``, the projector of component ``i``
# along axis ``i`` only. We take the velocity of the channel of Sec. 9, on the ``2^4 \times 2^4`` QTT grid of the box
# ``[0, 2] \times [0, 1]``, periodic in ``y``: ``u`` is 1 and ``v`` is 0 at the inlet ``x = 0`` (Dirichlet), and both
# are Neumann at the outlet ``x = 2``. Both components hold their inlet rows fixed in the time steps, and the divergence
# drops the inlet rows of ``\partial_x u``. We print the diagonals of the three projectors, reshaped on the grid, for
# the first six columns ``x_0, \dots, x_5`` and the first row ``y_0`` (every row ``y_j`` is the same).

channel_arch = QTTArchitecture((4, 4))
channel_sites = siteinds(channel_arch)
u_bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(1.0), NeumannBC(0.0)), periodic)
v_bcs = BoundaryConditions(BoundaryCond1d(DirichletBC(0.0), NeumannBC(0.0)), periodic)
channel_grid = Grid(channel_arch, (2.0, 1.0); periodic=DifferenTT.periodic_axes(u_bcs))
uniform = VectorField((ones_mps(channel_sites, 1), zeros_mps(channel_sites)), channel_arch, (u_bcs, v_bcs))
diagonal_on_grid(Z) = as_array(apply(Z, ones_mps(channel_sites, 1)), channel_arch)
held_u, held_v = (diagonal_on_grid(BoundaryNull(uniform; stencil_order=8)[i].u_row_deleter) for i in 1:2)
removed = diagonal_on_grid(divergence_dirichlet_correction(uniform; stencil_order=8))
(u_held=round.(Int, held_u[1:6, 1]), v_held=round.(Int, held_v[1:6, 1]), divergence_kept=round.(Int, removed[1:6, 1]))

# Why the divergence needs this is seen on the uniform inflow ``\mathbf{u} = (1, 0)``, which is divergence-free. Its
# truncated ``\partial_x u`` takes the ghost values of ``u`` at the inlet as zero, so the first four columns see a jump
# from 0 to 1 and the divergence is of order ``1/h`` there; at the outlet the Neumann mirror gives exactly zero. We
# print the largest divergence of each of the first six columns before the projector, and the largest one after it.

channel_ops = DifferentialOps(channel_grid, uniform; stencil_order=8)
div_uniform = divergence(Tuple(last(uniform)), channel_ops)
div_removed = apply(divergence_dirichlet_correction(uniform; stencil_order=8), div_uniform)
(
    before=round.(vec(maximum(abs, as_array(div_uniform, channel_arch); dims=2))[1:6]; sigdigits=4),
    after=maximum(abs, as_array(div_removed, channel_arch)),
)

# The source of the pressure equation is therefore zero on the ``q/2`` columns next to a Dirichlet velocity face: there
# the pressure satisfies the homogeneous equation, up to its own lifts and closures, instead of responding to a spurious
# source of order ``1/h``. Without the projector, this source would drive a spurious pressure gradient at the inlet.
#
# The selection by component is visible when only the tangential component has a Dirichlet side: a velocity whose ``u``
# is Neumann and whose ``v`` is Dirichlet at ``x = 0`` holds the inlet rows of ``v`` in the time steps, but removes no
# row from the divergence, since ``\partial_y v`` is not truncated at ``x = 0``.

tangential = VectorField(
    (ones_mps(channel_sites, 1), zeros_mps(channel_sites)),
    channel_arch,
    (BoundaryConditions(BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0)), periodic), v_bcs),
)
(
    v_held=round.(Int, diagonal_on_grid(BoundaryNull(tangential; stencil_order=8)[2].u_row_deleter)[1:6, 1]),
    divergence_projector=divergence_dirichlet_correction(tangential; stencil_order=8),
)

# ## 6. Homogeneous Neumann sides: the mirror
#
# A homogeneous Neumann condition, ``\partial u / \partial n = 0``, is imposed at the outlet of the channel for the
# velocity and at its inlet for the pressure. In the library the boundary lies on the stored node ``u_0``, and the ghost
# values are the mirror images of the stored ones about it,
#
# ```math
# u_{-k} = u_k, \qquad u_{n-1+k} = u_{n-1-k}, \qquad k = 1, \dots, q/2,
# ```
#
# which makes every central first difference at ``u_0`` vanish, since ``c_{-k} = -c_k``. This rule involves the
# unknowns, so substituting it changes the operator: the coefficient ``c_{-k}`` that row ``i`` applies to a ghost node
# moves to the stored node at the mirror position. `DifferenTT.neumann_correction_matrix` returns this change as a
# ``q \times q`` corner block ``M``, which is subtracted from the truncated operator, ``D_\text{closed} = D - M``. We
# print the left block of the second derivative for ``q = 2``, and for ``q = 8`` scaled as integers.

neumann_bc = BoundaryCond1d(NeumannBC(0.0), NeumannBC(0.0))
DifferenTT.neumann_correction_matrix(2, :left; stencil_order=2)

#-

as_integers(DifferenTT.neumann_correction_matrix(2, :left; stencil_order=8), 2, 8)

# Row 1 of the second block (node ``u_0``) receives ``c_{-1}, \dots, c_{-4}`` on the columns of ``u_1, \dots, u_4``, row
# 2 (node ``u_1``) receives ``c_{-2}, c_{-3}, c_{-4}`` on ``u_1, u_2, u_3``, and so on: the minus sign of ``M`` is only
# the convention of the subtraction. The closed operators of 8 nodes for ``q = 2`` and of 12 nodes for ``q = 8`` are:

as_integers(DifferenTT.onedim_differential_operator(2, arch8, 1, neumann_bc; stencil_order=2), 2, 2)

#-

as_integers(DifferenTT.onedim_differential_operator(2, arch12, 1, neumann_bc; stencil_order=8), 2, 8)

# `DifferenTT.left_neumann_boundary_correction` and `DifferenTT.right_neumann_boundary_correction` embed the two blocks
# in an operator of the size of the axis, a matrix in TT and an MPO in QTT (built on ``\log_2 q`` sites and padded with
# zeros), and `onedim_differential_operator` subtracts them. The dense check compares both embeddings with the
# difference between the truncated and the mirrored hand operators, and the closed operators with the hand elimination
# for the pure Neumann axis and for the mixed Neumann–Dirichlet axis of the pressure of the channel.

function neumann_block_error(q, m)
    errors = Float64[]
    for (side, rules) in ((:left, (:mirror, :truncate)), (:right, (:truncate, :mirror)))
        embed =
            side == :left ? DifferenTT.left_neumann_boundary_correction : DifferenTT.right_neumann_boundary_correction
        for (n, embedded) in (
            (12, embed(m, arch12, 12; stencil_order=q)),
            (16, contract_mpo(embed(m, qtt16, siteinds(2, 4); stencil_order=q))),
        )
            hand = ghost_operator(m, n, :truncate, :truncate; q) - ghost_operator(m, n, rules...; q)
            push!(errors, maximum(abs, embedded - hand))
        end
    end
    return maximum(errors)
end
(
    blocks=maximum(neumann_block_error(q, m) for q in (2, 8), m in (1, 2)),
    neumann=closure_errors(neumann_bc, :mirror, :mirror),
    neumann_dirichlet=closure_errors(BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), :mirror, :truncate),
)

# The mirror moves the coefficient of row 1 onto column 2, but nothing moves the coefficient of row 2 onto column 1, so
# the operator is no longer symmetric: in the ``q = 2`` operator above, the entry ``(1, 2)`` is 2 and the entry
# ``(2, 1)`` is 1. This is a problem for the pressure system, which is solved by CG and by ALS with the local CG of an
# SPD operator. The first-derivative row of the boundary node, on the other hand, is exactly zero, as the condition
# requires. We print the norm of ``A - A^\top`` of the mirrored second derivatives of 12 nodes, and the largest entry of
# the first-derivative row of ``u_0``.

(
    asymmetry=[
        norm(ghost_operator(2, 12, :mirror, :mirror; q) - ghost_operator(2, 12, :mirror, :mirror; q)') for q in (2, 8)
    ],
    first_derivative_row=maximum(
        abs, DifferenTT.onedim_differential_operator(1, arch12, 1, neumann_bc; stencil_order=8)[1, :]
    ),
)

# ## 7. Symmetrization: the Neumann weight
#
# The asymmetry of the mirror has a simple cure. For ``q = 2``, the mirrored row of ``u_0`` reads
# ``(2u_1 - 2u_0)/h^2 = f_0``, and the second row ``(u_0 - 2u_1 + u_2)/h^2 = f_1``; halving the first row gives entries
# ``1/h^2`` on both sides of the diagonal. The halved row is the finite-volume equation of the half cell ``[0, h/2]``
# that the boundary node owns: integrating ``u'' = f`` over it gives
# ``u'(h/2) - u'(0) \approx (u_1 - u_0)/h = (h/2) f_0``, where ``u'(0) = 0`` is the Neumann condition. Multiplying a row
# by a nonzero factor does not change the solution, as long as the right-hand side is multiplied by the same factor.
#
# The library applies this weight to whole ``N``-dimensional rows: ``W = W_1 \otimes \dots \otimes W_N``, where ``W_i``
# is the identity with ``1/2`` on the Neumann boundary rows of axis ``i``, so that a node shared by two Neumann faces
# gets ``1/4``, the quarter cell of a corner. In one dimension, `DifferenTT.onedim_neumann_symmetrization_correction`
# returns ``-\frac{1}{2} \sum_r \mathbf{e}_r \mathbf{e}_r^\top A``, minus half of the Neumann rows of an operator ``A``
# (the sum of `DifferenTT.left_neumann_symmetrization_correction` and
# `DifferenTT.right_neumann_symmetrization_correction`), and applied to the identity it gives ``W_i - \mathbb{I}``. We
# print the diagonal of ``W_i`` on 12 nodes, and check, for both stencil orders and for a pure Neumann and a
# Neumann–Dirichlet axis, that ``W_i A`` is symmetric while ``A`` is not; in QTT, the correction of the MPO of ``A`` on
# ``2^4`` nodes is compared with the dense one.

I12 = Matrix{Float64}(I, 12, 12)
W12 = I12 + DifferenTT.onedim_neumann_symmetrization_correction(I12, arch12, 1, neumann_bc)
diag(W12)

#-

function weight_check(q, bc, rules)
    A = ghost_operator(2, 12, rules...; q)
    W = I12 + DifferenTT.onedim_neumann_symmetrization_correction(I12, arch12, 1, bc)
    A16 = ghost_operator(2, 16, rules...; q)
    C16 = DifferenTT.onedim_neumann_symmetrization_correction(mat2mpo(A16, siteinds(2, 4)), qtt16, 1, bc)
    W16 = Diagonal([i in (1, 16) && (i == 1 ? rules[1] : rules[2]) == :mirror ? 0.5 : 1.0 for i in 1:16])
    return (
        A_asymmetry=norm(A - A'),
        WA_asymmetry=norm(W * A - (W * A)'),
        qtt=maximum(abs, A16 + contract_mpo(C16) - W16 * A16),
    )
end
[
    weight_check(q, bc, rules) for q in (2, 8),
    (bc, rules) in
    ((neumann_bc, (:mirror, :mirror)), (BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)), (:mirror, :truncate)))
]

# In two dimensions, [`neumann_weight_mpo`](@ref) lifts each ``W_i`` with [`ndim_operator_mpo`](@ref) and multiplies
# them. We build it for a field on the ``2^4 \times 2^4`` QTT grid of ``[0, 1] \times [0, 1.3]`` with Neumann sides on
# both ends of ``x`` and a Neumann–Dirichlet axis ``y``, and print its diagonal on the nodes ``x_0, x_1, x_{15}`` and
# ``y_0, y_1, y_{15}``, together with the bond dimensions of the MPO.

mixed_bcs = BoundaryConditions(neumann_bc, BoundaryCond1d(NeumannBC(0.0), DirichletBC(0.0)))
mixed_field = ScalarField(sites2d, arch2d, mixed_bcs)
W_mpo = neumann_weight_mpo(mixed_field)
W_grid = as_array(apply(W_mpo, ones_mps(sites2d, 1)), arch2d)
(weights=round.(W_grid[[1, 2, 16], [1, 2, 16]]; digits=12), bonds=linkdims(W_mpo))

# The corners ``(x_0, y_0)`` and ``(x_{15}, y_0)`` are shared by two Neumann faces and get ``1/4``; ``y_{15}`` is a
# Dirichlet side, where the factor ``W_2`` is 1. Within an axis, ``W_i`` has bond dimension 3 when both sides are
# Neumann and 2 when one is, and the bond between the two axes of this non-interleaved QTT has dimension 1, since ``W``
# is a Kronecker product (an interleaved QTT threads the bonds of each axis through the other).
#
# [`symmetrized_laplacian`](@ref) multiplies the Kronecker sum of [`kronecker_laplacian`](@ref) by this weight,
# ``W\Delta``, and [`symmetrize_rhs`](@ref) multiplies the right-hand side by it. We check, for both stencil orders,
# that ``W\Delta`` equals the dense product of the two, that it is symmetric, and that the weighted system has the
# solution of the unweighted one: we solve ``-\Delta \mathbf{u} = \mathbf{f}`` and ``-W\Delta \mathbf{u} = W\mathbf{f}``
# densely, with a random ``\mathbf{f}`` (the Dirichlet side makes both systems non-singular).

f_random = random_mps(rng, sites2d; linkdims=4)
function symmetrized_check(q)
    Δ = contract_mpo(kronecker_laplacian(mixed_field, grid2d; stencil_order=q))
    WΔ = contract_mpo(symmetrized_laplacian(mixed_field, grid2d; stencil_order=q))
    W = contract_mpo(W_mpo)
    f, Wf = contract_mps(f_random), contract_mps(symmetrize_rhs(mixed_field, f_random))
    u, u_weighted = -Δ \ f, -WΔ \ Wf
    return (
        product=maximum(abs, WΔ - W * Δ) / maximum(abs, Δ),
        asymmetry=norm(WΔ - WΔ') / norm(WΔ),
        same_solution=norm(u - u_weighted) / norm(u),
    )
end
(q2=symmetrized_check(2), q8=symmetrized_check(8))

# One might instead halve the boundary row of each one-dimensional operator before the Kronecker sum,
# ``(W_1 D_{xx}) \otimes \mathbb{I} + \mathbb{I} \otimes (W_2 D_{yy})``. This operator is symmetric too, but it is not a
# row scaling of ``\Delta``: on a Neumann row of ``x``, it halves ``\partial_x^2`` and keeps ``\partial_y^2``, i.e. the
# tangential derivative gets twice the weight of the normal one, and the solution changes. This is why the library
# weights whole rows (the comment in `symmetrized_laplacian` says so). We compare the two operators for ``q = 8``.

function per_axis_check(q)
    D_xx = ghost_operator(2, 16, :mirror, :mirror; q) ./ grid2d.dx[1]^2
    D_yy = ghost_operator(2, 16, :mirror, :truncate; q) ./ grid2d.dx[2]^2
    W_x, W_y = Diagonal([i in (1, 16) ? 0.5 : 1.0 for i in 1:16]), Diagonal([i == 1 ? 0.5 : 1.0 for i in 1:16])
    per_axis = kron(W_x * D_xx, I(16)) + kron(I(16), W_y * D_yy)
    Δ = kron(D_xx, I(16)) + kron(I(16), D_yy)
    f = contract_mps(f_random)
    return (
        asymmetry=norm(per_axis - per_axis') / norm(per_axis),
        solution_change=norm(-per_axis \ (kron(W_x, W_y) * f) - (-Δ \ f)) / norm(-Δ \ f),
    )
end
per_axis_check(8)

# The weight is only for the linear systems. The explicit terms, such as the diffusion of the velocity in the predictor
# ([`diffusion`](@ref)), must approximate ``\nabla^2 u`` on every node, the boundary nodes included, so they use the
# unweighted operators, as `kronecker_laplacian` does. On the function ``u(x) = \cos(\pi x)``, which satisfies the
# Neumann condition at both ends of ``[0, 1]``, the unweighted second derivative of ``q = 8`` gives ``-\pi^2`` at
# ``x = 0``, and the weighted one gives half of it, as the finite-volume reading predicts.

neumann_field = ScalarField(siteinds(qtt16), qtt16, BoundaryConditions(neumann_bc))
cosine = vec2mps(cos.(π .* grid16.points[1]), neumann_field.usites)
(
    exact=(-π^2),
    kronecker=contract_mps(apply(kronecker_laplacian(neumann_field, grid16; stencil_order=8), cosine))[1],
    symmetrized=contract_mps(apply(symmetrized_laplacian(neumann_field, grid16; stencil_order=8), cosine))[1],
)

# The figure collects the boundary blocks of the closures of Secs. 2 to 7, for the second derivative with ``q = 8`` on
# 16 nodes, as differences with the truncated band: (a) the wrapped entries of the periodic closure, (b) the mirrored
# entries ``-M`` of two Neumann sides, (c) the Dirichlet lift blocks ``C_L`` and ``C_R``, which act on the right-hand
# side, and (d) the asymmetry ``A - A^\top`` of the mirrored operator, which the weight ``W`` removes. Each nonzero
# entry is printed in its cell, scaled by 5040, and the colors follow its sign and its logarithm. Only the ``q/2 = 4``
# rows next to each side are touched.

function boundary_blocks()
    truncated = ghost_operator(2, 16, :truncate, :truncate; q=8)
    mirrored = ghost_operator(2, 16, :mirror, :mirror; q=8)
    lift = zeros(16, 16)
    lift[1:4, 1:4] = DifferenTT.onedim_left_dirichlet_correction(2, arch12, 1; stencil_order=8)[1:4, 1:4]
    lift[13:16, 13:16] = DifferenTT.onedim_right_dirichlet_correction(2, arch12, 1; stencil_order=8)[9:12, 9:12]
    panels = (
        ("(a) periodic: wrapped − truncated", ghost_operator(2, 16, :wrap, :wrap; q=8) - truncated),
        ("(b) Neumann: mirrored − truncated", mirrored - truncated),
        ("(c) Dirichlet: lift blocks C_L, C_R", lift),
        ("(d) Neumann: A − Aᵀ (removed by W)", mirrored - mirrored'),
    )
    function panel(label, block)
        values = round.(Int, 5040 .* block)
        fig = heatmap(
            1:16,
            1:16,
            sign.(values) .* log10.(1 .+ abs.(values));
            yflip=true,
            aspect_ratio=1,
            color=cgrad(:RdBu; rev=true),
            clims=(-4.2, 4.2),
            title=label,
            titlefontsize=10,
            xlabel="column",
            ylabel="row",
            xlims=(0.5, 16.5),
            ylims=(0.5, 16.5),
            colorbar=false,
            framestyle=:box,
        )
        for i in 1:16, j in 1:16
            iszero(values[i, j]) || annotate!(fig, j, i, text(string(values[i, j]), 5, :black))
        end
        return fig
    end
    return plot(
        [panel(label, block) for (label, block) in panels]...; layout=(1, 4), size=(1500, 420), margin=5Plots.mm
    )
end
boundary_blocks()
