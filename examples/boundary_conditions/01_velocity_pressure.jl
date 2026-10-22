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
