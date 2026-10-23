```@meta
CurrentModule = DifferenTT
```

# Preconditioners

DMRG-type solvers reduce a large linear system in the TT format to a sequence of small local problems, which they solve
with a classical Krylov method. For the SPD operators of elliptic PDEs these local problems are solved by CG, whose
iteration count grows with the square root of the condition number of the local operator, and this condition number
grows as ``\mathcal{O}(h^{-2})`` with the grid spacing ``h``. Preconditioning the local CG solves is therefore what
keeps fine grids affordable. This page condenses Secs. 3 and 4 of Torrente et al., *Factorized approximate inverse
preconditioners in the tensor train format*, which we refer to as "the paper", and cites its equations by number. The
construction we describe, the factorized approximate inverse (FAI), is built by [`fai_preconditioner`](@ref).

## Why approximate inverses lose definiteness

The natural way to precondition CG in the TT format is an approximate inverse ``M^{-1} \approx A^{-1}``, stored as a
tensor train operator (MPO) and applied to the residual. Operator inversion in the TT format is well established, for
instance by solving ``AX + XA = 2I`` with a TT solver (Oseledets & Dolgov, 2012; [`linsolve_preconditioner`](@ref)).
However, CG requires an SPD preconditioner, and truncating a TT operator controls its norm error but not the sign of
its eigenvalues. Definiteness is only guaranteed once the truncation error falls below ``\lambda_{\min}(A^{-1}) =
1/\lambda_{\max}(A)``, a relative accuracy of order ``1/\kappa(A)``, which gets tighter as the grid is refined. Fig. 3
of the paper shows that both the TT solution of ``AX + XA = 2I`` and the exact inverse truncated by SVD have negative
eigenvalues over a range of bond dimensions that widens with the grid size. Direct approximate inverses are therefore
unreliable for PCG precisely when aggressive truncation is needed to keep their cost low.

## The factorized approximate inverse

In order to enforce definiteness by construction, we turn to the factorized sparse approximate inverses (FSAI) of
Kolotilina & Yeremin (1993). The guiding idea is to approximate the inverse not of ``A`` itself but of its Cholesky
factor. We write

```math
A = L_A L_A^\top, \qquad M^{-1} = G_L^\top G_L,
```

where ``G_L \approx L_A^{-1}`` is lower triangular. Since ``\mathbf{v}^\top G_L^\top G_L \mathbf{v} = \|G_L
\mathbf{v}\|^2``, the preconditioner is positive semidefinite for any ``G_L``, and definite whenever ``G_L`` is
non-singular, so no truncation of ``G_L`` can make it indefinite.

The Cholesky factor itself is out of reach in the TT format, but it is not needed. Minimizing ``\|\mathbf{I} - G_L
L_A\|_F^2`` over lower-triangular ``G_L`` gives ``(G_L A)_{ij} = (L_A^\top)_{ij}`` for ``i \ge j`` (Eq. (9)), whose
right-hand side vanishes below the diagonal. Up to a diagonal factor, ``G_L`` is therefore the lower-triangular
approximate inverse ``\tilde{G}_L`` defined by

```math
(\tilde{G}_L A)_{ij} = \delta_{ij}, \qquad i \ge j,
```

which is Eq. (10). We then set ``G_L = D \tilde{G}_L``, with ``D`` chosen so that ``G_L A G_L^\top`` has a unit
diagonal, which gives (Eq. (11))

```math
D = \left[\mathrm{diag}(\tilde{G}_L A \tilde{G}_L^\top)\right]^{-1/2} = \left[\mathrm{diag}(\tilde{G}_L)\right]^{-1/2},
```

where the second equality follows from Eq. (10), since only the lower-triangular entries of ``\tilde{G}_L A``
contribute to the diagonal of ``\tilde{G}_L A \tilde{G}_L^\top``. Note that this shortcut holds only if
``\tilde{G}_L`` satisfies Eq. (10) exactly. For an exact ``\tilde{G}_L``, ``G_L = L_A^{-1}`` and ``M^{-1} =
A^{-1}``.

## Imposing triangularity in the TT format

In the sparse setting, Eq. (10) splits into one small dense system per row, restricted to the lower-triangular pattern.
In the TT format we cannot restrict the unknowns entry by entry, so we instead project the whole system onto the
lower-triangular subspace.

The first step is a mask for lower-triangular matrices. With ``\mathbb{L}`` and ``\mathbb{J}`` the lower-triangular
matrices of ones with and without the diagonal, ``\mathbb{O}`` the matrix of ones and ``\mathbb{I}`` the identity,
all of size ``d_k \times d_k``, the mask ``(P_L)_{ij} = \Theta(i - j)`` of the full ``d_1 \cdots d_N`` space is a
tensor train operator of bond dimension 2 (Eq. (14)):

```math
P_L^{(1:N)} = \begin{bmatrix} \mathbb{I}^{(d_1)} & \mathbb{J}^{(d_1)} \end{bmatrix}
    \mathop{\bowtie}_{k=2}^{N-1} \begin{bmatrix} \mathbb{I}^{(d_k)} & \mathbb{J}^{(d_k)} \\ & \mathbb{O}^{(d_k)}
    \end{bmatrix} \bowtie \begin{bmatrix} \mathbb{L}^{(d_N)} \\ \mathbb{O}^{(d_N)} \end{bmatrix},
```

where ``\bowtie`` is the strong Kronecker product. Vectorizing the mask and placing it on the diagonal of an operator
on the vectorized space gives the projector ``\mathbb{P}_L``, which keeps the lower-triangular entries of a vectorized
matrix and has the same bond dimension 2 (Eq. (20), Fig. 4). In DifferenTT, [`lower_triang_mpo`](@ref) builds the
mask and [`triang_projector_mpo`](@ref) the projector.

We then vectorize Eq. (10), fusing at each site the row and column indices of the unknown into one index of dimension
``d^2``, and project it (Eq. (22)):

```math
\left[\mathbb{P}_L (\mathbf{I} \otimes A) \mathbb{P}_L + \mu (\mathbb{I} - \mathbb{P}_L)\right]
    \mathrm{vec}(\tilde{G}_L) = \mathrm{vec}(\mathbf{I}),
```

where ``\mathbf{I} \otimes A`` applies ``X \mapsto XA`` to the vectorized unknown, and ``\mu > 0`` turns the entries
above the diagonal into the decoupled equations ``\mu (\tilde{G}_L)_{ij} = 0``. The operator is SPD, so the system
can be solved with CG-based DMRG solvers. Since ``\mathbb{P}_L`` acts on each row of the unknown separately, the
operator is block diagonal with one block per row, and on each row the system reduces to the classical FSAI condition
of Eq. (10), whatever the value of ``\mu`` (Appendix B.1). The exact solution is therefore independent of ``\mu``;
``\mu`` only weights how strongly an approximate solver enforces the triangularity. Note that this enforcement is
variational: an approximate solver can leave small entries above the diagonal, unlike the sparse construction, which
excludes them by design.

## The rescaling ``D``

The diagonal ``\mathrm{diag}(\tilde{G}_L)`` is extracted as an MPS by contracting delta tensors ([`diagmps`](@ref),
Fig. 6), and ``D = [\mathrm{diag}(\tilde{G}_L)]^{-1/2}`` (Eq. (23)) is evaluated element-wise by the recursive
sketched interpolation (RSI) of Meng, Khoo, Li & Stoudenmire (2026) ([`elementwise_function`](@ref), Sec. 3.2). RSI
evaluates the function on sketches of the tensor train; we use uniform rather than Gaussian sketches, so that the
sketched entries stay positive and ``x^{-1/2}`` is defined on all of them.

The rescaling is only well defined if every diagonal entry of ``\tilde{G}_L`` is positive, which an approximate solver
does not guarantee. Lemma 1 of the paper settles when it holds. For the exact solution, the diagonal entries are
``g_i^\star = 1/(L_A)_{ii}^2``, with ``1/A_{ii} \le g_i^\star \le 1/\lambda_{\min}(A)``. For a computed
``\tilde{G}_L``, the relative error of each diagonal entry is bounded by the relative energy-norm error ``\eta_i`` of
its row, so a sign change requires ``\eta_i \ge 1``, i.e. losing the whole row; and by Corollary 1, the weighted
fraction of rows with a non-positive diagonal is at most ``\varepsilon_E^2``, with ``\varepsilon_E`` the global
relative energy error. On uniform grids with constant coefficients the exact diagonal is nearly constant and has a TT
rank close to one, which is why sign changes are not observed in practice (Appendix B.5).

!!! warning "Known limitation"
    On strongly non-uniform grids, or with strongly varying coefficients, the rows with a large ``A_{ii}`` have a
    small ``g_i^\star`` and are the cheapest to flip, and the diagonal of ``\tilde{G}_L`` is no longer nearly
    constant, so the shortcut ``D = [\mathrm{diag}(\tilde{G}_L)]^{-1/2}`` can lose accuracy (Appendix B.6). Since
    ``g_i^\star A_{ii} \ge 1``, sampled values of ``(\tilde{G}_L)_{ii} A_{ii}`` noticeably below one would detect it.
    Neither this diagnostic nor the energy-based rescaling ``D_{ii} = (\tilde{\mathbf{g}}_i^\top A
    \tilde{\mathbf{g}}_i)^{-1/2}``, which RSI could evaluate on the uncontracted network (Sec. 3.2, Fig. 7), is
    implemented; both are candidates for a later release.

## Applying the FAI inside ALS

A DMRG-type solver projects the operator onto the sites of each local problem, ``A_{\text{loc}} = U^\top A U``, where
the columns of ``U`` span the basis given by the other tensors of the current solution (Fig. 2). The preconditioner
must be projected too. Contracting ``G_L^\top G_L`` into one MPO would square its bond dimension, and truncating the
product would break its definiteness. We therefore keep track of the projection ``U^\top G_L U`` of the factor alone,
and apply its transpose by contracting the same tensors a second time on their output indices, which gives the
two-layer local preconditioner of Fig. 8,

```math
M^{-1}_{\text{loc}} = (U^\top G_L^\top U)(U^\top G_L U).
```

Storing and updating it costs as much as for any other MPO, and applying it costs ``1 + 2\chi_{G_L}/\chi_A`` times
the application of an operator of bond dimension ``\chi_A`` (Sec. 4). Note that the intermediate ``U U^\top`` makes
``M^{-1}_{\text{loc}}`` differ from ``U^\top G_L^\top G_L U``; it is still positive semidefinite, and definite whenever
``U^\top G_L U`` is non-singular. In DifferenTT, [`FactorizedPreconditioner`](@ref) wraps ``G_L``, and passing it to
[`linearsolve`](@ref) with a DMRG-type solver builds the projected problem (`ReducedFAIPrecondLinearProblem`).

**Local and global preconditioning.** With the DMRG-type solvers ([`ALS`](@ref), [`ResidualALS`](@ref),
[`AMEn`](@ref), [`ExpandedALS`](@ref)), the preconditioner acts on the local problems, as above, and the keyword
`preconditioner_sites` restricts it to the local problems whose first site is in a given set. The global solvers
[`GlobalCG`](@ref) and [`GlobalGMRES`](@ref) instead apply ``M^{-1}`` to the whole residual, as ``G_L^\top (G_L
\mathbf{r})`` with truncation after each factor. The global solvers are not used in the paper, since truncating the
iterates of CG perturbs the conjugacy of its search directions (Sec. 2.1).

## From the paper to the code

| Step | Paper | DifferenTT |
|:--- |:--- |:--- |
| Lower-triangular mask ``P_L`` | Eq. (14) | [`lower_triang_mpo`](@ref) |
| Projector ``\mathbb{P}_L`` | Eq. (20), Fig. 4 | [`triang_projector_mpo`](@ref) |
| Projected system for ``\tilde{G}_L`` | Eq. (22) | `DifferenTT.fai_dg` (internal) |
| ``\mathrm{diag}(\tilde{G}_L)`` as an MPS | Fig. 6 | [`diagmps`](@ref) |
| ``D = [\mathrm{diag}(\tilde{G}_L)]^{-1/2}`` by RSI | Eq. (23), Sec. 3.2 | [`elementwise_function`](@ref) |
| ``G_L = D \tilde{G}_L`` | Eq. (11) | [`fai_preconditioner`](@ref), [`FactorizedPreconditioner`](@ref) |
| Two-layer local application | Sec. 4, Fig. 8 | [`FactorizedPreconditioner`](@ref) in [`linearsolve`](@ref) |

The paper's other terms map as follows: MALS is [`ResidualALS`](@ref) with `nsite = 2`, whose truncation is chosen by
the local residual (Oseledets & Dolgov, 2012); the Sinc preconditioner ``B_M`` of Eq. (26) is
[`sincquadrature_preconditioner`](@ref); the anisotropic operator ``\mathcal{L}(\rho)`` of Eqs. (28)–(29) is
[`equicorrelated_laplacian`](@ref); and the approximate inverses of Fig. 3 come from
[`linsolve_preconditioner`](@ref).

## Choosing the parameters

The paper's experiments use ``\mu = 10^7``, and the exact solution of Eq. (22) does not depend on ``\mu``: the results
do not change for ``\mu`` between ``10^5`` and ``10^8``. The value does set the conditioning of the build. The operator
of Eq. (22) is block diagonal, with the leading principal submatrices of ``A`` and ``\mu`` times the identity on the
entries above the diagonal, so its condition number is ``\max(\mu, \lambda_{\max}(A)) / \min(\mu, \lambda_{\min}(A))``.
A ``\mu`` far above the top of the spectrum of ``A`` therefore makes the local Krylov solves of the build slower and, with a
capped number of sweeps, can return a worse preconditioner. On a ``2^5 \times 2^5`` QTT pressure operator with
``\lambda_{\max}(A) \approx 8 \cdot 10^3``, ``\chi_{G_L} = 4`` and 30 MALS sweeps, ``\mu = 10^3`` built in 4.6 s with
``\kappa(G_L^\top G_L A) = 5.3``, whereas ``\mu = 10^7`` gave ``\kappa = 146`` and did not converge within the 30 sweeps.
Since ``\lambda_{\max}(A)`` grows as ``h^{-2}``, the paper's ``\mu = 10^7`` is close to the top of the spectrum only on
its finest grids, and on a coarse grid we recommend ``\mu \approx \lambda_{\max}(A)``, estimated from the stencil or from a
few Lanczos iterations (see [`fai_preconditioner`](@ref) and the tutorial
[Factorized approximate inverse preconditioners](@ref)). The FAI is
built by MALS with a local tolerance between ``10^{-5}`` and ``10^{-8}``, stopping once the relative residual of Eq.
(22) is below ``10^{-1}`` (`residual_tol = 0.1`), and with ``\chi_{G_L}`` between 4 and 10 (`totalmaxdim`). The bond
dimension of ``G_L`` plays the role of the sparsity pattern of the classical FSAI: it sets the cost of storing and
applying the preconditioner, at the expense of its quality.

The FAI pays off when its build is amortized and its application is cheap. In the Chorin projection of Sec. 5.2, the
same pressure operator is solved at every time step, so the preconditioner is built once per run and reused, and on QTT
sites of dimension ``d = 2`` the unknown of Eq. (22) has sites of dimension only 4; the cumulative solve time then drops
below that of the unpreconditioned solver after a few time steps (Fig. 13). In the high-dimensional anisotropic
diffusion of Sec. 5.1, it reduces the iterations but not the solve time: the vectorized unknown of Eq. (22) has sites of
dimension ``d^2 = 225``, which makes the build expensive, and the operator has bond dimension only 3, so a
preconditioned local iteration costs several plain ones (Sec. 4). The scripts of both experiments, `tt_anisotropy_stats`
and `chorin_pipeline_stats`, are part of the paper bundle.

```julia
using DifferenTT, ITensorMPS, Random
arch = QTTArchitecture((6, 6))                       # 2⁶ × 2⁶ grid, 12 sites
sites = siteinds(arch)
bc = BoundaryConditions(ntuple(_ -> BoundaryCond1d(DirichletBC(0.0), DirichletBC(0.0)), 2)...)
A = -1.0 * symmetrized_laplacian(ScalarField(sites, arch, bc), Grid(arch, (1.0, 1.0)))
b = random_mps(Xoshiro(1), sites; linkdims=2)

cg = LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-5)
build = ResidualALS(; sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=8), updater_kwargs=cg)
M⁻¹ = fai_preconditioner(A; μ=1e7, solver=build, totalmaxdim=8, residual_tol=0.1, rng=Xoshiro(1))

numiter = KrylovSweepStats()
solver = ResidualALS(;
    sweep_kwargs=SweepingArgs(; nsweeps=10, maxdim=64, cutoff=0.0, (observer!)=ResidualObserver(; rtol=1e-6)),
    updater_kwargs=LocalUpdaterArgs(; ishermitian=true, isposdef=true, rtol=1e-8, numiter_store=numiter),
)
x = linearsolve(solver, A, b, M⁻¹, nothing)
sum(DifferenTT.positions_sum(numiter))               # total local CG iterations
```

In our run, the preconditioner reduced the local CG iterations of this solve from 1639 to 435, while its build took
more than 30 times as long as one unpreconditioned solve, which illustrates why the FAI is meant to be reused.

## Checking definiteness

The local spectra are diagnosed through [`LocalUpdaterArgs`](@ref), and nothing is computed unless a consumer is
attached. `cond_store` and `pc_cond_store` ([`KrylovCondStats`](@ref)) record the condition numbers of
``A_{\text{loc}}`` and ``M^{-1}_{\text{loc}} A_{\text{loc}}``; `spectrum_store` ([`KrylovSpectrumStats`](@ref)) records
their extreme eigenvalues with their sign, and by Sylvester's law of inertia a negative eigenvalue of a local
``M^{-1}_{\text{loc}}`` certifies that the global ``M^{-1}`` is not positive definite. `spd_tol` stops a CG solve with
a [`NotSPDError`](@ref) at the first local block whose spectrum is clearly not SPD, and requires `ishermitian = true`
and `isposdef = true`. A non-SPD block raises no warning: only `spd_tol` and `spectrum_store` detect it. All these
diagnostics diagonalize the dense local blocks, at a cost of ``\mathcal{O}(n_{\text{loc}}^3)``, so they are meant for
small diagnostic runs.

## Other preconditioners

- [`sincquadrature_preconditioner`](@ref) builds the exponential-sum approximation ``B_M`` of the inverse of a
  Kronecker sum of one-dimensional operators (Khoromskij, 2009), with bond dimension at most ``2M + 1`` and no solve.
  It is the natural choice for separable operators, but its quality degrades as couplings between the dimensions grow,
  as for ``\mathcal{L}(\rho)`` with large anisotropy (Sec. 5.1).
- [`linsolve_preconditioner`](@ref) solves ``AX + XA = 2I`` (or ``AX = I``) in the TT format. It is accurate in norm,
  but for an SPD operator it can be indefinite once truncated (Fig. 3).
- [`rank1_preconditioner`](@ref) inverts the bond-dimension-1 truncation of the operator. It is exact for Kronecker
  products and cheap, but rough otherwise; we use it to precondition the builds of other preconditioners.
- [`newtoniter_preconditioner`](@ref) refines an approximate inverse by Newton–Schulz iterations, which converge
  quadratically once ``\|I - X_0 A\| < 1``, but whose bond dimension grows with every step.
- [`inverse_laplacian_preconditioner`](@ref) is the exact QTT inverse of the one-dimensional second-order Laplacian
  with Dirichlet–Dirichlet or Dirichlet–Neumann conditions (Kazeev & Khoromskij, 2012), of bond dimension 5 or 4,
  useful as a reference.

Any MPO can also be wrapped in a [`Preconditioner`](@ref), and any factor in a [`FactorizedPreconditioner`](@ref).
