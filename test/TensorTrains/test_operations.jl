using DifferenTT
using Test
using Adapt
using ITensors
using ITensorMPS
using LinearAlgebra
include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# TODO(release): test that the array_type arguments and adapt calls keep every tensor on the backend of the inputs, with
# a GPU-like array type that runs on the CPU (e.g. JLArrays, which needs a test dependency); see the WP3a report

# Dense reference for reversing the site order of an operator on sites of dimensions d: mat2tensor splits the row and
# column indices into digits, site 1 first; reading the same array back in column-major order makes the last site the
# most significant
reverse_sites_dense(M::AbstractMatrix, d::Vector{Int}) = reshape(DifferenTT.mat2tensor(M, d), size(M))

@testset "utils.jl" begin
    @testset "replace_mpo_siteinds!" begin
        nsite = 4
        old_sites = siteinds(2, nsite)
        new_sites = siteinds(2, nsite)
        mpo = random_mpo(old_sites)
        M_old = contract_mpo(mpo)

        # The non-mutating version leaves its argument untouched
        copied = DifferenTT.replace_mpo_siteinds(mpo, new_sites)
        @test all(old_sites[i] in inds(mpo[i]) for i in 1:nsite)
        @test contract_mpo(copied) ≈ M_old

        @test DifferenTT.replace_mpo_siteinds!(mpo, new_sites) === mpo

        # After replacement, old site indices should be gone and the dense operator unchanged
        for i in 1:nsite
            @test new_sites[i] ∈ inds(mpo[i])
            @test new_sites[i]' ∈ inds(mpo[i])
            @test old_sites[i] ∉ inds(mpo[i])
        end
        @test contract_mpo(mpo) ≈ M_old
    end

    @testset "transpose_mpo" begin
        nsite = 4
        sites = siteinds(2, nsite)
        mpo = random_mpo(sites)
        transposed_mpo = DifferenTT.transpose_mpo(mpo)
        @test transposed_mpo isa MPO
        @test contract_mpo(transposed_mpo) ≈ transpose(contract_mpo(mpo))
        @test all(transposed_mpo[i] == swapinds(mpo[i], siteind(mpo, i), siteind(mpo, i)') for i in 1:nsite)
    end

    @testset "reverse_mpo" begin
        nsite = 4
        sites = siteinds(2, nsite)
        mpo = random_mpo(sites)
        reversed_mpos = DifferenTT.reverse_mpo(mpo)
        @test reversed_mpos isa MPO
        # Reversing the site order permutes the Kronecker factors: compare as tensors with the site axes reversed
        Mt = reshape(contract_mpo(mpo), fill(2, 2nsite)...)
        Rt = reshape(contract_mpo(reversed_mpos), fill(2, 2nsite)...)
        @test Rt ≈ permutedims(Mt, (reverse(1:nsite)..., reverse((nsite+1):(2nsite))...))
        @test all(hastags(l, "l=$k") for (k, l) in enumerate(linkinds(reversed_mpos)))
        @test length(reversed_mpos) == nsite
        # Check connectivity
        for i in 1:(nsite-1)
            @test !isnothing(commonind(reversed_mpos[i], reversed_mpos[i+1]))
        end
        # Check that original MPO is not mutated and objects are different
        @test reversed_mpos[1] !== mpo[nsite]
    end

    @testset "_nothing_sum_handler" begin
        sites = siteinds(2, 4)
        a = random_mps(sites)
        b = random_mps(sites)

        # All nothing → nothing
        @test isnothing(DifferenTT._nothing_sum_handler(nothing, nothing))

        # One element → returns that element
        result = DifferenTT._nothing_sum_handler(a, nothing)
        @test result isa MPS

        # Two elements → sum
        result2 = DifferenTT._nothing_sum_handler(a, b)
        @test result2 isa MPS
    end

    @testset "_nothing_apply_handler" begin
        sites = siteinds(2, 3)
        A = random_mpo(sites)
        B = random_mpo(sites)
        @test isnothing(DifferenTT._nothing_apply_handler(nothing, nothing))
        @test DifferenTT._nothing_apply_handler(nothing, A, nothing) === A
        AB = DifferenTT._nothing_apply_handler(A, nothing, B)
        @test contract_mpo(AB) ≈ contract_mpo(A) * contract_mpo(B)
    end

    # Regression test: an axis outside 1:N gave the identity instead of an error
    @testset "ndim_operator_mpo (TT) validates the axis" begin
        arch = TTArchitecture((2, 3))
        sites = siteinds(arch)
        @test_throws ArgumentError DifferenTT.ndim_operator_mpo(rand(2, 2), 3, sites, arch)
        @test_throws ArgumentError DifferenTT.ndim_operator_mpo(rand(2, 2), 0, sites, arch)
        arch1 = TTArchitecture((2,))
        @test_throws ArgumentError DifferenTT.ndim_operator_mpo(rand(2, 2), 2, siteinds(arch1), arch1)
    end

    @testset "transpose_mpo and reverse_mpo with unequal site dimensions" begin
        for d in ([2, 3, 4], [3])
            sites = DifferenTT.site_indices(d)
            M = randn(rng(), prod(d), prod(d))
            A = DifferenTT.mat2mpo(M, sites)
            @test todense(DifferenTT.transpose_mpo(A)) ≈ transpose(M)
            @test todense(DifferenTT.reverse_mpo(A)) ≈ reverse_sites_dense(M, d)
            @test dim.(only.(siteinds(DifferenTT.reverse_mpo(A); plev=0))) == reverse(d)
            # reversing twice gives the operator back
            @test todense(DifferenTT.reverse_mpo(DifferenTT.reverse_mpo(A))) ≈ M
        end
        # the reference itself, on a Kronecker product: reversing the sites reverses the factors
        a, b = randn(rng(), 2, 2), randn(rng(), 3, 3)
        @test reverse_sites_dense(kron(a, b), [2, 3]) ≈ kron(b, a)
    end

    @testset "contract_mps and contract_mpo: site 1 is the most significant" begin
        sites = DifferenTT.site_indices([2, 3])
        x, y = [1.0, 2.0], [3.0, 4.0, 5.0]
        X = MPS([ITensor(x, sites[1]), ITensor(y, sites[2])])
        @test contract_mps(X) == kron(x, y)
        a, b = [1.0 2; 3 4], [1.0 0 2; 0 3 0; 4 0 5]
        A = MPO([ITensor(a, sites[1]', sites[1]), ITensor(b, sites[2]', sites[2])])
        @test contract_mpo(A) == kron(a, b)
        # the rows are the output (primed) indices
        @test contract_mpo(DifferenTT.transpose_mpo(A)) == kron(transpose(a), transpose(b))
    end

    @testset "diagmpo and diagmps" begin
        for d in ([2, 3, 2], [4])
            sites = DifferenTT.site_indices(d)
            M = randn(rng(), prod(d), prod(d))
            A = DifferenTT.mat2mpo(M, sites)
            DA = diagmpo(A)
            @test todense(DA) ≈ Diagonal(diag(M))
            @test only.(siteinds(DA; plev=0)) == sites
            dA = diagmps(A)
            @test todense(dA) ≈ diag(M)
            @test siteinds(dA) == sites
            @test maxlinkdim(DA) == maxlinkdim(dA) == maxlinkdim(A)
            # the input is left untouched
            @test todense(A) ≈ M
        end
    end

    @testset "concatenate_mpos and concatenate_mpss equal the Kronecker product" begin
        r = rng()
        s1, s2, s3 = DifferenTT.site_indices([2, 3]), DifferenTT.site_indices([2]), DifferenTT.site_indices([3, 2])
        Ms = [randn(r, 6, 6), randn(r, 2, 2), randn(r, 6, 6)]
        As = [DifferenTT.mat2mpo(M, s) for (M, s) in zip(Ms, (s1, s2, s3))]
        new_sites = DifferenTT.site_indices([2, 3, 2, 3, 2])
        C = DifferenTT.concatenate_mpos(new_sites, As...)
        @test length(C) == 5
        @test only.(siteinds(C; plev=0)) == new_sites
        @test todense(C) ≈ kron(Ms...)
        @test linkdims(C)[[2, 3]] == [1, 1]  # the glue links between the operators
        # no index is shared with the inputs
        @test isempty(intersect(linkinds(C), vcat(linkinds.(As)...)))
        @test todense(As[1]) ≈ Ms[1]
        @test_throws DimensionMismatch DifferenTT.concatenate_mpos(new_sites[1:4], As...)

        vs = [randn(r, 6), randn(r, 2), randn(r, 6)]
        Xs = [DifferenTT.vec2mps(v, s) for (v, s) in zip(vs, (s1, s2, s3))]
        Cx = DifferenTT.concatenate_mpss(new_sites, Xs...)
        @test siteinds(Cx) == new_sites
        @test todense(Cx) ≈ kron(vs...)
        @test isempty(intersect(linkinds(Cx), vcat(linkinds.(Xs)...)))
        @test_throws DimensionMismatch DifferenTT.concatenate_mpss(new_sites[1:4], Xs...)
        # a single operand is only renamed
        @test todense(DifferenTT.concatenate_mpss(DifferenTT.site_indices([2, 3]), Xs[1])) ≈ vs[1]
    end

    @testset "onedim_eye_padding" begin
        sites = DifferenTT.site_indices([2, 2, 3])
        M = randn(rng(), 3, 3)
        A = DifferenTT.mat2mpo(M, sites[3:3])
        @test todense(DifferenTT.onedim_eye_padding(A, sites)) ≈ kron(I(4), M)
        B = DifferenTT.mat2mpo(M, DifferenTT.site_indices([3]))
        right = DifferenTT.site_indices([3, 2, 2])
        P = DifferenTT.onedim_eye_padding(B, right, :right)
        @test todense(P) ≈ kron(M, I(4))
        @test only.(siteinds(P; plev=0)) == right
        # nothing to pad: the operator itself
        @test DifferenTT.onedim_eye_padding(A, sites[3:3]) === A
        @test_throws ArgumentError DifferenTT.onedim_eye_padding(A, sites, :middle)
        @test_throws ArgumentError DifferenTT.onedim_eye_padding(
            DifferenTT.mat2mpo(randn(rng(), 12, 12), sites), sites[1:2]
        )
    end

    @testset "ndim_operator_mpo (TT) equals the Kronecker product" begin
        r = rng()
        for arch in (TTArchitecture((2, 3, 4)), TTArchitecture((2, 3, 4), Dict(1 => 3, 2 => 1, 3 => 2)))
            sites = siteinds(arch)
            for axis in 1:3
                n = variable_dims(arch)[axis]
                A = randn(r, n, n)
                factors = [
                    arch.site_map[k] == axis ? A : Matrix(1.0I, site_dim(arch, k), site_dim(arch, k)) for k in 1:3
                ]
                op = DifferenTT.ndim_operator_mpo(A, axis, sites, arch)
                @test todense(op) ≈ kron(factors...)
                @test maxlinkdim(op) == 1
            end
        end
        arch1 = TTArchitecture((5,))
        A = randn(r, 5, 5)
        @test todense(DifferenTT.ndim_operator_mpo(A, 1, siteinds(arch1), arch1)) ≈ A
        arch = TTArchitecture((2, 3))
        @test_throws DimensionMismatch DifferenTT.ndim_operator_mpo(randn(r, 2, 2), 1, siteinds(arch)[1:1], arch)
    end

    @testset "ndim_operator_sum_mpo (TT) equals the dense Kronecker sum" begin
        r = rng()
        lift(M, k, d) = kron([j == k ? M : Matrix(1.0I, d[j], d[j]) for j in eachindex(d)]...)
        for arch in (TTArchitecture((2, 3, 4)), TTArchitecture((2, 3, 4), Dict(1 => 3, 2 => 1, 3 => 2)))
            sites = siteinds(arch)
            d = collect(site_dims(arch))
            c = arch.site_map
            ops = ntuple(v -> randn(r, variable_dims(arch)[v], variable_dims(arch)[v]), 3)
            S = DifferenTT.ndim_operator_sum_mpo(ops, sites, arch)
            @test todense(S) ≈ sum(lift(ops[c[k]], k, d) for k in 1:3)
            @test linkdims(S) == [2, 2]

            # pairwise term: sum over the sites i < j of P_{c(i)} ⊗ Q_{c(j)}
            P = ntuple(v -> randn(r, variable_dims(arch)[v], variable_dims(arch)[v]), 3)
            Q = ntuple(v -> randn(r, variable_dims(arch)[v], variable_dims(arch)[v]), 3)
            SP = DifferenTT.ndim_operator_sum_mpo(ops, sites, arch; pair_ops=(P, Q))
            pairs = sum(lift(P[c[i]], i, d) * lift(Q[c[j]], j, d) for i in 1:3 for j in (i+1):3)
            @test todense(SP) ≈ todense(S) + pairs
            @test linkdims(SP) == [3, 3]
        end

        # two sites, and one site (where the pairwise term has no pair)
        arch = TTArchitecture((3, 2))
        sites = siteinds(arch)
        A, B = randn(r, 3, 3), randn(r, 2, 2)
        P, Q = (randn(r, 3, 3), randn(r, 2, 2)), (randn(r, 3, 3), randn(r, 2, 2))
        @test todense(DifferenTT.ndim_operator_sum_mpo((A, B), sites, arch)) ≈ kron(A, I(2)) + kron(I(3), B)
        @test todense(DifferenTT.ndim_operator_sum_mpo((A, B), sites, arch; pair_ops=(P, Q))) ≈
            kron(A, I(2)) + kron(I(3), B) + kron(P[1], Q[2])
        arch1 = TTArchitecture((4,))
        A = randn(r, 4, 4)
        @test todense(DifferenTT.ndim_operator_sum_mpo((A,), siteinds(arch1), arch1)) ≈ A
        @test todense(DifferenTT.ndim_operator_sum_mpo((A,), siteinds(arch1), arch1; pair_ops=((A,), (A,)))) ≈ A
        @test_throws DimensionMismatch DifferenTT.ndim_operator_sum_mpo((A, B), sites[1:1], arch)
    end

    # Regression test: the backend was read as the outermost wrapper type of the matrix and the matrices were rebuilt
    # with it, so an adjoint or transposed matrix was transposed back, and a Symmetric or Diagonal first operator
    # symmetrized or diagonalized the other operators of the sum
    @testset "ndim operators (TT) with wrapped matrices" begin
        arch = TTArchitecture((3, 2))
        sites = siteinds(arch)
        A = [1.0 2 3; 4 5 6; 7 8 10]
        B = [1.0 2; 3 4]
        for W in (A', transpose(A), Symmetric(A), UpperTriangular(A), Diagonal(diag(A)), view(A, :, :))
            @test todense(DifferenTT.ndim_operator_mpo(W, 1, sites, arch)) ≈ kron(Matrix(W), I(2))
            @test todense(DifferenTT.ndim_operator_sum_mpo((W, B), sites, arch)) ≈ kron(Matrix(W), I(2)) + kron(I(3), B)
            @test todense(DifferenTT.ndim_operator_sum_mpo((W, B), sites, arch; pair_ops=((W, B), (W, B')))) ≈
                kron(Matrix(W), I(2)) + kron(I(3), B) + kron(Matrix(W), Matrix(B'))
        end
        @test backend_type(A') == backend_type(Diagonal([1.0])) == backend_type(view(A, :, 1)) == Array
    end

    @testset "_nothing_sum_handler and _sum_elements values" begin
        sites = DifferenTT.site_indices([2, 3, 2])
        a = DifferenTT.vec2mps(randn(rng(), 12), sites)
        b = DifferenTT.vec2mps(randn(rng(), 12), sites)
        c = DifferenTT.vec2mps(randn(rng(), 12), sites)
        @test DifferenTT._nothing_sum_handler(nothing, a, nothing) === a
        @test todense(DifferenTT._nothing_sum_handler(a, nothing, b, c)) ≈ todense(a) + todense(b) + todense(c)
        # cutoff = 0 and no maxdim: the exact direct sum, whose bond dimensions add up
        exact = DifferenTT._nothing_sum_handler(a, b; cutoff=0.0)
        @test todense(exact) ≈ todense(a) + todense(b)
        @test linkdims(exact) == linkdims(a) .+ linkdims(b)
        @test maxlinkdim(DifferenTT._nothing_sum_handler(a, b; maxdim=1)) == 1
        # operators, and a single site, where the tensors are added directly
        A = DifferenTT.mat2mpo(randn(rng(), 12, 12), sites)
        @test todense(DifferenTT._sum_elements([A, A, A])) ≈ 3 * todense(A)
        s1 = DifferenTT.site_indices([3])
        x, y = randn(rng(), 3), randn(rng(), 3)
        z = DifferenTT._sum_elements((DifferenTT.vec2mps(x, s1), DifferenTT.vec2mps(y, s1)))
        @test z isa MPS && todense(z) ≈ x + y
    end
end
