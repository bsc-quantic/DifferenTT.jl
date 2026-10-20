using DifferenTT
using Test
using ITensors
using LinearAlgebra
using ITensorMPS

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

# Dense reference, bit by bit: the lifted operator acts on the value of variable `ax`, read from its bits (most
# significant first, reversed for a negative var_nsites), and is the identity on the values of the other variables
function varvals(bits, q)
    return map(eachindex(q.var_nsites)) do k
        b = [bits[i] for i in eachindex(bits) if q.site_map[i] == k]
        q.var_nsites[k] < 0 && reverse!(b)
        DifferenTT.binary_to_decimal(b)
    end
end
function reference(opd, ax, q)
    n = length(q.site_map)
    M = zeros(2^n, 2^n)
    for r in 0:(2^n-1), c in 0:(2^n-1)
        vr = varvals(DifferenTT.decimal_to_binary(r, n), q)
        vc = varvals(DifferenTT.decimal_to_binary(c, n), q)
        all(vr[k] == vc[k] for k in eachindex(vr) if k != ax) && (M[r+1, c+1] = opd[vr[ax]+1, vc[ax]+1])
    end
    return M
end

# Blocked, interleaved, permuted by var_map and reversed layouts, with a one-site variable in some of them
const LAYOUTS = [
    QTTArchitecture((2, 3)),
    QTTArchitecture((2, 3); interleaved=true),
    QTTArchitecture((2, 3), Dict(1 => 2, 2 => 1)),
    QTTArchitecture((3, 2), Dict(1 => 2, 2 => 1); interleaved=true),
    QTTArchitecture((-2, 3)),
    QTTArchitecture((2, -3); interleaved=true),
    QTTArchitecture((1, -2, 2), Dict(1 => 3, 2 => 1, 3 => 2); interleaved=true),
    QTTArchitecture((-3,)),
]

@testset "qttoperations.jl" begin
    @testset "ndim_operator_mpo" begin
        N = 2
        var_nsites = (2, 3)

        # Blocked architecture
        arch = DifferenTT.QTTArchitecture(var_nsites)
        full_sites = siteinds(arch)

        sites_1 = siteinds(2, 2)
        mpo_1 = MPO(sites_1, "Id")
        lifted_mpo_1 = DifferenTT.ndim_operator_mpo(mpo_1, 1, full_sites, arch)
        @test length(lifted_mpo_1) == length(full_sites)
        mps_test = random_mps(rng(), full_sites)
        @test norm(apply(lifted_mpo_1, mps_test)) > 0

        # Reversal case
        arch_rev = DifferenTT.QTTArchitecture((-2, 3))
        lifted_mpo_rev = DifferenTT.ndim_operator_mpo(mpo_1, 1, full_sites, arch_rev)
        @test length(lifted_mpo_rev) == length(full_sites)

        # Interleaved architecture
        arch_int = DifferenTT.QTTArchitecture(var_nsites; interleaved=true)
        full_sites_int = siteinds(arch_int)
        lifted_mpo_int = DifferenTT.ndim_operator_mpo(mpo_1, 1, full_sites_int, arch_int)
        @test length(lifted_mpo_int) == length(full_sites_int)
        mps_test_int = random_mps(rng(), full_sites_int)
        @test norm(apply(lifted_mpo_int, mps_test_int)) > 0
    end

    # Regression test: a variable with a single site threw a BoundsError (its core has no link indices)
    @testset "ndim_operator_mpo with one-site variables" begin
        for (nbits, interleaved) in (((1,), false), ((2, 1, 2), false), ((2, 1, 2), true), ((-1, 2), true))
            arch = QTTArchitecture(nbits; interleaved)
            sites = siteinds(arch)
            for ax in eachindex(nbits)
                onedim_sites = siteinds(2, abs(nbits[ax]))
                n = 2^abs(nbits[ax])
                op = DifferenTT.mat2mpo(reshape(collect(1.0:(n^2)), n, n), onedim_sites)
                lifted = DifferenTT.ndim_operator_mpo(op, ax, sites, arch)
                @test contract_mpo(lifted) ≈ reference(contract_mpo(op), ax, arch)
            end
        end
    end

    # Regression test: zeros_padding matched the site indices by their position in the tensor, which transposed the
    # block of an MPO whose tensors store (s, s') instead of (s', s)
    @testset "zeros_padding" begin
        s = siteinds(2, 1)
        block = [1.0 2.0; 3.0 4.0]
        for m in (MPO([ITensor(block, s[1]', s[1])]), MPO([ITensor(permutedims(block), s[1], s[1]')]))
            for (corner, row, col) in ((:upleft, 1, 1), (:upright, 1, 2), (:bottomleft, 2, 1), (:bottomright, 2, 2))
                E = zeros(2, 2)
                E[row, col] = 1.0
                @test DifferenTT.contract_mpo(DifferenTT.zeros_padding(m, siteinds(2, 2), corner)) ≈ kron(E, block)
            end
        end
        @test_throws ArgumentError DifferenTT.zeros_padding(MPO([ITensor(block, s[1]', s[1])]), siteinds(2, 2), :middle)
    end

    @testset "ndim_operator_mpo (QTT) for every layout" begin
        r = rng()
        for arch in LAYOUTS
            sites = siteinds(arch)
            for ax in eachindex(arch.var_nsites)
                nb = abs(arch.var_nsites[ax])
                op = DifferenTT.mat2mpo(randn(r, 2^nb, 2^nb), siteinds(2, nb))
                lifted = DifferenTT.ndim_operator_mpo(op, ax, sites, arch)
                @test todense(lifted) ≈ reference(todense(op), ax, arch)
                @test only.(siteinds(lifted; plev=0)) == sites
                @test maxlinkdim(lifted) == max(1, maxlinkdim(op))
            end
        end
        arch = QTTArchitecture((2, 3))
        sites = siteinds(arch)
        op = DifferenTT.mat2mpo(randn(r, 4, 4), siteinds(2, 2))
        @test_throws DimensionMismatch DifferenTT.ndim_operator_mpo(op, 1, sites[1:4], arch)
        @test_throws DimensionMismatch DifferenTT.ndim_operator_mpo(op, 2, sites, arch)
        @test_throws DimensionMismatch DifferenTT.ndim_operator_mpo(op, 3, sites, arch)
    end

    @testset "ndim_operator_sum_mpo (QTT) for every layout" begin
        r = rng()
        for arch in LAYOUTS
            sites = siteinds(arch)
            nv = length(arch.var_nsites)
            ops = ntuple(nv) do ax
                nb = abs(arch.var_nsites[ax])
                DifferenTT.mat2mpo(randn(r, 2^nb, 2^nb), siteinds(2, nb))
            end
            S = DifferenTT.ndim_operator_sum_mpo(ops, sites, arch)
            @test todense(S) ≈ sum(reference(todense(ops[ax]), ax, arch) for ax in 1:nv)
            # bond k: the states "nothing placed" and "done", plus the bond of every operator whose sites lie on both
            # sides of k (in the bit order of arch, so reversed for a negative var_nsites)
            expected = map(1:(length(sites)-1)) do k
                2 + sum(1:nv; init=0) do ax
                    p = findall(==(ax), arch.site_map)
                    (length(p) > 1 && p[1] <= k < p[end]) || return 0
                    w = arch.var_nsites[ax] < 0 ? DifferenTT.reverse_mpo(ops[ax]) : ops[ax]
                    return linkdims(w)[searchsortedlast(p, k)]
                end
            end
            @test linkdims(S) == expected
        end
        arch = QTTArchitecture((2, 3))
        ops = (DifferenTT.mat2mpo(randn(r, 4, 4), siteinds(2, 2)), DifferenTT.mat2mpo(randn(r, 8, 8), siteinds(2, 3)))
        @test_throws DimensionMismatch DifferenTT.ndim_operator_sum_mpo(ops, siteinds(arch)[1:4], arch)
    end

    @testset "zeros_padding of a multi-site operator" begin
        sites = siteinds(2, 4)
        M = randn(rng(), 4, 4)
        A = DifferenTT.mat2mpo(M, siteinds(2, 2))
        for (corner, row, col) in ((:upleft, 1, 1), (:upright, 1, 2), (:bottomleft, 2, 1), (:bottomright, 2, 2))
            E = zeros(2, 2)
            E[row, col] = 1.0
            Z = DifferenTT.zeros_padding(A, sites, corner)
            @test todense(Z) ≈ kron(E, E, M)
            @test only.(siteinds(Z; plev=0)) == sites
            @test maxlinkdim(Z) == maxlinkdim(A)
        end
        @test DifferenTT.zeros_padding(A, sites[3:4]) === A
        @test_throws ArgumentError DifferenTT.zeros_padding(A, sites[1:1])
    end
end
