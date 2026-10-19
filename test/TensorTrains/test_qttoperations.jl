using DifferenTT
using Test
using ITensors
using LinearAlgebra
using ITensorMPS

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
        mps_test = random_mps(full_sites)
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
        mps_test_int = random_mps(full_sites_int)
        @test norm(apply(lifted_mpo_int, mps_test_int)) > 0
    end

    # Regression test: a variable with a single site threw a BoundsError (its core has no link indices)
    @testset "ndim_operator_mpo with one-site variables" begin
        # Dense reference: the operator acts on the value of variable `ax`, read from its bits (most significant first,
        # reversed for a negative var_nsites), and is the identity on the values of the other variables
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
end
