using Test
using DifferenTT
using ITensors, ITensorMPS, LinearAlgebra

# If these are not exported by DifferenTT, we explicitly import them
import DifferenTT: QTTArchitecture, TTArchitecture

@testset "TensorTrains.Architecture" begin
    @testset "QTTArchitecture" begin
        # Default initialization (blocked, natural order)
        var_nsites = (3, 2, 4)
        arch = QTTArchitecture(var_nsites)

        @test arch.var_nsites == var_nsites
        @test arch.is_interleaved == false
        @test arch.var_map == Dict(1=>1, 2=>2, 3=>3)
        @test arch.site_map == [1, 1, 1, 2, 2, 3, 3, 3, 3]

        # Interleaved mode
        arch_interleaved = QTTArchitecture(var_nsites, interleaved=true)
        @test arch_interleaved.is_interleaved == true
        @test arch_interleaved.var_map == Dict(1=>1, 2=>2, 3=>3)
        # Expected site_map: 1, 2, 3, 1, 2, 3, 1, 3, 3
        @test arch_interleaved.site_map == [1, 2, 3, 1, 2, 3, 1, 3, 3]

        # Blocked with custom var_map (order 2, 3, 1)
        # var_map maps Logical Dim => Physical Block Position
        custom_map = Dict(1=>3, 2=>1, 3=>2)
        arch_custom = QTTArchitecture(var_nsites, custom_map)
        @test arch_custom.is_interleaved == false
        @test arch_custom.var_map == custom_map
        # Block 1 is var 2 (2 sites)
        # Block 2 is var 3 (4 sites)
        # Block 3 is var 1 (3 sites)
        @test arch_custom.site_map == [2, 2, 3, 3, 3, 3, 1, 1, 1]

        # Negative nsites tests
        var_nsites_neg = (-3, 2)
        arch_neg = QTTArchitecture(var_nsites_neg)
        @test arch_neg.var_nsites == (-3, 2)
        @test arch_neg.site_map == [1, 1, 1, 2, 2] # Same mapping, sign used elsewhere

        # Errors
        @test_throws ArgumentError QTTArchitecture((2, 2), Dict(1=>1))
        @test_throws ArgumentError QTTArchitecture((2, 2), Dict(1=>1, 3=>2))
        @test_throws ArgumentError QTTArchitecture((2, 2), Dict(1=>1, 2=>3))
    end

    @testset "TTArchitecture" begin
        # Default initialization
        var_dims = (10, 20, 30)
        arch = TTArchitecture(var_dims)

        @test arch.var_dims == var_dims
        @test arch.var_map == Dict(1=>1, 2=>2, 3=>3)
        @test arch.site_map == [1, 2, 3]

        # Custom var_map
        custom_map = Dict(1=>3, 2=>1, 3=>2)
        arch_custom = TTArchitecture(var_dims, custom_map)
        @test arch_custom.var_dims == var_dims
        @test arch_custom.var_map == custom_map
        @test arch_custom.site_map == [2, 3, 1]

        # Errors
        @test_throws ArgumentError TTArchitecture((10, 10), Dict(1=>1))
        @test_throws ArgumentError TTArchitecture((10, 10), Dict(1=>1, 3=>2))
        @test_throws ArgumentError TTArchitecture((10, 10), Dict(1=>1, 2=>3))
    end

    # Regression test for B4: the export list named the misspelt sited_dim instead of site_dim
    @testset "site_dim is exported" begin
        @test :site_dim in names(DifferenTT)
        @test !(:sited_dim in names(DifferenTT))
        @test site_dim(QTTArchitecture((3, 2)), 4) == 2
        @test [site_dim(TTArchitecture((5, 7)), n) for n in 1:2] == [5, 7]
        @test [site_dim(TTArchitecture((5, 7), Dict(1 => 2, 2 => 1)), n) for n in 1:2] == [7, 5]
    end

    # Regression test: site_dims of a TTArchitecture returned the dimensions in variable order, so siteinds(arch) built
    # sites of the wrong dimension whenever var_map permuted variables of different dimensions
    @testset "site_dims follows the site order" begin
        arch = TTArchitecture((5, 7, 3), Dict(1 => 3, 2 => 1, 3 => 2))
        @test site_dims(arch) == (7, 3, 5)
        @test collect(site_dims(arch)) == [site_dim(arch, n) for n in 1:3]
        @test dim.(siteinds(arch)) == [7, 3, 5]
        @test variable_dims(arch) == (5, 7, 3)
        sites = siteinds(arch)
        op = DifferenTT.ndim_operator_mpo(collect(reshape(1.0:49.0, 7, 7)), 2, sites, arch)
        @test DifferenTT.contract_mpo(op) ≈ kron(collect(reshape(1.0:49.0, 7, 7)), I(3), I(5))
    end

    # Every query of an architecture, for the four QTT layouts and for TT (blocked, interleaved, permuted by var_map,
    # reversed by a negative var_nsites, and both)
    @testset "queries for every layout" begin
        layouts = [
            (QTTArchitecture((2, 3)), [1, 1, 2, 2, 2]),
            (QTTArchitecture((2, 3); interleaved=true), [1, 2, 1, 2, 2]),
            (QTTArchitecture((2, 3), Dict(1 => 2, 2 => 1)), [2, 2, 2, 1, 1]),
            (QTTArchitecture((2, 3), Dict(1 => 2, 2 => 1); interleaved=true), [2, 1, 2, 1, 2]),
            (QTTArchitecture((-2, 3)), [1, 1, 2, 2, 2]),
            (QTTArchitecture((2, -3), Dict(1 => 2, 2 => 1); interleaved=true), [2, 1, 2, 1, 2]),
        ]
        for (arch, site_map) in layouts
            @test arch.site_map == site_map
            @test total_sites(arch) == 5
            @test [nsites_per_variable(arch, v) for v in 1:2] == [2, 3]
            @test variable_dims(arch) == (4, 8)
            @test collect(site_dims(arch)) == fill(2, 5) == [site_dim(arch, n) for n in 1:5]
            # each variable has 2^nsites values, so the grid has prod(variable_dims) = 2^total_sites points
            @test prod(variable_dims(arch)) == prod(site_dims(arch))
            sites = siteinds(arch)
            @test dim.(sites) == fill(2, 5)
            @test all(hastags(s, "Site") && hastags(s, "n=$n") for (n, s) in enumerate(sites))
            @test all(plev(s) == 0 for s in sites)
        end

        # interleaving follows the physical order of var_map and skips variables that ran out of sites
        arch = QTTArchitecture((2, -3, 1), Dict(1 => 2, 2 => 3, 3 => 1); interleaved=true)
        @test arch.site_map == [3, 1, 2, 1, 2, 2]
        @test variable_dims(arch) == (4, 8, 2)

        # one variable with one site
        arch = QTTArchitecture((1,))
        @test arch.site_map == [1]
        @test total_sites(arch) == 1 && variable_dims(arch) == (2,) && dim.(siteinds(arch)) == [2]

        # TT: unequal dimensions, permuted, one site
        for (arch, sdims) in [
            (TTArchitecture((3, 5, 4)), (3, 5, 4)),
            (TTArchitecture((3, 5, 4), Dict(1 => 2, 2 => 3, 3 => 1)), (4, 3, 5)),
            (TTArchitecture((6,)), (6,)),
        ]
            N = length(sdims)
            @test total_sites(arch) == N
            @test all(nsites_per_variable(arch, v) == 1 for v in 1:N)
            @test site_dims(arch) == sdims
            @test prod(variable_dims(arch)) == prod(sdims)
            @test dim.(siteinds(arch)) == collect(sdims)
            @test [site_dim(arch, arch.var_map[v]) for v in 1:N] == collect(variable_dims(arch))
        end

        # keyword arguments reach the Index constructor
        @test all(plev(s) == 1 for s in siteinds(QTTArchitecture((2,)); plev=1))
        @test site_dims(TTArchitecture((3, 4), nothing)) == (3, 4)
    end

    @testset "site_indices and link_indices" begin
        sites = DifferenTT.site_indices([3, 2, 4])
        @test dim.(sites) == [3, 2, 4]
        @test all(hastags(s, "Site,n=$n") for (n, s) in enumerate(sites))
        links = DifferenTT.link_indices([2, 5])
        @test dim.(links) == [2, 5]
        @test all(hastags(l, "Link,l=$n") for (n, l) in enumerate(links))
        @test allunique(vcat(sites, links))
        @test isempty(DifferenTT.link_indices(Int[]))
    end

    @testset "errors" begin
        @test_throws ArgumentError QTTArchitecture((2, 2), Dict(1 => 1, 2 => 1))
        @test_throws ArgumentError QTTArchitecture((2, 2), Dict(1 => 1, 2 => 2, 3 => 3))
        @test_throws ArgumentError TTArchitecture((3, 3), Dict(1 => 2, 2 => 2))
        @test_throws BoundsError site_dim(TTArchitecture((3, 3)), 3)
        @test_throws BoundsError nsites_per_variable(QTTArchitecture((2, 3)), 3)
    end
end
