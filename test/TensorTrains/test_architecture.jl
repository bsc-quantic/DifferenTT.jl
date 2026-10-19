using Test
using DifferenTT

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
end
