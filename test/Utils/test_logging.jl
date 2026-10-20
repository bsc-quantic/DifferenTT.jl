using Test
using DifferenTT
using ITensors
using ITensorMPS
using JLD2
using TimerOutputs

include(joinpath(@__DIR__, "..", "testutils.jl"))
using .TestUtils

@testset "Logging" begin
    # Test initialization
    log_dir = mktempdir()
    logger = init_logger(log_dir)

    @test isdir(log_dir)
    @test logger isa SimpleLogger
    @test logger.log_dir == log_dir

    # Test log_data with key-value
    log_data(logger, "test_file.jld2", "test_key", 42)
    @test isfile(joinpath(log_dir, "test_file.jld2"))

    jldopen(joinpath(log_dir, "test_file.jld2"), "r") do file
        @test file["test_key"] == 42
    end

    # Test log_data appending
    log_data(logger, "test_file.jld2", "test_key2", "hello")
    jldopen(joinpath(log_dir, "test_file.jld2"), "r") do file
        @test file["test_key"] == 42
        @test file["test_key2"] == "hello"
    end

    # Test log_data with Dict
    data_dict = Dict("dict_key1" => 1.0, "dict_key2" => [1, 2, 3])
    log_data(logger, "test_dict", data_dict) # without .jld2 extension
    @test isfile(joinpath(log_dir, "test_dict.jld2"))

    jldopen(joinpath(log_dir, "test_dict.jld2"), "r") do file
        @test file["dict_key1"] == 1.0
        @test file["dict_key2"] == [1, 2, 3]
    end

    # Test log_data with Dict appending
    data_dict2 = Dict("dict_key3" => "world")
    log_data(logger, "test_dict", data_dict2)
    jldopen(joinpath(log_dir, "test_dict.jld2"), "r") do file
        @test file["dict_key1"] == 1.0
        @test file["dict_key3"] == "world"
    end

    # Regression test: an existing key was not updated, JLD2 threw "already present"
    log_data(logger, "test_file.jld2", "test_key", 43)
    log_data(logger, "test_dict", Dict("dict_key1" => 2.0, "dict_key4" => 4))
    jldopen(joinpath(log_dir, "test_file.jld2"), "r") do file
        @test file["test_key"] == 43
        @test file["test_key2"] == "hello"
    end
    jldopen(joinpath(log_dir, "test_dict.jld2"), "r") do file
        @test file["dict_key1"] == 2.0
        @test file["dict_key3"] == "world"
        @test file["dict_key4"] == 4
    end

    # Test Nothing logger
    nothing_logger = nothing
    # Should not throw errors
    log_data(nothing_logger, "none_file", "key", 1)
    log_data(nothing_logger, "none_file", data_dict)
    @test !isfile(joinpath(log_dir, "none_file.jld2"))

    # Test timeit_to existence
    @test timeit_to isa TimerOutput

    # Regression test for B4: enable_debug_timings! was exported but not defined
    @testset "debug timings" begin
        @test isdefined(DifferenTT, :enable_debug_timings!)
        try
            enable_debug_timings!()
            @test Base.invokelatest(DifferenTT.timeit_debug_enabled)
            disable_debug_timings!()
            @test !Base.invokelatest(DifferenTT.timeit_debug_enabled)
        finally
            disable_debug_timings!()
        end
    end
end

@testset "init_logger creates the directory and its parents" begin
    root = mktempdir()
    nested = joinpath(root, "results", "cavity", "run1")
    @test !isdir(nested)
    logger = init_logger(nested)
    @test isdir(nested)
    @test logger isa DifferenTT.SimpleLogger
    @test logger isa DifferenTT.Logger
    @test logger.log_dir == nested
    # An existing directory (and the files in it) is left alone
    log_data(logger, "keep", "x", 1)
    @test init_logger(nested).log_dir == nested
    @test isfile(joinpath(nested, "keep.jld2"))
    # A path that is a file cannot become a directory
    blocked = joinpath(root, "blocked")
    write(blocked, "not a directory")
    @test_throws Exception init_logger(joinpath(blocked, "sub"))
end

@testset "log_data round trips" begin
    dir = mktempdir()
    logger = init_logger(dir)
    sites = siteinds("S=1/2", 4)
    psi = random_mps(rng(1), sites; linkdims=3)
    A = rand(rng(2), 3, 4)

    # key/value method: scalars, arrays, strings, tuples and MPS survive the file unchanged
    log_data(logger, "values", "number", 3.5)
    log_data(logger, "values", "array", A)
    log_data(logger, "values", "tuple", (1, "two", 3.0))
    log_data(logger, "values", "mps", psi)
    jldopen(joinpath(dir, "values.jld2"), "r") do file
        @test Set(keys(file)) == Set(["number", "array", "tuple", "mps"])
        @test file["number"] === 3.5
        @test file["array"] == A
        @test file["tuple"] == (1, "two", 3.0)
        @test mps_equal(file["mps"], psi)
        @test norm(file["mps"]) ≈ norm(psi)
    end

    # Dict method: the same keys, with the extension added once, and later calls add or replace keys
    log_data(logger, "table.jld2", Dict("a" => 1, "b" => [1.0, 2.0]))
    @test !isfile(joinpath(dir, "table.jld2.jld2"))
    log_data(logger, "table", Dict("b" => [3.0], "c" => psi))
    jldopen(joinpath(dir, "table.jld2"), "r") do file
        @test Set(keys(file)) == Set(["a", "b", "c"])
        @test file["a"] == 1
        @test file["b"] == [3.0] # replaced by the second call
        @test mps_equal(file["c"], psi)
    end

    # Overwriting with a value of another type
    log_data(logger, "values", "number", "now a string")
    jldopen(joinpath(dir, "values.jld2"), "r") do file
        @test file["number"] == "now a string"
        @test file["array"] == A
    end

    # Two files in one directory are independent, and nothing else is written
    @test sort(readdir(dir)) == ["table.jld2", "values.jld2"]

    # The loggers of the evolutions store a VectorField: its states are read back unchanged
    arch = QTTArchitecture((2, 2))
    bc = BoundaryCond1d(PeriodicBC(), PeriodicBC())
    bcs = BoundaryConditions(bc, bc)
    field_sites = siteinds(arch)
    sx, sy = random_mps(rng(3), field_sites), random_mps(rng(4), field_sites)
    field = VectorField((sx, sy), arch, (bcs, bcs))
    log_data(logger, "fields", "u_fields", field)
    jldopen(joinpath(dir, "fields.jld2"), "r") do file
        loaded = file["u_fields"]
        @test length(loaded) == 1
        @test mps_equal(last(loaded), (sx, sy))
    end
end
