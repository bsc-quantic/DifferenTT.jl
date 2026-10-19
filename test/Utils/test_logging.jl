using Test
using DifferenTT
using JLD2
using TimerOutputs

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

    # Test Nothing logger
    nothing_logger = nothing
    # Should not throw errors
    log_data(nothing_logger, "none_file", "key", 1)
    log_data(nothing_logger, "none_file", data_dict)
    @test !isfile(joinpath(log_dir, "none_file.jld2"))

    # Test timeit_to existence
    @test timeit_to isa TimerOutput
end
