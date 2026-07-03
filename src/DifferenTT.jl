module DifferenTT

using Adapt
using ITensors
using ITensorMPS

# Include TensorTrain related files
include("TensorTrains/architecture.jl")
include("TensorTrains/utils.jl")

# Include PDE Problem related files
include("PdeProblem/boundaryconditions.jl")
include("PdeProblem/grid.jl")
include("PdeProblem/scalarfields.jl")
include("PdeProblem/vectorfields.jl")

# Export types and functions
export Architecture, QTTArchitecture, TTArchitecture
export nsites_per_variable, total_sites, site_dims, variable_dims
export backend_type
export BoundaryCond, PeriodicBC, OpenBC, DirichletBC, NeumannBC, BoundaryCond1d, BoundaryConditions
export ScalarField, VectorField, Grid

end

