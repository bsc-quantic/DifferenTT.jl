```@meta
CurrentModule = DifferenTT
```

# GPU (experimental)

The tensor-train arithmetic of DifferenTT runs on any array backend that ITensors supports, so the tensors of a field
or an operator may live on a GPU. This support is experimental in v0.1, and we state precisely what it covers.

The keyword `array_type`, which the constructors of tensor trains, fields and operators accept, selects only the array
backend of the tensors, for instance `Array` (the default) or `CuArray` from CUDA.jl. It does not select the element type.
A [`ScalarField`](@ref) or [`VectorField`](@ref) moves its states to its backend, [`backend_type`](@ref) returns the
backend of an object, and the element-wise products of tensor trains on different backends throw an `ArgumentError`.
The array conversions rely on Adapt.jl.

What has been checked and what has not:

  - The backend is propagated by reading the code of the constructors, the operators, the element-wise products and the
    solvers, which keep the cores of the results on the backend of their inputs.
  - No test runs on a GPU in the continuous integration, and the test suite has no CPU stand-in for a GPU array type yet.
    We therefore do not claim that every function runs on a GPU, or that it is faster than on a CPU.
  - Some small matrices, such as the one-dimensional operators, are assembled on the host and moved to the backend
    afterward, so that building an operator is not a GPU computation.

CUDA.jl is not a dependency of DifferenTT. To try a GPU, load it in your own environment and pass `array_type=CuArray`.
Reports of what works and what does not are welcome in the issue tracker.
