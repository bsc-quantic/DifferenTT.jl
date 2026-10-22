# [Tensor trains](@id api_tensortrains)

Architectures, constructors, operations, element-wise products, and cross interpolation in the TT and QTT formats. The [manual pages](@ref "Architectures and conventions") describe the conventions.

Exported names and names declared `public` are listed here; the other names of the area are on the
[Internals](@ref) page.

```@autodocs
Modules = [DifferenTT]
Pages = ["TensorTrains/architecture.jl",
         "TensorTrains/utils.jl",
         "TensorTrains/constructors.jl",
         "TensorTrains/operations.jl",
         "TensorTrains/qttconstructors.jl",
         "TensorTrains/qttoperations.jl",
         "TensorTrains/elementwise.jl",
         "TensorTrains/tci.jl"]
Private = false
```
