# Third-party notices

DifferenTT.jl is released under the MIT license (see [`LICENSE`](LICENSE)). Parts of it are adapted from, or follow,
the work of others, as listed below.

## KrylovKit.jl

The preconditioned conjugate gradient and GMRES solvers of
[`src/LinearSolvers/KrylovSolvers/cg.jl`](src/LinearSolvers/KrylovSolvers/cg.jl) and
[`src/LinearSolvers/KrylovSolvers/gmres.jl`](src/LinearSolvers/KrylovSolvers/gmres.jl)
(`DifferenTT.preconditioned_linsolve`) are adapted from the CG and GMRES implementations of
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl), to which we added the preconditioner. The tensor-train
solvers of `ttcg.jl` and `ttgmres.jl` follow the same structure. KrylovKit.jl carries the following notice:

> The KrylovKit.jl package is licensed under the MIT "Expat" License:
>
> > Copyright (c) 2017: Jutho.
> >
> > Permission is hereby granted, free of charge, to any person obtaining a copy
> > of this software and associated documentation files (the "Software"), to deal
> > in the Software without restriction, including without limitation the rights
> > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> > copies of the Software, and to permit persons to whom the Software is
> > furnished to do so, subject to the following conditions:
> >
> > The above copyright notice and this permission notice shall be included in all
> > copies or substantial portions of the Software.
> >
> > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> > SOFTWARE.

## TT-Toolbox

The exact QTT inverse of the one-dimensional Dirichlet Laplacian (`_dd_exact_inverse_laplacian` in
[`src/LinearSolvers/preconditioners.jl`](src/LinearSolvers/preconditioners.jl), used by
`inverse_laplacian_preconditioner`) follows the construction of the MATLAB
[TT-Toolbox](https://github.com/oseledets/TT-Toolbox) rather than the formulas of the paper it implements,
V. A. Kazeev and B. N. Khoromskij, *Low-rank explicit QTT representation of the Laplace operator and its inverse*,
SIAM J. Matrix Anal. Appl. 33 (2012) 742–758. TT-Toolbox carries the following notice:

> Copyright (C) 2009-2012 Ivan Oseledets, Sergey Dolgov, Vladimir Kazeev,
> Thomas Mach, Olga Lebedeva, Dmitry Savostyanov, Pavel Zhlobich, Le Song
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in
> the Software without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
> of the Software, and to permit persons to whom the Software is furnished to do
> so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Recursive sketched interpolation (RSI)

The element-wise products and maps of [`src/TensorTrains/elementwise.jl`](src/TensorTrains/elementwise.jl)
(`tt_mul`, `_rsi_tt_mul`, `elementwise_function`) implement the recursive sketched interpolation algorithm of
Meng, Khoo, Li & Stoudenmire, arXiv:2602.17974. Our implementation started from a rough
implementation that its authors shared with us by private communication, and that they agreed to have published.
The pivoted rank-revealing LDU decomposition (`prrldu`) and the interpolative decompositions (`interpolative`) in the
same file came with that implementation.
