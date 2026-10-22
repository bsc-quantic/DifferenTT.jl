# Archive

This folder keeps scripts that are no longer part of the maintained experiments. Their contents are **unmaintained** and
may not run with the current API of DifferenTT: they are not tested, they are not covered by the smoke mode, and they are
not edited when the library changes.

| Folder | What it was |
|---|---|
| `tt_poisson` | `poissonNd.jl`, the N-dimensional TT Poisson demonstration (Sinc and FAI preconditioners), from which `examples/03` derives. Its FAI call is updated to `fai_preconditioner`, but it is not run in smoke mode |
| `qtt_poisson` | Early demonstrations of the QTT Poisson solver in one and two dimensions (`poisson1d.jl`, `poisson2d.jl`, and the figures of the 2D case). The tutorials `examples/01` and `examples/02` replace them |
