# GC memory ballast, shared by the workers of fai_benchmarks/.
#
# Julia 1.12 derives its GC thresholds from the heap size, so a worker with a small live heap collects very often. The
# ballast is a large, never-written buffer: the GC counts it as heap, which raises its thresholds, but its pages are never
# touched, so it costs no resident memory. This cuts the GC time; without it, GC time diluted the measured speedups and
# made the comparisons less reliable.
#
# Usage: set GC_BALLAST_GB to the size in GB (0, the default here, disables it; the submit.sh scripts default to 7).
const GC_BALLAST_GB = parse(Float64, get(ENV, "GC_BALLAST_GB", "0"))
const GIB = 2^30
gc_max_memory() = ccall(:jl_gc_get_max_memory, UInt64, ())

function allocate_ballast(gb::Float64)
    gb > 0 || return UInt8[]
    max_gb = 0.4 * 0.8 * gc_max_memory() / GIB   # keep well below the GC's heap cap
    if gb > max_gb
        @warn "Clamping GC ballast" requested_gb=gb max_gb
        gb = max_gb
    end
    try
        return Vector{UInt8}(undef, round(Int, gb * GIB))
    catch e
        e isa OutOfMemoryError || rethrow()
        @warn "Could not reserve the GC ballast, running without it" gb
        return UInt8[]
    end
end

# Must stay a global: if it is ever freed, the effect disappears. Never write into it.
const GC_BALLAST = allocate_ballast(GC_BALLAST_GB)
@info "GC ballast" gb=length(GC_BALLAST) / GIB heap_cap_gb=gc_max_memory() / GIB
