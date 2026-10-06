# Root finding and scans on the complex-ω plane.

"""
    muller(f, z0, z1, z2; tol = 1e-13, maxit = 100) -> (root, converged)

Muller's method for a scalar analytic function of a complex variable.
"""
function muller(f, z0, z1, z2; tol = 1e-13, maxit = 100, maxstep = 0.5)
    f0, f1, f2 = f(z0), f(z1), f(z2)
    for _ in 1:maxit
        h1 = z1 - z0
        h2 = z2 - z1
        δ1 = (f1 - f0) / h1
        δ2 = (f2 - f1) / h2
        d = (δ2 - δ1) / (h2 + h1)
        b = δ2 + h2 * d
        Dd = sqrt(b^2 - 4 * f2 * d)
        E = abs(b + Dd) > abs(b - Dd) ? b + Dd : b - Dd
        h = iszero(E) ? oftype(z2, 1e-3) : -2 * f2 / E
        abs(h) > maxstep && (h *= maxstep / abs(h))
        z3 = z2 + h
        z0, z1, z2 = z1, z2, z3
        f0, f1, f2 = f1, f2, f(z3)
        (abs(h) < tol * max(1, abs(z3)) || iszero(f2)) && return z3, true
        isfinite(z3) || return z3, false
    end
    return z2, false
end

"""
    find_root(model, k, ω0; δ = 1e-3, ftol = 1e-8, kwargs...) -> (ω, converged)
    find_root(f, ω0; δ, ftol, kwargs...)

Muller iteration from `ω0` (seeds ω0, ω0+δ, ω0+iδ). `model` is anything with a `dispersion(model, ω, k)` method;
`f` is any analytic function of ω (use it to solve in rescaled frequency units).
"""
function find_root(f::Function, ω0; δ = 1e-3, ftol = 1e-8, kwargs...)
    ω0 = complex(ω0)
    ω, ok = muller(f, ω0, ω0 + δ, ω0 + im * δ; kwargs...)
    # a vanishing step is not enough: D is O(1), require a small residual too
    return ω, ok && isfinite(ω) && abs(f(ω)) < ftol
end
find_root(m, k::Wavevector, ω0; kwargs...) = find_root(ω -> dispersion(m, ω, k), ω0; kwargs...)

"""
    find_modes(model, k; re, im, imin = -Inf, tol_res = 1e-8) -> Vector{ComplexF64}

All roots found from a grid of Muller seeds (`re` × `imag` ranges), de-duplicated and sorted by
real part. Only a seed search: roots outside the seeded region are not found
(use `count_roots` to check a region).
"""
function find_modes(m, k::Wavevector; re = 0.0:0.25:4.5, imag = (0.002, 0.01, 0.03),
                    imin = -Inf, tol_res = 1e-8)
    roots = ComplexF64[]
    for r in re, i in imag
        ω, ok = find_root(m, k, complex(r, i); δ = 1e-3, ftol = tol_res)
        (ok && Base.imag(ω) > imin) || continue
        any(z -> abs(z - ω) < 1e-6, roots) || push!(roots, ω)
    end
    sort!(roots; by = real)
    return roots
end

"""
    track_branch(model, ks, ω0; δ = 1e-4, kwargs...) -> Vector{ComplexF64}
    track_branch(D, ks, ω0; δ, kwargs...)           # D(ω, k) any analytic function

Follow a root along a sequence of wavevectors by continuation (linear extrapolation as seed).
`NaN` is stored where the root finder fails, and tracking restarts from the last good root.
Choose `δ` and `maxstep` (Muller) on the scale of ω.
"""
function track_branch(D::Function, ks::AbstractVector, ω0; δ = 1e-4, kwargs...)
    out = fill(complex(NaN, NaN), length(ks))
    prev = complex(ω0)
    for (i, k) in enumerate(ks)
        seed = i > 2 && isfinite(out[i - 1]) && isfinite(out[i - 2]) ? 2out[i - 1] - out[i - 2] : prev
        ω, ok = find_root(ω -> D(ω, k), seed; δ, kwargs...)
        if ok
            out[i] = ω
            prev = ω
        end
    end
    return out
end
track_branch(m, ks::AbstractVector{<:Wavevector}, ω0; kwargs...) =
    track_branch((ω, k) -> dispersion(m, ω, k), ks, ω0; kwargs...)
# a function with Wavevector parameters is not a model (resolves the ambiguity with the method above)
track_branch(D::Function, ks::AbstractVector{<:Wavevector}, ω0; kwargs...) =
    invoke(track_branch, Tuple{Function,AbstractVector,Any}, D, ks, ω0; kwargs...)

"""
    count_roots(f, re0, re1, im0, im1; maxdarg = 0.1) -> Int

Number of roots of the analytic function `f` inside the rectangle, by the argument principle
(winding of f along the boundary, each edge sampled at 257 points and refined until the phase step is < `maxdarg`).
`f` must have no poles inside; D here is entire in ω.
"""
function count_roots(f, re0, re1, im0, im1; maxdarg = 0.1)
    corners = [complex(re0, im0), complex(re1, im0), complex(re1, im1), complex(re0, im1), complex(re0, im0)]
    total = 0.0
    for e in 1:4
        a, b = corners[e], corners[e + 1]
        ts = collect(range(0, 1; length = 257))
        while true
            vals = [f(a + t * (b - a)) for t in ts]
            dargs = [angle(vals[j + 1] / vals[j]) for j in 1:(length(ts) - 1)]
            bad = findall(abs.(dargs) .> maxdarg)
            isempty(bad) && (total += sum(dargs); break)
            length(ts) > 2^16 && error("count_roots: root on or near the contour")
            new = [(ts[j] + ts[j + 1]) / 2 for j in bad]
            sort!(append!(ts, new))
        end
    end
    return round(Int, total / (2π))
end

"""
    gradient_coefficients(model, ω, k; species = 1) -> (A, B, C)

D is affine in the gradients of each species: D = A + B κn + C κT for the gradients of
`model.species[species]`, the others fixed.
"""
function gradient_coefficients(m::Model, ω, k::Wavevector; species = 1)
    d(κn, κT) = dispersion(with_gradients(m; species, κn, κT), ω, k)
    A = d(0.0, 0.0)
    return A, d(1.0, 0.0) - A, d(0.0, 1.0) - A
end

"""
    neutral_gradients(model, ω, k; species = 1) -> (κn, κT) or nothing

For real ω and k, the gradients (κn, κT) of `species` for which D(ω) = 0 exactly: the
neutral-stability point. Solves the 2×2 real system from Re D = Im D = 0.
"""
function neutral_gradients(m::Model, ω::Real, k::Wavevector; species = 1)
    A, B, C = gradient_coefficients(m, complex(ω), k; species)
    M = [real(B) real(C); Base.imag(B) Base.imag(C)]
    abs(det2(M)) < 1e-14 && return nothing
    κ = M \ [-real(A), -Base.imag(A)]
    return (κn = κ[1], κT = κ[2])
end
det2(M) = M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1]

"""
    harmonic_root(model, k, p; width = 0.4, imag = (0.002, 0.01)) -> ComplexF64 or nothing

The most unstable root within `width` of the Larmor harmonic `p` (seeded on a grid, so this
is a search, not a proof that no root exists; compare with `count_roots`).
"""
function harmonic_root(m, k::Wavevector, p::Integer; width = p == 0 ? 0.3 : 0.4,
                       imag = (0.002, 0.01), step = 0.1)
    roots = find_modes(m, k; re = (p - width + 0.1):step:(p + width), imag)
    roots = filter(z -> abs(real(z) - p) < width, roots)
    return isempty(roots) ? nothing : roots[argmax(Base.imag.(roots))]
end

"""
    critical_kappaT(model, k, κn, p; species = 1, w = 0.35, n = 1501) -> Float64

Smallest positive κT of `species` for which D has a root at real ω ∈ (p-w, p+w) at the given κn
(neutral stability; `Inf` if none). D is affine in κT, so for each real ω the unique κT is
`-(A + B κn)/C`, and a neutral point needs it to be real. `κn` may be a vector (the
coefficients on the ω grid are then computed once).
"""
function critical_kappaT(m::Model, k::Wavevector, κn, p::Integer; species = 1, w = 0.35, n = 1501)
    coef(ω) = gradient_coefficients(m, complex(ω), k; species)
    ωs = collect(range(p - w, p + w; length = n))
    cs = coef.(ωs)
    crit(κn) = _critical_kappaT(coef, κn, ωs, cs)
    return κn isa Number ? crit(κn) : crit.(κn)
end

# Bisect Im κT(ω) = 0 in each sign-change interval of the precomputed grid.
function _critical_kappaT(coef, κn, ωs, cs)
    κ(c) = -(c[1] + c[2] * κn) / c[3]
    vals = κ.(cs)
    best = Inf
    for i in 1:(length(ωs) - 1)
        Base.imag(vals[i]) * Base.imag(vals[i + 1]) < 0 || continue
        lo, hi = ωs[i], ωs[i + 1]
        flo = Base.imag(vals[i])
        for _ in 1:60
            mid = (lo + hi) / 2
            fm = Base.imag(κ(coef(mid)))
            if fm * flo < 0
                hi = mid
            else
                lo, flo = mid, fm
            end
        end
        v = real(κ(coef((lo + hi) / 2)))
        v > 0 && (best = min(best, v))
    end
    return best
end
