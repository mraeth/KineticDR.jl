# Gyrokinetic response with a magnetic drift. The v∥ average of the resonant denominator is
# exact (two Z functions) for any Im ω; v⊥ enters through the magnetic moment μ = v⊥²/(2T/m).
# The μ quadrature is not: below the real axis it replaces the drift-resonance cut of the μ integral
# by `nmu` cuts (see [`GyrokineticDrift`](@ref)).

"""
    quadratic_average(n0, n2, a, b, e) -> ⟨(n0 + n2 s²) / (a s² - b s + e)⟩_s,  s ~ N(0, 1)

Exact for complex `e` (analytic in e, including Im e < 0, with the branch cut of the root
pair along the negative imaginary axis of e - b²/4a).

The roots r± = (b ± δ)/2a of the denominator are labelled by continuation from Im e > 0, where
r₋ lies in the upper and r₊ in the lower half plane: ⟨1/(s-r₋)⟩ = Z(r₋/√2)/√2 and
⟨1/(s-r₊)⟩ = -Z(-r₊/√2)/√2. Both are entire in r, so the result continues to Im e < 0.
Partial fractions give [⟨N/(s - r₊)⟩ - ⟨N/(s - r₋)⟩]/δ with N = n0 + n2 s², i.e. `linear_moments`
at -r₊ and r₋, without cancellation as a → 0 (r₊ → ∞, the linear denominator).
"""
function quadratic_average(n0, n2, a::Real, b::Real, e)
    if a == 0
        I0, I2 = linear_moments(e, b)
        return n0 * I0 + n2 * I2
    end
    ω′ = e - b^2 / (4a)                       # δ² = -4a ω′
    δ = a > 0 ? -2im * sqrt(a) * sqrt_cut(ω′) : 2 * sqrt(-a) * sqrt_cut(ω′)
    # r₊ r₋ = e/a: the smaller root from the product, not the difference
    if abs(b + δ) >= abs(b - δ)
        rp, rm = (b + δ) / (2a), 2e / (b + δ)
    else
        rp, rm = 2e / (b - δ), (b - δ) / (2a)
    end
    P0, P2 = linear_moments(-rp, 1.0)         # ⟨1/(s - r₊)⟩, ⟨s²/(s - r₊)⟩
    M0, M2 = linear_moments(rm, 1.0)          # -⟨1/(s - r₋)⟩, -⟨s²/(s - r₋)⟩
    return (n0 * (P0 + M0) + n2 * (P2 + M2)) / δ
end

# √z with the cut on the negative imaginary axis: analytic across the real axis.
sqrt_cut(z) = (r = sqrt(-im * z); r * sqrt(complex(im)))

# ⟨1/(ω - q s)⟩ and ⟨s²/(ω - q s)⟩ for s ~ N(0, 1), continued to Im ω < 0 like Z: -a Z(ζ) and -2a ζ (1 + ζZ(ζ)),
# a = 1/(√2|q|), ζ = aω; 1/ω for q = 0.
function linear_moments(ω, q)
    q == 0 && return (1 / ω, 1 / ω)
    a = 1 / (sqrt(2.0) * abs(q))
    ζ = a * ω
    return -a * plasma_Z(ζ), -2a * ζ * one_plus_zZ(ζ)
end

"Gauss–Laguerre nodes and weights for ∫₀^∞ e^{-μ} f(μ) dμ (Golub–Welsch)."
function gauss_laguerre(n::Integer)
    J = SymTridiagonal([2k + 1.0 for k in 0:(n - 1)], [Float64(k) for k in 1:(n - 1)])
    E = eigen(J)
    return E.values, E.vectors[1, :] .^ 2
end

"""
    GyrokineticDrift(; c, cperp = c, alpha = 0, nmu = 128)

Gyrokinetic response (J₀², no cyclotron harmonics) with the magnetic drift
ω_D = -k_y (T_s/q_s) (c v∥²/v_ts² + c⊥ μ), c = 1/(B R) for curvature and ∇B (`c`, `cperp` in 1/ρ_i,
B = 1 units). `alpha` multiplies ω_D in the drive numerator: 0 is gyrokinetic, 1 is bslLD's
prescribed drift (the drift does no work against E). v⊥ by `nmu`-point Gauss–Laguerre in μ.
`c = cperp = 0` is `Gyrokinetic` up to the μ quadrature, which must resolve the oscillation of J₀²(k⊥ρ√(2μ)):
`nmu` ≳ (k⊥ρ_s)². At k⊥ρ_s = 10, 64 nodes are 15% off `Gyrokinetic`, the default 128 agree to 1e-14.

The μ integrand has a pole at ω + k_y c⊥ μ ≈ 0, i.e. a distance ~|ω/(k_y c⊥)| from the real axis, which
Gauss–Laguerre resolves only geometrically: for k_y c = 0.01, ω = -0.03 + 0.01i the relative error of the
response is 1e-3 (nmu = 24), 2e-4 (40), 1.4e-5 (80), 3e-7 (160), 1e-9 (320). Check `nmu` for ω_D ~ ω.

Below the real axis: each node μ_k has a branch point at real ω = k_z² v_t²/(4 k_y τ c) - k_y τ c⊥ μ_k
(lab units, τ = T_s/q_s) with its cut
hanging down, so the μ integral's continuum of drift resonances becomes `nmu` cuts. For Im ω < 0 with Re ω
in the range of these points the response is a quadrature artefact (depends on `nmu`), not the continuation
of the μ integral. Trust Im ω > 0; check `nmu` convergence for damped drift modes.
"""
struct GyrokineticDrift <: AbstractResponse
    c::Float64
    cperp::Float64
    alpha::Float64
    mu::Vector{Float64}
    wmu::Vector{Float64}
end
function GyrokineticDrift(; c, cperp = c, alpha = 0.0, nmu = 128)
    mu, wmu = gauss_laguerre(nmu)
    return GyrokineticDrift(float(c), float(cperp), float(alpha), mu, wmu)
end

gordeyev(::GyrokineticDrift, args...) =
    throw(ArgumentError("GyrokineticDrift has no Gordeyev sum; use nonadiabatic_response"))

# Own units (|Ω_s| = ρ_ts = v_ts = 1): ω_D = -k_y (c s² + c⊥ μ), `c` already the species' own c. The v∥ average at
# one μ, W_μ = ⟨(ω - α ω_D - ω_*)/(ω - k_z v∥ - ω_D)⟩, shared by the local response and the radial kernel.
function drift_vpar(r::GyrokineticDrift, ω, k::Wavevector, ωn, ωT, μ)
    a = k.ky * r.c
    n0 = ω - ωn - ωT * (μ - 1.5) + r.alpha * k.ky * r.cperp * μ
    n2 = -ωT / 2 + r.alpha * a
    return quadratic_average(n0, n2, a, abs(k.kz), ω + k.ky * r.cperp * μ)
end

function nonadiabatic_response(r::GyrokineticDrift, ω, k::Wavevector, ωn::Number, ωT::Number)
    kp = kperp(k)
    return sum(w * besselj0(kp * sqrt(2μ))^2 * drift_vpar(r, ω, k, ωn, ωT, μ) for (μ, w) in zip(r.mu, r.wmu))
end

# ω_D/|Ω_s| = -k_y,s sgn(q_s) ρ_ts (c s² + c⊥ μ): c is a lab quantity, rescale to the species' units.
function own_units(r::GyrokineticDrift, s::Species)
    f = sign(s.q) * thermal_gyroradius(s)
    return GyrokineticDrift(r.c * f, r.cperp * f, r.alpha, r.mu, r.wmu)
end
