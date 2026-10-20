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

# ⟨1/(ω - q s)⟩ and ⟨s²/(ω - q s)⟩ for s ~ N(0, 1), continued to Im ω < 0 like Z.
function linear_moments(ω, q)
    q = abs(q)
    ζ = ω / (sqrt(2.0) * q)
    if !(abs(ζ) < 6)                              # also q == 0: asymptotic series Σ (2n∓1)!! r^n, summed to
        r = (q / ω)^2                             # its smallest term (≲ e^{-36} at |ζ| = 6)
        t0, t2 = one(r), one(r)
        s0, s2 = t0, t2
        for n in 1:40
            t0 *= (2n - 1) * r
            t2 *= (2n + 1) * r
            s0 += t0
            s2 += t2
            abs(t2) < 1e-17 * abs(s2) && break
        end
        I0, I2 = s0 / ω, s2 / ω
        if q > 0 && imag(ζ) < 0                   # Landau term 2i√π e^{-ζ²} of Z below the real axis
            L = -im * sqrt(2π) / q * exp(-ζ^2)
            I0 += L
            I2 += (ω / q)^2 * L
        end
        return I0, I2
    end
    I0 = -plasma_Z(ζ) / (sqrt(2.0) * q)
    return I0, (ω / q)^2 * I0 - ω / q^2
end

"Gauss–Laguerre nodes and weights for ∫₀^∞ e^{-μ} f(μ) dμ (Golub–Welsch)."
function gauss_laguerre(n::Integer)
    J = SymTridiagonal([2k + 1.0 for k in 0:(n - 1)], [Float64(k) for k in 1:(n - 1)])
    E = eigen(J)
    return E.values, E.vectors[1, :] .^ 2
end

"""
    GyrokineticDrift(; c, cperp = c, alpha = 0, nmu = 64)

Gyrokinetic response (J₀², no cyclotron harmonics) with the magnetic drift
ω_D = -k_y (T_s/q_s) (c v∥²/v_ts² + c⊥ μ), c = 1/(B R) for curvature and ∇B (`c`, `cperp` in 1/ρ_i,
B = 1 units). `alpha` multiplies ω_D in the drive numerator: 0 is gyrokinetic, 1 is bslLD's
prescribed drift (the drift does no work against E). v⊥ by `nmu`-point Gauss–Laguerre in μ.
`c = cperp = 0` is `Gyrokinetic` up to the μ quadrature.

The μ integrand has a pole at ω + k_y c⊥ μ ≈ 0, i.e. a distance ~|ω/(k_y c⊥)| from the real axis, which
Gauss–Laguerre resolves only geometrically: for k_y c = 0.01, ω = -0.03 + 0.01i the relative error of the
response is 1e-3 (nmu = 24), 2e-4 (40), 1.4e-5 (80), 3e-7 (160), 1e-9 (320). Check `nmu` for ω_D ~ ω.

Below the real axis: each node μ_k has a branch point at real ω = k_z²/(4 k_y c) - k_y c⊥ μ_k with its cut
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
function GyrokineticDrift(; c, cperp = c, alpha = 0.0, nmu = 64)
    mu, wmu = gauss_laguerre(nmu)
    return GyrokineticDrift(float(c), float(cperp), float(alpha), mu, wmu)
end

gordeyev(::GyrokineticDrift, args...) =
    error("GyrokineticDrift has no Gordeyev sum; use nonadiabatic_response")

# Own units (|Ω_s| = ρ_ts = 1): ω_D = -k_y (c s² + c⊥ μ). `c` here is already the species' own c.
function nonadiabatic_response(r::GyrokineticDrift, ω, k::Wavevector, ωn::Number, ωT::Number)
    ky, kz, kp = k.ky, k.kz, kperp(k)
    a, b = ky * r.c, abs(kz)
    acc = zero(complex(ω))
    for (μ, w) in zip(r.mu, r.wmu)
        e = ω + ky * r.cperp * μ
        n0 = ω - ωn - ωT * (μ - 1.5) + r.alpha * ky * r.cperp * μ
        n2 = -ωT / 2 + r.alpha * a
        acc += w * besselj0(kp * sqrt(2μ))^2 * quadratic_average(n0, n2, a, b, e)
    end
    return acc
end

# ω_D/|Ω_s| = -k_y,s sgn(q_s) ρ_ts (c s² + c⊥ μ): c is a lab quantity, rescale to the species' units.
function own_units(r::GyrokineticDrift, s::Species)
    f = sign(s.q) * thermal_gyroradius(s)
    return GyrokineticDrift(r.c * f, r.cperp * f, r.alpha, r.mu, r.wmu)
end
