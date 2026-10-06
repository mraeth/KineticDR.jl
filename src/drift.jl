# Gyrokinetic response with a magnetic drift. The v∥ average of the resonant denominator is
# exact (two Z functions) for any Im ω; v⊥ enters through the magnetic moment μ = v⊥²/(2T/m).

"""
    quadratic_average(n0, n2, a, b, e) -> ⟨(n0 + n2 s²) / (a s² - b s + e)⟩_s,  s ~ N(0, 1)

Exact for complex `e` (analytic in e, including Im e < 0, with the branch cut of the root
pair along the negative imaginary axis of e - b²/4a). For |a| ≤ `A_SMALL` the denominator is
taken as linear (error O(a/|e|)); above it the Z route loses ~log10(1/a²) digits (2e-10 at a = 1e-5).

The roots r± = (b ± δ)/2a of the denominator are labelled by continuation from Im e > 0, where
r₋ lies in the upper and r₊ in the lower half plane: ⟨1/(s-r₋)⟩ = Z(r₋/√2)/√2 and
⟨1/(s-r₊)⟩ = -Z(-r₊/√2)/√2. Both are entire in r, so the result continues to Im e < 0.
"""
function quadratic_average(n0, n2, a::Real, b::Real, e)
    if abs(a) <= A_SMALL
        I0, I2 = linear_moments(e, b)
        return n0 * I0 + n2 * I2
    end
    ω′ = e - b^2 / (4a)                       # δ² = -4a ω′
    δ = a > 0 ? -2im * sqrt(a) * sqrt_cut(ω′) : 2 * sqrt(-a) * sqrt_cut(ω′)
    rp, rm = (b + δ) / (2a), (b - δ) / (2a)
    Gp = -plasma_Z(-rp / sqrt(2.0)) / sqrt(2.0)         # ⟨1/(s - r₊)⟩
    Gm = plasma_Z(rm / sqrt(2.0)) / sqrt(2.0)           # ⟨1/(s - r₋)⟩
    # (n0 + n2 s²)/D = n2/a + (p + q s)/D
    p = n0 - n2 * e / a
    q = n2 * b / a
    return n2 / a + ((p + q * rp) * Gp - (p + q * rm) * Gm) / δ
end
const A_SMALL = 3e-8

# √z with the cut on the negative imaginary axis: analytic across the real axis.
sqrt_cut(z) = (r = sqrt(-im * z); r * sqrt(complex(im)))

# ⟨1/(ω - q s)⟩ and ⟨s²/(ω - q s)⟩ for s ~ N(0, 1).
function linear_moments(ω, q)
    q = abs(q)
    ζ = ω / (sqrt(2.0) * q)
    if !(abs(ζ) < 6)                              # also q == 0: asymptotic series
        r = (q / ω)^2
        return (1 + r * (1 + r * (3 + 15r))) / ω, (1 + r * (3 + r * (15 + 105r))) / ω
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
"""
struct GyrokineticDrift <: AbstractResponse
    c::Float64
    cperp::Float64
    alpha::Float64
    nmu::Int
    mu::Vector{Float64}
    wmu::Vector{Float64}
end
function GyrokineticDrift(; c, cperp = c, alpha = 0.0, nmu = 64)
    mu, wmu = gauss_laguerre(nmu)
    return GyrokineticDrift(float(c), float(cperp), float(alpha), nmu, mu, wmu)
end
scaled(r::GyrokineticDrift, f) = GyrokineticDrift(r.c * f, r.cperp * f, r.alpha, r.nmu, r.mu, r.wmu)

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
own_units(r::GyrokineticDrift, s::Species) = scaled(r, sign(s.q) * thermal_gyroradius(s))
