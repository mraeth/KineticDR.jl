# Gyrokinetic response with a magnetic drift. The v∥ average of the resonant denominator is
# exact (two Z functions) for any Im ω; v⊥ enters through the magnetic moment μ = v⊥²/(2T/m).
# The μ quadrature is not: below the real axis it replaces the drift-resonance cut of the μ integral
# by `nmu` cuts (see [`GyrokineticDrift`](@ref)).

"""
    polynomial_average(n, a, b, e) -> ⟨(n₀ + n₁ s + … + n₆ s⁶) / (a s² - b s + e)⟩_s,  s ~ N(0, 1)

`n` holds the numerator coefficients (up to degree 6; shorter is padded with zeros). Exact for complex `e` (analytic
in e, including Im e < 0, with the branch cut of the root pair along the negative imaginary axis of e - b²/4a).
See `average_kernel`.
"""
polynomial_average(n, a::Real, b::Real, e) = (K = average_kernel(a, b, e); sum(n[j] * K[j] for j in eachindex(n)))

"""
    average_kernel(a, b, e) -> K,  ⟨s^j / (a s² - b s + e)⟩ = K[j + 1], j = 0 … 6

The roots r± = (b ± δ)/2a of the denominator are labelled by continuation from Im e > 0, where
r₋ lies in the upper and r₊ in the lower half plane: ⟨1/(s-r₋)⟩ = Z(r₋/√2)/√2 and
⟨1/(s-r₊)⟩ = -Z(-r₊/√2)/√2. Both are entire in r, so the result continues to Im e < 0.
Partial fractions give [⟨s^j/(s - r₊)⟩ - ⟨s^j/(s - r₋)⟩]/δ, i.e. `power_moments` at -r₊ (s → -s) and r₋,
without cancellation as a → 0 (r₊ → ∞, the linear denominator).
"""
function average_kernel(a::Real, b::Real, e)
    if a == 0                                 # linear denominator; b < 0 by s → -s
        I = power_moments(e, b)
        return b < 0 ? ntuple(j -> (-1)^(j - 1) * I[j], 7) : I
    end
    ω′ = e - b^2 / (4a)                       # δ² = -4a ω′
    δ = a > 0 ? -2im * sqrt(a) * sqrt_cut(ω′) : 2 * sqrt(-a) * sqrt_cut(ω′)
    # r₊ r₋ = e/a: the smaller root from the product, not the difference
    if abs(b + δ) >= abs(b - δ)
        rp, rm = (b + δ) / (2a), 2e / (b + δ)
    else
        rp, rm = 2e / (b - δ), (b - δ) / (2a)
    end
    P = power_moments(-rp, 1.0)               # ⟨s^j/(s - r₊)⟩ = (-1)^j P_j
    M = power_moments(rm, 1.0)                # -⟨s^j/(s - r₋)⟩ = M_j
    return ntuple(j -> ((-1)^(j - 1) * P[j] + M[j]) / δ, 7)
end

"`polynomial_average` with an even numerator n0 + n2 s² (the electrostatic response)."
quadratic_average(n0, n2, a::Real, b::Real, e) = polynomial_average((n0, 0, n2), a, b, e)

# √z with the cut on the negative imaginary axis: analytic across the real axis.
sqrt_cut(z) = (r = sqrt(-im * z); r * sqrt(complex(im)))

# ⟨s^j/(ω - q s)⟩, j = 0 … 6, for s ~ N(0, 1), continued to Im ω < 0 like Z. Exact relation:
# I_j = (ω I_{j-1} - ⟨s^{j-1}⟩)/q. For |ζ| ≥ 6 the even ones come from their asymptotic series
# Σ_n ⟨s^{j+2n}⟩ (q/ω)^{2n}/ω and I_j = q I_{j+1}/ω for odd j, which avoids the cancellation in ω I - ⟨s^{j-1}⟩.
function power_moments(ω, q)
    q = abs(q)
    ζ = ω / (sqrt(2.0) * q)
    if !(abs(ζ) < 6)                              # also q == 0: series summed to its smallest term (≲ e^{-36} at |ζ| = 6)
        r = (q / ω)^2
        t = (one(r), one(r), 3one(r), 15one(r))   # ⟨s^{2m}⟩ = (2m - 1)!!, m = 0 … 3
        S = t
        for n in 1:40
            t = ntuple(m -> t[m] * (2n + 2m - 3) * r, 4)
            S = S .+ t
            abs(t[4]) < 1e-17 * abs(S[4]) && break
        end
        E = S ./ ω                                # I0, I2, I4, I6
        if q > 0 && imag(ζ) < 0                   # Landau term 2i√π e^{-ζ²} of Z below the real axis: residue (ω/q)^j L
            L = -im * sqrt(2π) / q * exp(-ζ^2)
            E = E .+ ntuple(m -> (ω / q)^(2m - 2) * L, 4)
        end
        return (E[1], q * E[2] / ω, E[2], q * E[3] / ω, E[3], q * E[4] / ω, E[4])
    end
    gauss = (1, 0, 1, 0, 3, 0)                    # ⟨s^j⟩, j = 0 … 5
    I = -plasma_Z(ζ) / (sqrt(2.0) * q)
    out = (I,)
    for j in 1:6
        I = (ω * I - gauss[j]) / q
        out = (out..., I)
    end
    return out
end

# ⟨1/(ω - q s)⟩ and ⟨s²/(ω - q s)⟩
linear_moments(ω, q) = ((m = power_moments(ω, q)); (m[1], m[3]))

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
`c = cperp = 0` is `Gyrokinetic` up to the μ quadrature, electrostatic and electromagnetic; the quadrature must resolve
the oscillation of J₀²(k⊥ρ√(2μ)): `nmu` ≳ (k⊥ρ_s)². At k⊥ρ_s = 10, 64 nodes are 15% off `Gyrokinetic`, the default
128 agree to 1e-14.

Signs: the ∇B drift of a slab with B(x) is ω = k_y (T/q) μ ∂x ln B, so c⊥ = -∂x ln B; an equilibrium in
pressure balance has c⊥ = `pressure_balance_cperp(species, β)` and c = 0 (no curvature). Bad curvature of a torus
(outboard midplane, x outward) is c = c⊥ = 1/R > 0. The diamagnetic drive keeps the gradient of F at fixed
energy, as in flux-tube gyrokinetics: ∇B enters only through ω_D.

Electromagnetic: the p = 0 term of the hot tensor with the drift in its denominator, per μ node
with the gyrocentre drift current, plus the lowest-order polarisation and E×B of `Gyrokinetic`, which the drift does
not change at this order. The gradient drive is in the integrand (ω - ω_*(v)). `alpha ≠ 0` is electrostatic only.

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

# Electromagnetic: frame k⊥ ∥ x̂ (own units, positive gyration), `u` the lab ŷ (the direction of ∇B × b̂ and of the drift)
# in that frame. Gyrokinetic currents with the drift: h = (ω - ω_*)⟨χ⟩/D, D = ω - k∥s - ω_D, carries the gyrocentre
# current V h, V = V₀ + J₀ v_D, V₀ = (0, i√(2μ) J₁, J₀ s), v_D = -(c s² + c⊥ μ) u, J at √(2bμ); the non-resonant part
# (pull-back and magnetisation) keeps V₀ and does not drift. With -iω⟨χ⟩ = V₀†E - iJ₀ D₀ φ, D₀ = D + ω_D, and the
# Coulomb-gauge φ = i k·E/k²:
#   M = -∫ (ω - ω_*) [V V₀†/D + J₀² v_D kᵀ/k² + J₀ ω_D V kᵀ/(D k²)]
# (ω_*(v) the drive at fixed energy). For v_D = 0 it is Stix's p = 0 term; k·V = J₀(k∥s + ω_D) closes continuity, so
# the longitudinal row is the electrostatic `GyrokineticDrift` charge. The parallel row is the gyrokinetic j∥, the ŷ row
# the δB∥ equation up to O(ω/Ω, ρ/L). Plus the lowest-order polarisation and E×B of `Gyrokinetic` (`Mp`).
function species_susceptibility(r::GyrokineticDrift, ω, k::Wavevector, ωn, ωT, u)
    r.alpha == 0 || throw(ArgumentError("GyrokineticDrift: alpha ≠ 0 (prescribed drift) is electrostatic only"))
    b = kperp(k)^2
    ky = sqrt(b) * u[1]                                   # lab k_y in own units
    kv = SVector(sqrt(b), 0.0, k.kz)
    kk = kv / (b + k.kz^2)                                # k/k²
    a = ky * r.c
    T = typeof(complex(ω))
    M = zero(SMatrix{3,3,T})
    gauss = (1, 0, 1, 0, 3, 0, 15)                        # ⟨s^j⟩
    for (μ, w) in zip(r.mu, r.wmu)
        K = average_kernel(a, k.kz, ω + ky * r.cperp * μ) # ⟨s^j/D⟩
        avgD(p) = sum(p[j] * K[j] for j in eachindex(p))
        avg(p) = sum(p[j] * gauss[j] for j in eachindex(p))
        f = (ω - ωn - ωT * (μ - 1.5), 0.0, -ωT / 2)       # ω - ω_*(v)
        z = sqrt(2b * μ)
        J0, J1 = besselj0(z), besselj1(z)
        vD = (-r.cperp * μ, 0.0, -r.c)                    # v_D/u, polynomial in s; ω_D = k_y v_D/u
        V0 = ((0.0,), (im * sqrt(2μ) * J1,), (0.0, J0))
        V = (J0 .* u[1] .* vD, (im * sqrt(2μ) * J1, 0, 0) .+ J0 .* u[2] .* vD, (0.0, J0))
        fV = map(Vi -> _polymul(f, Vi), V)
        t1 = SMatrix{3,3}(avgD(_polymul(fV[i], conj.(V0[j]))) for i in 1:3, j in 1:3)
        dr = SVector(avg(_polymul(f, J0^2 .* u[1] .* vD)), avg(_polymul(f, J0^2 .* u[2] .* vD)), zero(T))
        cr = SVector(ntuple(i -> avgD(_polymul(fV[i], J0 * ky .* vD)), 3))
        M -= w * (t1 + (dr + cr) * transpose(kk))
    end
    Mp, MpT = _gyrokinetic_polarization_T(ω, k)
    return M + (1 - ωn / ω) * Mp - (ωT / ω) * (Mp + MpT)
end
susceptibility(r::GyrokineticDrift, ω, k::Wavevector, ωn, ωT) =
    throw(ArgumentError("GyrokineticDrift needs the lab k_y: use susceptibility(species, ω, k)"))
susceptibility(r::GyrokineticDrift, ω, k::Wavevector) = susceptibility(r, ω, k, 0, 0)

# product of polynomials in s given as coefficient tuples
_polymul(p, q) = ntuple(n -> sum(p[i] * q[n - i + 1] for i in max(1, n - length(q) + 1):min(n, length(p))), length(p) + length(q) - 1)

"""
    pressure_balance_cperp(species, beta) -> c⊥

∇B coefficient of `GyrokineticDrift` for a slab in pressure balance, B²/2μ₀ + Σ_s p_s = const:
∂x ln B = -(β/2) Σ_s N_s T_s (κn_s + κT_s) (units N_i T_i, β = 2μ₀ N_i T_i/B²), so c⊥ = -∂x ln B =
(β/2) Σ_s N_s T_s (κn_s + κT_s). Use it with `c = 0` (no curvature) and the same β as the field.
"""
pressure_balance_cperp(species, beta) = beta / 2 * sum(s -> s.N * s.T * (s.κn + s.κT), species)
