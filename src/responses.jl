# Kinetic response of one species in its own units (frequencies in |Ω_s|, lengths in ρ_ts). Interface:
#     gordeyev(r, ω, k) -> (S, S_T)
# S(T) = Σ_p Γ_p(T k⊥²) a Z((ω-p) a), a = 1/(|kz|√(2T)), is the Gordeyev sum at T = 1 and S_T
# its T-derivative. The nonadiabatic density response to φ follows for any Maxwellian background:
#     ∫h d³v / φ = -ω S + ωT S_T + ωn S                       (PRL 133, 195101, Eq. 9)
# so a kinetic response model only has to supply (S, S_T). A fluid response (`Cold`) has no Gordeyev sum and
# defines `charge_response(::Species{R}, ω, k)` instead, at the Species level and in lab units (it adds its own
# gradient drive). A polarisation outside W, -w N m k⊥² electrostatically and ω² w in M_xx = M_yy, has one origin
# for both paths: `polarization_weight(r, ω)` in the species' units (0 for the kinetic responses).

"""
    GordeyevSeries(; pmax = nothing, tol = 1e-15)

Full-orbit response: sum over all cyclotron harmonics p. `pmax = nothing` truncates adaptively
when Γ_p < tol; an integer truncates at ±pmax (pmax = 0 is the electrostatic gyrokinetic limit; electromagnetically
the p ≠ 0 polarisation and E×B currents are missing, use `Gyrokinetic()`).
"""
Base.@kwdef struct GordeyevSeries <: AbstractResponse
    pmax::Union{Nothing,Int} = nothing
    tol::Float64 = 1e-15
end

"""
    Gyrokinetic()

Gyrokinetic response. Electrostatic: only the p = 0 term (the J₀² response of slab ITG/ETG). Electromagnetic:
the p = 0 term of the hot tensor plus the polarisation and E×B currents of p ≠ 0 to lowest order in ω/Ω.
"""
struct Gyrokinetic <: AbstractResponse end

"""
    GordeyevIntegral(; tmax_factor = 40)

Direct evaluation of G = -iω ∫₀^∞ exp(-T k⊥²(1-cos t) - T kz² t²/2 + iωt) dt (Eq. 7) by
quadrature, and of its T-derivative. Needs Im ω > 0 or kz ≠ 0 (then also Im ω < 0). For validation only.
"""
Base.@kwdef struct GordeyevIntegral <: AbstractResponse
    tmax_factor::Float64 = 40.0
end

"""
    Cold()

Cold fluid response (Stix S, D, P): M = [S' -iD' 0; iD' S' 0; 0 0 -1] with S' = -ω²/(ω²-1),
D' = ω/(ω²-1). Electrostatic (`charge_response`), with the Boussinesq polarisation charge and the E×B advection
of the density gradient, as the local kinetic theory:
q δn/φ = -w N m k⊥² - q N κn k_y/ω + (q² N/m) k_z²/ω², w = Ω²/(Ω² - ω²). No temperature: κT must be 0.
"""
struct Cold <: AbstractResponse end

"""
    DriftKinetic(; polarization = false)

Drift-kinetic species, no finite Larmor radius. Electromagnetic: the m → 0 limit of the hot tensor at
fixed ζ = ω/(√2 |k∥| v_t) (electrons): E×B Hall current, parallel Landau response -ζ² Z'(ζ), and the
n = 0 coupling to δB∥ (μ∇∥B force and perpendicular pressure): M_yy = 2b ζ Z(ζ),
M_yz = -M_zy = -i √(b/2) ζ Z'(ζ), b = (k⊥ρ_t)², which stay finite in χ as m → 0.
Electrostatic: the p = 0 term at k⊥ρ = 0 (Γ₀ = 1, any gradients).
`polarization = true` adds the polarisation drift (M_xx = M_yy += ω², i.e. χ_xx = N m/q² in units
of ω_pi²/Ω_i²; electrostatic, the charge response -N m k⊥²).
"""
Base.@kwdef struct DriftKinetic <: AbstractResponse
    polarization::Bool = false
end

# Weight w of the Boussinesq polarisation outside W, in the species' units (ω in |Ω_s|), the one origin of the
# charge -w N m k⊥² (B = 1) and of M_xx = M_yy = ω² w: 0 for the kinetic responses, 1 or 0 (`DriftKinetic`),
# 1/(1 - ω²) for a cold fluid.
polarization_weight(r::AbstractResponse, ω) = 0.0
polarization_weight(r::DriftKinetic, ω) = r.polarization ? 1.0 : 0.0
polarization_weight(r::Cold, ω) = 1 / (1 - ω^2)

function _series(ω, k::Wavevector, ps)
    x = kperp(k)^2
    kz = abs(k.kz)
    S = zero(complex(ω))
    ST = zero(complex(ω))
    for p in ps
        Γ = Gamma_p(p, x)
        dΓ = dGamma_p(p, x)
        if kz == 0
            S -= Γ / (ω - p)
            ST -= x * dΓ / (ω - p)
        else
            a = 1 / (kz * sqrt(2.0))
            ζ = (ω - p) * a
            Z = plasma_Z(ζ)
            S += a * Γ * Z
            ST += a * (x * dΓ * Z - Γ * Z / 2 + Γ * ζ * one_plus_zZ(ζ))
        end
    end
    return S, ST
end

# Harmonics -pm:pm to sum at x = k⊥². The tensor weights Γ_p/x and dΓ_p as well as Γ_p: cut where
# the next harmonic has Γ_p < tol min(x, 1) and |dΓ_p| < tol. At x = 0 only Γ_1/x → 1/2 survives.
function _harmonics(r::GordeyevSeries, x)
    r.pmax !== nothing && return -r.pmax:r.pmax
    pm = x == 0 ? 1 : 0
    while (Gamma_p(pm + 1, x) > r.tol * min(x, 1) || abs(dGamma_p(pm + 1, x)) > r.tol) && pm < 5000
        pm += 1
    end
    return -pm:pm
end

gordeyev(r::GordeyevSeries, ω, k::Wavevector) = _series(ω, k, _harmonics(r, kperp(k)^2))

gordeyev(::Gyrokinetic, ω, k::Wavevector) = _series(ω, k, 0:0)

gordeyev(::DriftKinetic, ω, k::Wavevector) = _series(ω, Wavevector(0.0, 0.0, k.kz), 0:0)   # Γ₀(0) = 1
gordeyev(::Cold, ω, k::Wavevector) = throw(ArgumentError("Cold has no Gordeyev sum (its charge response is the fluid one)"))

# Orbit-integral cut-off: the integrand is bounded by exp(-γt - kz² t²/2); stop where it is e^{-tmax_factor}.
function _tmax(r::GordeyevIntegral, γ, kz2)
    (γ > 0 || kz2 > 0) || throw(ArgumentError("GordeyevIntegral needs Im ω > 0 or kz ≠ 0"))
    kz2 == 0 && return r.tmax_factor / γ
    return (-γ + sqrt(γ^2 + 2 * kz2 * r.tmax_factor)) / kz2
end

function gordeyev(r::GordeyevIntegral, ω, k::Wavevector)
    x = kperp(k)^2
    kz2 = k.kz^2
    tmax = _tmax(r, imag(ω), kz2)
    nper = ceil(Int, tmax / (2π))
    base(t) = exp(-x * (1 - cos(t)) - kz2 * t^2 / 2 + im * ω * t)
    I0 = zero(complex(ω))
    I1 = zero(complex(ω))
    for n in 0:(nper - 1)
        a, b = 2π * n, 2π * (n + 1)
        I0 += quadgk(base, a, b; rtol = 1e-13)[1]
        I1 += quadgk(t -> (-(1 - cos(t)) * x - kz2 * t^2 / 2) * base(t), a, b; rtol = 1e-13)[1]
    end
    # dT acting on exp(-T k⊥²(1-cos t) - T kz² t²/2): factor (-k⊥²(1-cos t) - kz² t²/2); above
    # x = k⊥² so k⊥²·(1-cos t) = x (1-cos t). G = -iω I0, G_T = -iω I1, S = -G/ω.
    return im * I0, im * I1
end

"""
    Boltzmann()

No kinetic response (h = 0): the Boltzmann/adiabatic species, density response -q N φ/T. Electromagnetic: the
ζ → 0 limit of `DriftKinetic`, an isothermal massless fluid along the field line, δn = -qNψ/T with E∥ = -ik∥ψ,
carrying the E×B and the diamagnetic current (needs k∥ ≠ 0). It lacks only the transit-time damping of δB∥,
M_yy = 2b ζZ(ζ) of `DriftKinetic`, which is O(k⊥²ρ²) relative to the E×B current.
"""
struct Boltzmann <: AbstractResponse end
gordeyev(::Boltzmann, ω, k::Wavevector) = (zero(complex(ω)), zero(complex(ω)))

"""
    nonadiabatic_response(r, ω, k, ωn, ωT) -> ∫h d³v / (q φ/T)

Response for given diamagnetic drive frequencies ωn = ω_*, ωT = ω_*T (all in the species' own
units): `-ω S + ωT S_T + ωn S`.
"""
function nonadiabatic_response(r::AbstractResponse, ω, k::Wavevector, ωn::Number, ωT::Number)
    S, ST = gordeyev(r, ω, k)
    return -ω * S + ωT * ST + ωn * S
end

# ── Electromagnetic responses ─────────────────────────────────────────────────────────────────
# Interface:  susceptibility_T(r, ω, k) -> (M, T∂_T M), M the 3×3 tensor in the species' own units for positive
# gyration, b̂ = ẑ and k = (k⊥, 0, k∥), k⊥ = kperp(k) ≥ 0, and T∂_T M its temperature derivative (the drive of a
# temperature gradient). The species susceptibility is χ_s = (ω_ps²/ω²) M (see `susceptibility(::Species, ω, k)`).
# Gradient drive: with F = N(X) F_M(v; T(X)), X the guiding centre, the Vlasov response is
#   δf = (q/T) F (1 - ω_*(v)/ω) ∫ v'·E dτ + (a term odd in v: no current, the charge `charge_vector`),
# ω_*(v) = ωn + ωT (v²/2 - 3/2), and v²/2 - 3/2 is T∂_T of the Maxwellian. So
#   M(ω, k; ωn, ωT) = (1 - ωn/ω) M - (ωT/ω)(M + T∂_T M) + polarisation   (species units, ωn = ω_*/|Ω|),
# with T∂_T M = ½ d/ds M(ω, s k) at s = 1 (k⊥ρ_t and k∥v_t scale with √T). Every M passes through
# `_susceptibility(r, ω, k)` (M without polarisation; `first(susceptibility_T)` unless a response has a cheaper M-only
# path) and gets `_polarization(r, ω)` added once. A response without a temperature (`Cold`) defines only
# `_susceptibility`: its Species has no ω_* (`_drive`), its density gradient enters through `charge_vector`.
# Conventions differ from the electrostatic hook: there the ω_*T weight is S_T itself (W = -(ω - ωn)S + ωT S_T),
# here it is M + T∂_T M.

"""
    susceptibility(r, ω, k[, ωn, ωT]) -> M

Susceptibility tensor of response `r` in its own units, with the gradient drive ωn = ω_*/|Ω|, ωT = ω_*T/|Ω|:
(1 - ωn/ω) M - (ωT/ω)(M + T∂_T M) from `susceptibility_T`, plus a polarisation drift that the drive does not weight
(`DriftKinetic(polarization = true)`).
"""
susceptibility(r::AbstractResponse, ω, k::Wavevector) = _susceptibility(r, ω, k) + _polarization(r, ω)
function susceptibility(r::AbstractResponse, ω, k::Wavevector, ωn, ωT)
    ωn == 0 && ωT == 0 && return susceptibility(r, ω, k)
    M, MT = susceptibility_T(r, ω, k)
    return (1 - ωn / ω) * M - (ωT / ω) * (M + MT) + _polarization(r, ω)
end

"""
    susceptibility_T(r, ω, k) -> (M, T∂_T M)

Susceptibility tensor of response `r` in its own units (frame k⊥ ∥ x̂, positive gyration) and its temperature
derivative, which carries the drive of a temperature gradient. The electromagnetic interface of a kinetic response.
"""
susceptibility_T(r::AbstractResponse, ω, k::Wavevector) =
    throw(ArgumentError("$(nameof(typeof(r))) has no electromagnetic susceptibility with a temperature drive"))
# M without the polarisation; a response with a cheaper M-only path overrides it.
_susceptibility(r::AbstractResponse, ω, k::Wavevector) = first(susceptibility_T(r, ω, k))
# Polarisation drift, Boussinesq: M_xx = M_yy = ω² w (`polarization_weight`), not weighted by the gradient drive,
# as the electrostatic -w N m k⊥².
_polarization(r::AbstractResponse, ω) = (S = ω^2 * polarization_weight(r, ω); @SMatrix [S 0 0; 0 S 0; 0 0 0])

# Cold: T = 0, no ω_* (no `susceptibility_T`); the density gradient enters through the charge (`charge_vector`)
_susceptibility(::Cold, ω, k::Wavevector) = (D = ω / (ω^2 - 1); @SMatrix [0 -im*D 0; im*D 0 0; 0 0 -one(D)])

function susceptibility_T(::DriftKinetic, ω, k::Wavevector)
    b = kperp(k)^2
    κ = abs(k.kz)
    c = zero(complex(ω))
    if κ == 0
        A, B, P = -one(c), c, -one(c)                        # ζZ, ζZ', -ζ²Z' at |ζ| → ∞
        AT, BT, PT = c, c, c
    else
        ζ = ω / (sqrt(2.0) * κ)
        Z = plasma_Z(ζ)
        Zp = -2 * one_plus_zZ(ζ)
        Zpp = -2 * (Z + ζ * Zp)
        A, B, P = ζ * Z, ζ * Zp, -ζ^2 * Zp
        AT, BT, PT = -(A - P) / 2, -(B + ζ^2 * Zpp) / 2, -P + ζ^3 * Zpp / 2
    end
    sz = (k.kz < 0 ? 1 : -1) * im * sqrt(b / 2)
    yz, yzT = sz * B, sz * (B / 2 + BT)
    return (@SMatrix [c im*ω 0; -im*ω 2b*A yz; 0 -yz P]), (@SMatrix [c c 0; c 2b*(A+AT) yzT; 0 -yzT PT])
end

# Stix hot-Maxwellian tensor (harmonics ns) and, for Val(true), its T-derivative, b = k⊥², ζ_n = (ω - n)/(√2|k∥|).
# Elements odd in k∥ (xz, yz) carry sign(k∥); the series itself uses |k∥| so that Z is the Landau-continued
# function. T∂_T acts as b → b(1 + δ), ζ → ζ(1 - δ/2).
function _tensor_series(ω, k::Wavevector, ns, ::Val{withT}) where {withT}
    b = kperp(k)^2
    κ = abs(k.kz)
    c = zero(complex(ω))
    M = zero(SVector{6,typeof(c)})                   # xx, xy, xz, yy, yz, zz
    MT = zero(SVector{6,typeof(c)})
    sb, sbT = sqrt(b / 2), sqrt(b / 2) / 2
    for n in ns
        Γ, dΓ = Gamma_p(n, b), dGamma_p(n, b)
        Γb = b == 0 ? (abs(n) == 1 ? 0.5 : 0.0) : Γ / b          # Γ_n/b, finite at b = 0
        if κ == 0
            A, B, C = -ω / (ω - n), c, ω / (ω - n)              # ζ0 Z, ζ0 Z', ζ0 ζn Z' at |ζ| → ∞
            AT, BT, CT = c, c, c
        else
            ζ0 = ω / (sqrt(2.0) * κ)
            ζn = (ω - n) / (sqrt(2.0) * κ)
            Z = plasma_Z(ζn)
            Zp = -2 * one_plus_zZ(ζn)
            A, B, C = ζ0 * Z, ζ0 * Zp, ζ0 * ζn * Zp
            if withT
                Zpp = -2 * (Z + ζn * Zp)
                AT, BT, CT = -(A + C) / 2, -(B + ζ0 * ζn * Zpp) / 2, -C - ζ0 * ζn^2 * Zpp / 2
            end
        end
        M += SVector(n^2 * Γb * A, im * n * dΓ * A, -n * sb * Γb * B, (n^2 * Γb - 2b * dΓ) * A, im * sb * dΓ * B, -Γ * C)
        if withT
            d2Γ = d2Gamma_p(n, b)
            ΓbT = b == 0 ? 0.0 : dΓ - Γb
            MT += SVector(n^2 * (ΓbT * A + Γb * AT), im * n * (b * d2Γ * A + dΓ * AT), -n * ((sbT * Γb + sb * ΓbT) * B + sb * Γb * BT),
                          (n^2 * ΓbT - 2b * dΓ - 2b^2 * d2Γ) * A + (n^2 * Γb - 2b * dΓ) * AT,
                          im * ((sbT * dΓ + sb * b * d2Γ) * B + sb * dΓ * BT), -(b * dΓ * C + Γ * CT))
        end
    end
    return _tensor(M, k), _tensor(MT, k)
end
function _tensor(m, k::Wavevector)
    sz = k.kz < 0 ? -1 : 1
    xx, xy, xz, yy, yz, zz = m[1], m[2], sz * m[3], m[4], sz * m[5], m[6]
    return @SMatrix [xx xy xz; -xy yy yz; xz -yz zz]
end

susceptibility_T(r::GordeyevSeries, ω, k::Wavevector) = _tensor_series(ω, k, _harmonics(r, kperp(k)^2), Val(true))
_susceptibility(r::GordeyevSeries, ω, k::Wavevector) = first(_tensor_series(ω, k, _harmonics(r, kperp(k)^2), Val(false)))

# Gyrokinetic limit of the hot tensor: the p = 0 term exactly; p ≠ 0 to lowest order in ω (ω ≪ Ω, k∥v_t ≪ Ω):
# M_xx += ω² (1 - Γ₀)/b, M_yy += ω² Σ_{p≠0} (Γ_p/b - 2b Γ_p'/p²), M_xy += -iω Γ₀' (polarisation, E×B with FLR).
# The p ≠ 0 parts of xz, yz, zz are O(ω k∥v_t, ω²) and dropped.
function susceptibility_T(::Gyrokinetic, ω, k::Wavevector)
    M0, M0T = _tensor_series(ω, k, 0:0, Val(true))
    b = kperp(k)^2
    P, PT = _one_minus_Gamma0_over_b(b)
    Y, YT = P, PT
    for p in 1:last(_harmonics(GordeyevSeries(), b))
        Y -= 4b * dGamma_p(p, b) / p^2
        YT -= 4(b * dGamma_p(p, b) + b^2 * d2Gamma_p(p, b)) / p^2
    end
    Mp(P, Y, X) = @SMatrix [ω^2*P X 0; -X ω^2*Y 0; 0 0 0]
    return M0 + Mp(P, Y, -im * ω * dGamma_p(0, b)), M0T + Mp(PT, YT, -im * ω * b * d2Gamma_p(0, b))
end

# P = (1 - Γ₀(b))/b and b dP/db, by the Taylor series below b = 0.5 where 1 - Γ₀ cancels:
# Γ₀ = Σ_k (-1)^k (2k)!/(k!³ 2^k) b^k
function _one_minus_Gamma0_over_b(b)
    b >= 0.5 && return (1 - Gamma_p(0, b)) / b, -dGamma_p(0, b) - (1 - Gamma_p(0, b)) / b
    P, PT, t = 0.0, 0.0, 1.0                          # t = k-th term of (1 - Γ₀)/b
    for k in 0:60
        P += t
        PT += k * t
        abs(t) < eps() * abs(P) && break
        t *= -b * (2k + 4) * (2k + 3) / (2 * (k + 2)^3)
    end
    return P, PT
end

# Boltzmann: the ζ → 0 limit of `DriftKinetic`. T b·∇δn = qN E∥, δn = -qNψ/T with E∥ = -ik∥ψ; the E×B current
# (M_xy = iω), the diamagnetic current of δn along k⊥ × b̂ (M_yz = iω k⊥/k∥), and j∥ from continuity, k·j = ωρ
# (M_zy = -M_yz, M_zz = ω²/k∥²). Species units; Hermitian for real ω. Isothermal along the field line: T∂_T M is the
# ζ → 0 limit of the drift-kinetic one, -ω²/k∥² in zz only, so the gradient drive weights xy, yz, zy by
# 1 - (ωn + ωT)/ω and zz by 1 - ωn/ω (the magnetic flutter across the density gradient, the kinetic -(ω_*/ω)(φ - ψ)).
function susceptibility_T(::Boltzmann, ω, k::Wavevector)
    k.kz == 0 && throw(ArgumentError("Boltzmann species need k∥ ≠ 0 electromagnetically (δn = -qNψ/T, E∥ = -ik∥ψ)"))
    c = zero(complex(ω))
    Y = im * ω * kperp(k) / k.kz
    Z = ω^2 / k.kz^2
    return (@SMatrix [c im*ω c; -im*ω c Y; c -Y Z]), (@SMatrix [c c c; c c c; c c -Z])
end

# Orbit integral for a Maxwellian (validation): M = iω ∫₀^∞ e^{iωτ} ⟨v v'ᵀ e^{i k·Δx}⟩ dτ with the
# orbit traced back by τ, v' = R(τ) v, Δx = (-sin τ v_x + (1 - cos τ) v_y, ·, -τ v_z). For a unit
# Maxwellian ⟨v_i v_m e^{i a·v}⟩ = (δ_im - a_i a_m) e^{-|a|²/2}, a = (-k⊥ sin τ, k⊥(1 - cos τ), -k∥ τ).
# T∂_T M from the same integrand: a Maxwellian at temperature T gives (Rᵀ - T a (Ra)ᵀ) e^{-T|a|²/2} per 1/T.
function _orbit_integral(r::GordeyevIntegral, ω, k::Wavevector, ::Val{withT}) where {withT}
    kp = kperp(k)
    kz = k.kz
    tmax = _tmax(r, imag(ω), kz^2)
    function G(τ)
        s, c = sincos(τ)
        a = SVector(-kp * s, kp * (1 - c), -kz * τ)
        R = @SMatrix [c -s 0; s c 0; 0 0 1]
        a2 = a' * a
        e = exp(-a2 / 2 + im * ω * τ)
        K = transpose(R) - a * transpose(R * a)
        withT || return K * e
        return hcat(K * e, (-a * transpose(R * a) - K * a2 / 2) * e)
    end
    acc = zero(SMatrix{3,withT ? 6 : 3,complex(typeof(float(imag(ω))))})
    for n in 0:(ceil(Int, tmax / (2π)) - 1)
        acc += quadgk(G, 2π * n, 2π * (n + 1); rtol = 1e-13)[1]
    end
    return im * ω * acc
end
_susceptibility(r::GordeyevIntegral, ω, k::Wavevector) = _orbit_integral(r, ω, k, Val(false))
function susceptibility_T(r::GordeyevIntegral, ω, k::Wavevector)
    acc = _orbit_integral(r, ω, k, Val(true))
    return acc[:, SVector(1, 2, 3)], acc[:, SVector(4, 5, 6)]
end
