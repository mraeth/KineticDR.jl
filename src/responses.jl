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
# Interface:  susceptibility(r, ω, k) -> M (3×3), in the species' own units for positive gyration,
# b̂ = ẑ and k = (k⊥, 0, k∥), k⊥ = kperp(k) ≥ 0. The species susceptibility is χ_s = (ω_ps²/ω²) M
# (see `susceptibility(::Species, ω, k)`). No gradients: homogeneous background.

susceptibility(r::AbstractResponse, ω, k::Wavevector) =
    throw(ArgumentError("$(nameof(typeof(r))) has no electromagnetic susceptibility"))
function susceptibility(r::Cold, ω, k::Wavevector)
    S, D = ω^2 * polarization_weight(r, ω), ω / (ω^2 - 1)
    return @SMatrix [S -im*D 0; im*D S 0; 0 0 -one(S)]
end

function susceptibility(r::DriftKinetic, ω, k::Wavevector)
    S = ω^2 * polarization_weight(r, ω)
    b = kperp(k)^2
    κ = abs(k.kz)
    if κ == 0
        A, B, P = -one(complex(ω)), zero(complex(ω)), -one(complex(ω))   # ζZ, ζZ', -ζ²Z' at |ζ| → ∞
    else
        ζ = ω / (sqrt(2.0) * κ)
        Zp = -2 * one_plus_zZ(ζ)
        A, B, P = ζ * plasma_Z(ζ), ζ * Zp, -ζ^2 * Zp
    end
    yz = (k.kz < 0 ? 1 : -1) * im * sqrt(b / 2) * B
    return @SMatrix [S im*ω 0; -im*ω S+2b*A yz; 0 -yz P]
end

# Stix hot-Maxwellian tensor (harmonics ns), b = k⊥², ζ_n = (ω - n)/(√2|k∥|). Elements odd in k∥
# (xz, yz) carry sign(k∥); the series itself uses |k∥| so that Z is the Landau-continued function.
function _tensor_series(ω, k::Wavevector, ns)
    b = kperp(k)^2
    κ = abs(k.kz)
    c = zero(complex(ω))
    xx, xy, xz, yy, yz, zz = c, c, c, c, c, c
    for n in ns
        Γ = Gamma_p(n, b)
        dΓ = dGamma_p(n, b)
        Γb = b == 0 ? (abs(n) == 1 ? 0.5 : 0.0) : Γ / b          # Γ_n/b, finite at b = 0
        if κ == 0
            A, B, C = -ω / (ω - n), c, ω / (ω - n)              # ζ0 Z, ζ0 Z', ζ0 ζn Z' at |ζ| → ∞
        else
            ζ0 = ω / (sqrt(2.0) * κ)
            ζn = (ω - n) / (sqrt(2.0) * κ)
            Zp = -2 * one_plus_zZ(ζn)
            A, B, C = ζ0 * plasma_Z(ζn), ζ0 * Zp, ζ0 * ζn * Zp
        end
        xx += n^2 * Γb * A
        xy += im * n * dΓ * A
        xz -= n * sqrt(b / 2) * Γb * B
        yy += (n^2 * Γb - 2b * dΓ) * A
        yz += im * sqrt(b / 2) * dΓ * B
        zz -= Γ * C
    end
    sz = k.kz < 0 ? -1 : 1
    xz, yz = sz * xz, sz * yz
    return @SMatrix [xx xy xz; -xy yy yz; xz -yz zz]
end

susceptibility(r::GordeyevSeries, ω, k::Wavevector) = _tensor_series(ω, k, _harmonics(r, kperp(k)^2))

# Gyrokinetic limit of the hot tensor: the p = 0 term exactly; p ≠ 0 to lowest order in ω (ω ≪ Ω, k∥v_t ≪ Ω):
# M_xx += ω² (1 - Γ₀)/b, M_yy += ω² Σ_{p≠0} (Γ_p/b - 2b Γ_p'/p²), M_xy += -iω Γ₀' (polarisation, E×B with FLR).
# The p ≠ 0 parts of xz, yz, zz are O(ω k∥v_t, ω²) and dropped.
function susceptibility(::Gyrokinetic, ω, k::Wavevector)
    b = kperp(k)^2
    P = _one_minus_Gamma0_over_b(b)
    Y = P
    for p in 1:last(_harmonics(GordeyevSeries(), b))
        Y -= 4b * dGamma_p(p, b) / p^2
    end
    X = -im * ω * dGamma_p(0, b)
    return _tensor_series(ω, k, 0:0) + @SMatrix [ω^2*P X 0; -X ω^2*Y 0; 0 0 0]
end

# (1 - Γ₀(b))/b, by its Taylor series below b = 0.5 where 1 - Γ₀ cancels: Γ₀ = Σ_k (-1)^k (2k)!/(k!³ 2^k) b^k
function _one_minus_Gamma0_over_b(b)
    b >= 0.5 && return (1 - Gamma_p(0, b)) / b
    P, t = 0.0, 1.0                                   # t = k-th term of (1 - Γ₀)/b
    for k in 0:60
        P += t
        abs(t) < eps() * abs(P) && break
        t *= -b * (2k + 4) * (2k + 3) / (2 * (k + 2)^3)
    end
    return P
end

# Boltzmann: the ζ → 0 limit of `DriftKinetic`. T b·∇δn = qN E∥, δn = -qNψ/T with E∥ = -ik∥ψ; the E×B current
# (M_xy = iω), the diamagnetic current of δn along k⊥ × b̂ (M_yz = iω k⊥/k∥), and j∥ from continuity, k·j = ωρ
# (M_zy = -M_yz, M_zz = ω²/k∥²). Species units; Hermitian for real ω.
function susceptibility(::Boltzmann, ω, k::Wavevector)
    k.kz == 0 && throw(ArgumentError("Boltzmann species need k∥ ≠ 0 electromagnetically (δn = -qNψ/T, E∥ = -ik∥ψ)"))
    c = zero(complex(ω))
    Y = im * ω * kperp(k) / k.kz
    return @SMatrix [c im*ω c; -im*ω c Y; c -Y ω^2/k.kz^2]
end

# Orbit integral for a Maxwellian (validation): M = iω ∫₀^∞ e^{iωτ} ⟨v v'ᵀ e^{i k·Δx}⟩ dτ with the
# orbit traced back by τ, v' = R(τ) v, Δx = (-sin τ v_x + (1 - cos τ) v_y, ·, -τ v_z). For a unit
# Maxwellian ⟨v_i v_m e^{i a·v}⟩ = (δ_im - a_i a_m) e^{-|a|²/2}, a = (-k⊥ sin τ, k⊥(1 - cos τ), -k∥ τ).
function susceptibility(r::GordeyevIntegral, ω, k::Wavevector)
    kp = kperp(k)
    kz = k.kz
    tmax = _tmax(r, imag(ω), kz^2)
    function G(τ)
        s, c = sincos(τ)
        a = SVector(-kp * s, kp * (1 - c), -kz * τ)
        R = @SMatrix [c -s 0; s c 0; 0 0 1]
        return (transpose(R) - a * transpose(R * a)) * exp(-(a' * a) / 2 + im * ω * τ)
    end
    acc = zero(SMatrix{3,3,complex(typeof(float(imag(ω))))})
    for n in 0:(ceil(Int, tmax / (2π)) - 1)
        acc += quadgk(G, 2π * n, 2π * (n + 1); rtol = 1e-13)[1]
    end
    return im * ω * acc
end
