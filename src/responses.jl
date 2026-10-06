# Kinetic response of one species in its own units (frequencies in |Ω_s|, lengths in ρ_ts). Interface:
#     gordeyev(r, ω, k) -> (S, S_T)
# S(T) = Σ_p Γ_p(T k⊥²) a Z((ω-p) a), a = 1/(|kz|√(2T)), is the Gordeyev sum at T = 1 and S_T
# its T-derivative. The nonadiabatic density response to φ follows for any Maxwellian background:
#     ∫h d³v / φ = -ω S + ωT S_T + ωn S                       (PRL 133, 195101, Eq. 9)
# so a response model only has to supply (S, S_T).

"""
    GordeyevSeries(; pmax = nothing, tol = 1e-15)

Full-orbit response: sum over all cyclotron harmonics p. `pmax = nothing` truncates adaptively
when Γ_p < tol; an integer truncates at ±pmax (pmax = 0 is the gyrokinetic limit).
"""
Base.@kwdef struct GordeyevSeries <: AbstractResponse
    pmax::Union{Nothing,Int} = nothing
    tol::Float64 = 1e-15
end

"Gyrokinetic response: only the p = 0 term (the J₀² response of slab ITG/ETG)."
struct Gyrokinetic <: AbstractResponse end

"""
    GordeyevIntegral(; tmax_factor = 40)

Direct evaluation of G = -iω ∫₀^∞ exp(-T k⊥²(1-cos t) - T kz² t²/2 + iωt) dt (Eq. 7) by
quadrature, and of its T-derivative. Needs Im ω > 0 (or kz ≠ 0). For validation only.
"""
Base.@kwdef struct GordeyevIntegral <: AbstractResponse
    tmax_factor::Float64 = 40.0
end

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

# Harmonics -pm:pm to sum at x = k⊥²: Γ_p decreases monotonically in |p|, cut where Γ_p < tol.
function _harmonics(r::GordeyevSeries, x)
    r.pmax !== nothing && return -r.pmax:r.pmax
    pm = 0
    while Gamma_p(pm + 1, x) > r.tol && pm < 5000
        pm += 1
    end
    return -pm:pm
end

gordeyev(r::GordeyevSeries, ω, k::Wavevector) = _series(ω, k, _harmonics(r, kperp(k)^2))

gordeyev(::Gyrokinetic, ω, k::Wavevector) = _series(ω, k, 0:0)

function gordeyev(r::GordeyevIntegral, ω, k::Wavevector)
    x = kperp(k)^2
    kz2 = k.kz^2
    γ = imag(ω)
    (γ > 0 || kz2 > 0) || error("GordeyevIntegral needs Im ω > 0 or kz ≠ 0")
    # integrand decays as exp(-γ t) and exp(-kz² t²/2); integrate over whole periods
    tmax = min(γ > 0 ? r.tmax_factor / γ : Inf, kz2 > 0 ? sqrt(2 * r.tmax_factor / kz2) : Inf)
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

"No kinetic response (h = 0): the Boltzmann/adiabatic species, density response -q N φ/T only. Electrostatic only."
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
# (see `susceptibility(::Species, ω, k)`). No gradients: homogeneous Maxwellian background.

"""
    Cold()

Cold fluid response (Stix S, D, P): M = [S' -iD' 0; iD' S' 0; 0 0 -1] with S' = -ω²/(ω²-1),
D' = ω/(ω²-1). Electromagnetic only.
"""
struct Cold <: AbstractResponse end

"""
    DriftKinetic(; polarization = false)

m → 0 (ω ≪ |Ω_s|) limit for electrons: E×B Hall current, parallel Landau response
-ζ² Z'(ζ), ζ = ω/(√2 |k∥| v_t), and with `polarization = true` the polarisation drift
(M_xx = M_yy = ω², i.e. χ_xx = m_e/m_i for electrons). Electromagnetic only.
"""
Base.@kwdef struct DriftKinetic <: AbstractResponse
    polarization::Bool = false
end

susceptibility(r::AbstractResponse, ω, k::Wavevector) =
    throw(ArgumentError("$(nameof(typeof(r))) has no electromagnetic susceptibility"))

function susceptibility(::Cold, ω, k::Wavevector)
    d = ω^2 - 1
    S, D = -ω^2 / d, ω / d
    return @SMatrix [S -im*D 0; im*D S 0; 0 0 -one(S)]
end

function susceptibility(r::DriftKinetic, ω, k::Wavevector)
    S = r.polarization ? ω^2 : zero(ω^2)
    κ = abs(k.kz)
    if κ == 0
        P = -one(complex(ω))
    else
        ζ = ω / (sqrt(2.0) * κ)
        P = 2ζ^2 * one_plus_zZ(ζ)                       # -ζ² Z'(ζ)
    end
    return @SMatrix [S im*ω 0; -im*ω S 0; 0 0 P]
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

# Orbit integral for a Maxwellian (validation): M = iω ∫₀^∞ e^{iωτ} ⟨v v'ᵀ e^{i k·Δx}⟩ dτ with the
# orbit traced back by τ, v' = R(τ) v, Δx = (-sin τ v_x + (1 - cos τ) v_y, ·, -τ v_z). For a unit
# Maxwellian ⟨v_i v_m e^{i a·v}⟩ = (δ_im - a_i a_m) e^{-|a|²/2}, a = (-k⊥ sin τ, k⊥(1 - cos τ), -k∥ τ).
function susceptibility(r::GordeyevIntegral, ω, k::Wavevector)
    kp = kperp(k)
    kz = k.kz
    γ = imag(ω)
    (γ > 0 || kz != 0) || error("GordeyevIntegral needs Im ω > 0 or kz ≠ 0")
    tmax = min(γ > 0 ? r.tmax_factor / γ : Inf, kz != 0 ? sqrt(2 * r.tmax_factor) / abs(kz) : Inf)
    function G(τ)
        s, c = sincos(τ)
        a = SVector(-kp * s, kp * (1 - c), -kz * τ)
        R = @SMatrix [c -s 0; s c 0; 0 0 1]
        return (transpose(R) - a * transpose(R * a)) * exp(-(a' * a) / 2 + im * ω * τ)
    end
    I = zero(SMatrix{3,3,complex(typeof(float(γ)))})
    for n in 0:(ceil(Int, tmax / (2π)) - 1)
        I += quadgk(G, 2π * n, 2π * (n + 1); rtol = 1e-13)[1]
    end
    return im * ω * I
end
