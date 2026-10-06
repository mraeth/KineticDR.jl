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

function gordeyev(r::GordeyevSeries, ω, k::Wavevector)
    x = kperp(k)^2
    if r.pmax !== nothing
        return _series(ω, k, -r.pmax:r.pmax)
    end
    # Γ_p decreases monotonically in |p|; find the cut-off, then sum symmetrically.
    pm = 0
    while Gamma_p(pm + 1, x) > r.tol && pm < 5000
        pm += 1
    end
    return _series(ω, k, -pm:pm)
end

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

"No kinetic response (h = 0): the Boltzmann/adiabatic species, density response -q N φ/T only."
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
