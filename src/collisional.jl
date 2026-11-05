# Collisional responses (electrostatic, local): BGK ions (ion–neutral collisions) and resistive isothermal electrons.
# Both override `charge_response(::Species{R}, ω, k)` (lab units) and have no Gordeyev sum of their own.

"""
    BGK(inner = GordeyevSeries(); ν)

Kinetic response `inner` with a number-conserving BGK (Krook) collision operator
C[f] = -ν (f - (n/N) F₀), F₀ the equilibrium Maxwellian at fixed temperature, `ν` in units of Ω_i (lab units).
It models charge exchange with a neutral background at the ion temperature: it conserves particles, not momentum or
energy. Along the unperturbed orbit the operator shifts ω → ω̃ = ω + iν, and the restoring term n F₀ is a source with
the same orbit integral as a potential, so

    q δn/φ = (q² N/T) (W(ω̃) - 1) / (1 + iν S(ω̃)),

with W the collisionless nonadiabatic response (all gradient drives) and S the Gordeyev sum of `inner`, both at ω̃ in
the species' units. ν = 0 is `inner`; k → 0 keeps the collisionless E×B density response (number conservation).
Electrostatic only; `inner` must carry its polarisation in W (a kinetic response, not `DriftKinetic(polarization =
true)` or `Cold`).
"""
struct BGK{R<:AbstractResponse} <: AbstractResponse
    inner::R
    ν::Float64
    function BGK(inner::R, ν::Real) where {R<:AbstractResponse}
        iszero(polarization_weight(inner, 0.0)) ||
            throw(ArgumentError("BGK: the inner response must hold its polarisation in W (got $(nameof(R)) with polarisation)"))
        inner isa Union{Cold,BGK} && throw(ArgumentError("BGK: inner response $(nameof(R)) has no Gordeyev sum"))
        return new{R}(inner, float(ν))
    end
end
BGK(inner::AbstractResponse = GordeyevSeries(); ν) = BGK(inner, ν)

gordeyev(r::BGK, ω, k::Wavevector) =
    throw(ArgumentError("BGK has no collisionless Gordeyev sum; use charge_response(::Species{<:BGK}, ω, k)"))

function charge_response(s::Species{<:BGK}, ω, k::Wavevector)
    Ω, ks = _species_units(s, k)
    r = s.response
    ω̃ = ω / Ω + im * r.ν / Ω
    S = first(gordeyev(r.inner, ω̃, ks))
    W = nonadiabatic_response(r.inner, ω̃, ks, _drive(s, k, Ω)...)
    return s.q^2 * s.N / s.T * (W - 1) / (1 + im * (r.ν / Ω) * S)
end

"""
    Resistive(; α, polarization = false)

Massless isothermal fluid species at k⊥ρ = 0 with a resistive parallel coupling (the Hasegawa–Wakatani electron
closure). Continuity with the E×B advection of the density gradient, δn_E×B (the k∥ = 0 response), and a relaxation
toward the Boltzmann response δn_B = -qNφ/T, -iω δn = -iω δn_E×B + α(δn_B - δn), gives

    q δn/φ = (ω ρ_E×B + iα ρ_B) / (ω + iα),   ρ_E×B = -q N κn k_y/ω,   ρ_B = -q² N/T,

i.e. `DriftKinetic()` at k∥ = 0 for α → 0 and `Boltzmann()` for α → ∞. `α = k∥² v_t²/ν_c` (k∥ v_t ≪ ν_c) in units of
Ω_i, a fixed number: the k∥ of the mode is not taken from the `Wavevector`. No temperature drive (isothermal).
`polarization = true` adds the polarisation charge -N m k⊥² (as `DriftKinetic(polarization = true)`), which carries
the drift-cyclotron coupling at k∥ = 0. Electrostatic only.
"""
struct Resistive <: AbstractResponse
    α::Float64
    polarization::Bool
end
Resistive(; α, polarization = false) = Resistive(float(α), polarization)
polarization_weight(r::Resistive, ω) = r.polarization ? 1.0 : 0.0

gordeyev(r::Resistive, ω, k::Wavevector) =
    throw(ArgumentError("Resistive is a fluid response; use charge_response(::Species{Resistive}, ω, k)"))

function charge_response(s::Species{Resistive}, ω, k::Wavevector)
    α = s.response.α
    return (-s.q * s.N * s.κn * k.ky - im * α * s.q^2 * s.N / s.T) / (ω + im * α) -
           polarization_weight(s, ω) * s.N * s.m * kperp(k)^2
end

"""
    ResistiveThermal(; α, polarization = false)

`Resistive` with a dynamic electron temperature (collisional Braginskii closure, Z = 1, massless electrons, k⊥ρ = 0):
density and temperature are advected by E×B on the background gradients (κn, κT), and the parallel flux and heat
flux relax them along B. With ñ = δn/N, T̃ = δT/T, φ̃ = -qφ/T and the drift frequencies ω_* = (T/q) k_y κn,
ω_*T = (T/q) k_y κT (lab units):

    -iω ñ        = -iω_* φ̃        - α (ñ + 1.71 T̃ - φ̃)
    -iω (3/2) T̃ = -iω_*T (3/2) φ̃ - 1.71 α (ñ + 1.71 T̃ - φ̃) - 3.16 α T̃

(parallel flux ∝ ∇∥(φ̃ - ñ - 1.71 T̃) with the thermal force 0.71, heat flux 0.71 T Γ∥ - 3.16 (NT/mν) ∇∥T, its
compression), and q δn/φ = -(q² N/T) ñ/φ̃. α → 0 is the E×B response (`DriftKinetic()` at k∥ = 0), α → ∞ the
isothermal Boltzmann response. α = k∥² v_t²/ν_c in units of Ω_i. Electrostatic only.
"""
struct ResistiveThermal <: AbstractResponse
    α::Float64
    polarization::Bool
end
ResistiveThermal(; α, polarization = false) = ResistiveThermal(float(α), polarization)
polarization_weight(r::ResistiveThermal, ω) = r.polarization ? 1.0 : 0.0

gordeyev(r::ResistiveThermal, ω, k::Wavevector) =
    throw(ArgumentError("ResistiveThermal is a fluid response; use charge_response(::Species{ResistiveThermal}, ω, k)"))

function charge_response(s::Species{ResistiveThermal}, ω, k::Wavevector)
    α = s.response.α
    ωn, ωT = (s.T / s.q) * k.ky * s.κn, (s.T / s.q) * k.ky * s.κT
    A = @SMatrix [(-im * ω + α) 1.71α; 1.71α (-1.5im * ω + (1.71^2 + 3.16) * α)]
    b = SVector(-im * ωn + α, -1.5im * ωT + 1.71α)
    ñ = (A \ b)[1]
    return -s.q^2 * s.N / s.T * ñ - polarization_weight(s, ω) * s.N * s.m * kperp(k)^2
end
