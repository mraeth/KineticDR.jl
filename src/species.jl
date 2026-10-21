# Species and the model. Every species, ions and electrons alike, is a `Species` with its own charge,
# mass, temperature, density, gradients and response; adiabatic electrons are `response = Boltzmann()`.
#
# Units: m_i = e = B = T_i = N_i = 1 (Ω_i = ρ_ti = 1). Species s has charge q_s (in e), mass m_s,
# temperature T_s, density N_s.
# Sign convention of the PRL (133, 195101): κn_s = d ln N_s/dx, κT_s = d ln T_s/dx and
# ω_*s = (T_s/q_s) k_y κn_s, ω_*Ts = (T_s/q_s) k_y κT_s. Maeyama (Phys. Plasmas 33, 082505) defines
# κ = -d ln N/dx = 1/L_n, so his inputs enter here as κn = -1/L_n, κT = -1/L_T.

"""
    Species(; q = 1, m = 1, T = 1, N = 1, κn = 0, κT = 0, response = GordeyevSeries())

The defaults are the reference ion. `response` is any `AbstractResponse`: `GordeyevSeries`
(all harmonics, or `pmax = 12` as in Maeyama), `Gyrokinetic` (p = 0), `Boltzmann` (adiabatic),
`GordeyevIntegral` (validation), `DriftKinetic` and `Cold` (no finite Larmor radius), `GyrokineticDrift`
(`Gyrokinetic` with the magnetic drift, electrostatic).
"""
struct Species{R<:AbstractResponse}
    q::Float64
    m::Float64
    T::Float64
    N::Float64
    κn::Float64
    κT::Float64
    response::R
end
Species(; q = 1.0, m = 1.0, T = 1.0, N = 1.0, κn = 0.0, κT = 0.0, response = GordeyevSeries()) =
    Species(float(q), float(m), float(T), float(N), float(κn), float(κT), response)

"Cyclotron frequency Ω_s = q_s/m_s (signed), thermal speed v_ts and thermal gyroradius ρ_ts = v_ts/|Ω_s|."
cyclotron_frequency(s::Species) = s.q / s.m
thermal_speed(s::Species) = sqrt(s.T / s.m)
thermal_gyroradius(s::Species) = thermal_speed(s) / abs(cyclotron_frequency(s))

"""
    nonadiabatic_response(s, ω, k) -> ∫h d³v / (q_s φ/T_s)

Nonadiabatic response of species `s` (Maeyama Eq. 8–10: Σ_ℓ W_ℓsk), from the dimensionless
Gordeyev sum in the species' own units (frequencies in |Ω_s|, lengths in ρ_ts).
"""
function nonadiabatic_response(s::Species, ω, k::Wavevector)
    Ω, ks = _species_units(s, k)
    return nonadiabatic_response(own_units(s.response, s), ω / Ω, ks, _drive(s, k, Ω)...)
end

"Hook: the response with any lab-unit parameters expressed in the species' own units (default: unchanged)."
own_units(r::AbstractResponse, ::Species) = r

# Gradient drive in the species' own units: ωn = ω_*s/|Ω_s|, ωT = ω_*Ts/|Ω_s|. A species without a temperature
# (`Cold`) has none; its density gradient enters through the charge, and κT must be 0.
_drive(s::Species, k::Wavevector, Ω) = ((s.T / s.q) * k.ky * s.κn / Ω, (s.T / s.q) * k.ky * s.κT / Ω)
_drive(s::Species{Cold}, k::Wavevector, Ω) = (_check_cold(s); (0.0, 0.0))
_check_cold(s) = s.κT == 0 || throw(ArgumentError("Cold species have no temperature: κT must be 0 (got $(s.κT))"))

# |Ω_s| and k in the species' own units: k⊥ρ_ts, k∥ v_ts/|Ω_s|. Lab-unit parameters of the response itself go
# through `own_units`.
function _species_units(s::Species, k::Wavevector)
    Ω = abs(cyclotron_frequency(s))
    ρ = thermal_gyroradius(s)
    return Ω, Wavevector(k.kx * ρ, k.ky * ρ, k.kz * thermal_speed(s) / Ω)
end

"""
    susceptibility(s, ω, k) -> χ̃_s (3×3)

Susceptibility of species `s` in units of ω_pi²/Ω_i² (current j = -iω χ̃ E), b̂ = ẑ, any k⊥ = (kx, ky):
χ̃_s = (q² N/m)/ω² · R M Rᵀ + i k gᵀ/k², with M = `susceptibility(s.response, ω/|Ω_s|, k, ωn, ωT)` in the species'
units in the frame where k⊥ lies along x (gradient drive ωn, ωT along y of the lab), and R the rotation about ẑ back
to (kx, ky). Negative charge (opposite gyration) is the mirror image y → -y in that frame.
The second term only acts on the longitudinal row: it is the charge g·E that the local current misses, the
divergence of the current across the density gradient (`charge_vector`), so that the longitudinal row is the full
charge, -i k·χ̃ E + g·E = ρ (the electrostatic `charge_response`).
"""
function susceptibility(s::Species, ω, k::Wavevector)
    Ω, ks = _species_units(s, k)
    ωn, ωT = _drive(s, k, Ω)
    fy = s.q < 0 ? -1 : 1                                         # mirror y -> -y
    kp = kperp(k)
    c, sn = kp == 0 ? (1.0, 0.0) : (k.kx / kp, k.ky / kp)
    M = species_susceptibility(own_units(s.response, s), ω / Ω, Wavevector(kperp(ks), 0.0, ks.kz), ωn, ωT, SVector(sn, fy * c))
    F = SVector(1, fy, 1)
    R = @SMatrix [c -sn 0; sn c 0; 0 0 1]
    χ = (s.q^2 * s.N / s.m) / ω^2 * (R * (F .* M .* transpose(F)) * transpose(R))
    s.κn == 0 && return χ
    kv = SVector(k.kx, k.ky, k.kz)
    return χ + im * kv * transpose(charge_vector(s, ω, k)) / (kv' * kv)
end

# The response's tensor in the frame k⊥ ∥ x̂ (own units, positive gyration). `u` is the lab ŷ in that frame (mirrored
# for negative charges), which the rotation hides: a response with a magnetic drift along ŷ (ω_D ∝ k_y, ∇B along x)
# needs it, the others ignore it.
species_susceptibility(r::AbstractResponse, ω, k::Wavevector, ωn, ωT, u) = susceptibility(r, ω, k, ωn, ωT)

"""
    charge_vector(s, ω, k) -> g

Charge g·E that the local current j misses (lab frame): ∇·j = ik·j + κn j_x for a current ∝ N(x), so
ρ = (k·j - iκn j_x)/ω. j_x is the E×B current qN E_y for every response (the local kinetic theory); the
polarisation current across the gradient is dropped (Boussinesq). A response whose j_x is something else overrides
this for `Species{R}`.
"""
charge_vector(s::Species, ω, k::Wavevector) = SVector(0, -im * s.q * s.N * s.κn / ω, 0)

"""
    charge_response(s, ω, k) -> q_s δn_s/φ

Charge response in units e N_i/T_i: (q² N/T)(W − 1) − w N m k⊥², with w the `polarization_weight` (0 for the
kinetic responses, whose W holds the polarisation; 0 or 1 for `DriftKinetic`). `Cold` is the fluid response.
A response without a Gordeyev sum overrides this method for `Species{R}` (lab units, its own gradient drive).
"""
charge_response(s::Species, ω, k::Wavevector) =
    s.q^2 * s.N / s.T * (nonadiabatic_response(s, ω, k) - 1) - polarization_weight(s, ω) * s.N * s.m * kperp(k)^2
function charge_response(s::Species{Cold}, ω, k::Wavevector)
    _check_cold(s)
    return -polarization_weight(s, ω) * s.N * s.m * kperp(k)^2 - s.q * s.N * s.κn * k.ky / ω + s.q^2 * s.N / s.m * k.kz^2 / ω^2
end

# Polarisation weight w of `s` at lab frequency ω (see `polarization_weight(r, ω)` in responses.jl).
polarization_weight(s::Species, ω) = polarization_weight(s.response, ω / abs(cyclotron_frequency(s)))

"""
    Model(species, field)

`species` is a tuple (or vector) of `Species`, `field` the field equation. `dispersion(model, ω, k)`
= `residual(field, charge responses, k)`; with `Poisson(lambda2)` and λ² in units ρ_ti² this is
the Maeyama dispersion relation (Eq. 20).
"""
struct Model{S<:Tuple,F<:AbstractFieldEquation}
    species::S
    field::F
end
Model(species::AbstractVector, field) = Model(Tuple(species), field)

"Rebuild `m` with gradients (κn, κT) for species number `species` (used by gradient scans)."
function with_gradients(m::Model; species = 1, κn, κT)
    s = m.species[species]
    return Model(Base.setindex(m.species, Species(s.q, s.m, s.T, s.N, float(κn), float(κT), s.response), species), m.field)
end
