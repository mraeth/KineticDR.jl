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
`GordeyevIntegral` (validation).
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
    Ω = abs(cyclotron_frequency(s))
    ρ = thermal_gyroradius(s)
    vt = thermal_speed(s)
    ks = Wavevector(k.kx * ρ, k.ky * ρ, k.kz * vt / Ω)          # k⊥ρ_ts, k∥ v_ts/|Ω_s|
    ωn = (s.T / s.q) * k.ky * s.κn / Ω                           # ω_*s / |Ω_s|
    ωT = (s.T / s.q) * k.ky * s.κT / Ω                           # ω_*Ts / |Ω_s|
    return nonadiabatic_response(own_units(s.response, s), ω / Ω, ks, ωn, ωT)
end

"Hook: the response with any lab-unit parameters expressed in the species' own units (default: unchanged)."
own_units(r::AbstractResponse, ::Species) = r

"Charge response q_s δn_s/φ in units e N_i/T_i: (q² N/T)(W − 1)."
charge_response(s::Species, ω, k::Wavevector) =
    s.q^2 * s.N / s.T * (nonadiabatic_response(s, ω, k) - 1)

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
