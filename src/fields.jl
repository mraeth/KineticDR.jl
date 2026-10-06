# Field equations. Interface: residual(field, ρs, k), where ρs is the tuple of charge-density
# responses q_s δn_s / φ of all species (`charge_response`).

"Quasineutrality, Σ_s q_s δn_s = 0. Residual -Σ ρs: equals 2 - ∫h/φ for ions + Boltzmann electrons at T_e = T_i."
struct Quasineutrality <: AbstractFieldEquation end
residual(::Quasineutrality, ρs::Tuple, k::Wavevector) = -sum(ρs)

"Poisson with (λ_D/ρ_i)² = `lambda2`: λ² k² φ = Σ_s q_s δn_s (sign as in Quasineutrality)."
struct Poisson{T<:Real} <: AbstractFieldEquation
    lambda2::T
end
Poisson(; lambda2) = Poisson(float(lambda2))
residual(f::Poisson, ρs::Tuple, k::Wavevector) = f.lambda2 * (k.kx^2 + k.ky^2 + k.kz^2) - sum(ρs)

"""
    dispersion(model, ω, k)

Dispersion function D(ω; k); modes are its roots.
"""
dispersion(m::Model, ω, k::Wavevector) = residual(m.field, map(s -> charge_response(s, ω, k), m.species), k)
