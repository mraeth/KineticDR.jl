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

# ── Electromagnetic field equations ──────────────────────────────────────────────────────────
# Dispersion matrix in units of ω_pi²/Ω_i²: 𝒟 = vacuum + Σ_s χ̃_s - (2/β)(k²/ω²)(I - k̂k̂), with
# (c/ω_pi)² = (2/β) ρ_i² and β = 2μ₀ N_i T_i/B² (= 8π N T/B² in Gaussian units).

abstract type AbstractEMField <: AbstractFieldEquation end

"""
    Darwin(; beta)

Darwin (no displacement current, no vacuum term) electromagnetic closure, as in hybrid
kinetic-ion/fluid-electron models. The longitudinal part of 𝒟 is quasineutrality.
"""
struct Darwin{T<:Real} <: AbstractEMField
    beta::T
end
Darwin(; beta) = Darwin(float(beta))

"""
    Maxwell(; beta, lambda2)

Full Maxwell: adds the vacuum term `lambda2` = (λ_D/ρ_i)² = (v_A/c)² = (Ω_i/ω_pi)² to the diagonal.
"""
struct Maxwell{T<:Real} <: AbstractEMField
    beta::T
    lambda2::T
end
Maxwell(; beta, lambda2) = Maxwell(promote(float(beta), float(lambda2))...)

_vacuum(::Darwin) = 0.0
_vacuum(f::Maxwell) = f.lambda2

"""
    dispersion_matrix(model, ω, k) -> 𝒟 (3×3)

Dispersion matrix of an electromagnetic model (`Darwin` or `Maxwell` field), 𝒟 Ê = 0;
its null vector is the polarisation.
"""
function dispersion_matrix(m::Model{<:Tuple,<:AbstractEMField}, ω, k::Wavevector)
    kv = SVector(k.kx, k.ky, k.kz)
    k2 = kv' * kv
    PT = one(SMatrix{3,3,Float64}) - kv * kv' / k2
    χ = sum(s -> susceptibility(s, ω, k), m.species)
    return χ - (2 / m.field.beta) * (k2 / ω^2) * PT + _vacuum(m.field) * I
end

"""
    dispersion(model::Model{<:Any,<:AbstractEMField}, ω, k)

det(ω² 𝒟): analytic in ω (no ω-dependent scaling), entire for kz ≠ 0 with kinetic species;
`Cold` species keep their poles at ω = |Ω_s|. It can vanish like ω² at ω = 0 (k∥ = 0, or
drift-kinetic electrons): discard roots at ω ≈ 0.
"""
dispersion(m::Model{<:Tuple,<:AbstractEMField}, ω, k::Wavevector) = det(ω^2 * dispersion_matrix(m, ω, k))
