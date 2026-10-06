"""
    KineticDR

Local electrostatic dispersion relations for magnetised plasmas: any number of species, each with
its own charge, mass, temperature, gradients and kinetic response, closed by a field equation.
Units: m_i = e = B = T_i = N_i = 1, so Ω_i = ρ_i = 1, lengths in ρ_i, frequencies in Ω_i,
potentials in T_i/e.
"""
module KineticDR

using LinearAlgebra: SymTridiagonal, eigen, diagm, det, mul!
using PlasmaCore: Grid, ScalarField
using SpecialFunctions: erfcx, besselix, besselj0
using QuadGK: quadgk

include("special.jl")
include("types.jl")
include("responses.jl")
include("species.jl")
include("drift.jl")
include("fields.jl")
include("solve.jl")
include("radial.jl")

export Wavevector, kperp
export AbstractResponse, GordeyevSeries, GordeyevIntegral, Gyrokinetic, GyrokineticDrift, Boltzmann, gordeyev
export Species, Model, nonadiabatic_response, charge_response, with_gradients
export cyclotron_frequency, thermal_speed, thermal_gyroradius
export AbstractFieldEquation, Quasineutrality, Poisson, residual, dispersion
export plasma_Z, Gamma_p, dGamma_p
export muller, find_root, find_modes, track_branch, count_roots, gradient_coefficients,
       neutral_gradients, harmonic_root, critical_kappaT

export RadialModel, RadialGrid, radial_grid, sheared_slab, log_gradient, local_model, local_roots,
       operator, eigenmode, find_eigenmodes, find_eigenmode, count_eigenvalues, radial_field

end
