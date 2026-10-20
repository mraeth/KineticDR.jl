"""
    KineticDR

Dispersion relations for magnetised plasmas: any number of species, each with its own charge, mass,
temperature, gradients and kinetic response, closed by an electrostatic or electromagnetic field
equation; local (`Model`) and radially resolved (the same `Model` with profile species x -> `Species`,
electrostatic, shearless slab).
Units: m_i = e = B = T_i = N_i = 1, so Ω_i = ρ_i = 1, lengths in ρ_i, frequencies in Ω_i,
potentials in T_i/e.
"""
module KineticDR

using LinearAlgebra: Diagonal, eigen, det, opnorm, I
using PlasmaCore: Grid, ScalarField
using SpecialFunctions: erfcx, besselix
using QuadGK: quadgk
using StaticArrays: SMatrix, SVector, @SMatrix

include("special.jl")
include("types.jl")
include("responses.jl")
include("species.jl")
include("fields.jl")
include("solve.jl")
include("radial.jl")

export Wavevector, kperp
export AbstractResponse, GordeyevSeries, GordeyevIntegral, Gyrokinetic, Boltzmann, Cold, DriftKinetic
export gordeyev, susceptibility
export Species, Model, nonadiabatic_response, charge_response, with_gradients
export cyclotron_frequency, thermal_speed, thermal_gyroradius
export AbstractFieldEquation, Quasineutrality, Poisson, residual, dispersion
export AbstractEMField, Darwin, Maxwell, dispersion_matrix
export plasma_Z, Gamma_p, dGamma_p
export muller, find_root, find_modes, track_branch, count_roots, gradient_coefficients,
       neutral_gradients, harmonic_root, critical_kappaT

export RadialGrid, radial_grid, log_gradient, species_profile, particle_density, local_roots,
       operator, eigenmode, find_eigenmodes, find_eigenmode, count_eigenvalues, radial_field

end
