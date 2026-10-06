# KineticDR.jl

Local electrostatic dispersion relations of a magnetised plasma: any number of species (ions and
electrons are the same type, each with its own charge, mass, temperature, density, gradients and
kinetic response) closed by a field equation. Full-orbit (all cyclotron harmonics), gyrokinetic and
Boltzmann responses; root finders and gradient scans on the complex-ω plane.

Units: m_i = e = B = T_i = N_i = 1 (Ω_i = 1, lengths in ρ_i, potentials in T_i/e), time dependence
exp(−iωt), growth for Im ω > 0.

Studies using it (validation against Raeth & Hallatschek, PRL **133**, 195101 (2024), and Maeyama,
Phys. Plasmas **33**, 082505 (2026)) live in a separate repository, `ibw_dispersion`.

## Install and test

Registered in [BSLRegistry](https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry):

```julia
pkg> registry add https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry.git
pkg> add KineticDR
```

Development, with the repository checked out:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
```

`Pkg.develop(path = "<checkout>")` uses a local clone from another environment.

## Usage

```julia
using KineticDR
ion = Species(κn = 0.23, κT = 0.34)                         # defaults q = m = T = N = 1, GordeyevSeries()
electron = Species(q = -1, T = 1.0, response = Boltzmann()) # adiabatic electrons, T_e/T_i = T
model = Model((ion, electron), Quasineutrality())
k = Wavevector(ky = 4.0, kz = 0.025)           # kx = 0 by default
dispersion(model, 1.05 + 0.01im, k)            # D(ω); modes are its roots
find_modes(model, k; re = 0:0.25:4.5)          # root search from a grid of seeds
harmonic_root(model, k, 1)                     # most unstable root next to Larmor harmonic 1
critical_kappaT(model, k, 0.4, 1)              # neutral-stability κT of species 1 at κn = 0.4 near p = 1
```

Kinetic electrons are `Species(q = -1, m = 1/3670, response = Gyrokinetic())`; more species, other
charges, masses and temperatures work the same way.

| Component | Interface | Implemented |
|---|---|---|
| species response | `gordeyev(r, ω, k) -> (S, S_T)` in the species' own units | `GordeyevSeries` (Σ_p Γ_p Z), `Gyrokinetic` (p = 0), `Boltzmann` (h = 0), `GordeyevIntegral` (time integral, validation) |
| field equation | `residual(field, ρs::Tuple, k)` | `Quasineutrality`, `Poisson(lambda2)` |

A new response is a subtype of `AbstractResponse` plus one `gordeyev` method.
`nonadiabatic_response(species, ω, k)` adds the gradient drive and rescales to the species' units,
`charge_response` gives q_s δn_s/φ. Gradient scans (`gradient_coefficients`, `neutral_gradients`,
`critical_kappaT`, `with_gradients`) act on species 1 unless `species = i` is given.

Solvers: `muller`, `find_root` (model or any analytic function), `find_modes`, `harmonic_root`,
`track_branch`, `count_roots` (argument principle).

## Sign convention

PRL convention for all species: κn = +d ln N/dx, κT = +d ln T/dx, ω_*s = +(T_s/q_s) k_y κn_s, i.e. the
drive numerator ω − ω_* with ω_* = +k_y(κn + κT(v²−3)/2) for ions. Conventions with 1/L = −d ln N/dx
(e.g. Maeyama) enter as κ = −1/L.

## Layout

```
src/KineticDR.jl   module, exports
src/special.jl     plasma Z function, Γ_p = I_p e^{-x}
src/types.jl       Wavevector, abstract types
src/responses.jl   GordeyevSeries, Gyrokinetic, Boltzmann, GordeyevIntegral; nonadiabatic_response kernel
src/species.jl     Species, Model, charge_response, with_gradients
src/fields.jl      Quasineutrality, Poisson, dispersion
src/solve.jl       root finders and gradient scans
test/runtests.jl   tests
```

Not implemented: drift and toroidal terms, k_x ≠ 0 with shear, radial eigenvalue problem.
