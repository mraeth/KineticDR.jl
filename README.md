# KineticDR.jl

Local dispersion relations of a magnetised plasma: any number of species (ions and electrons are
the same type, each with its own charge, mass, temperature, density, gradients and kinetic response)
closed by a field equation. Electrostatic (with density/temperature gradients) and electromagnetic
(Darwin or full Maxwell, homogeneous background). Full-orbit (all cyclotron harmonics), gyrokinetic,
drift-kinetic, cold and Boltzmann responses; root finders and gradient scans on the complex-ω plane.

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

## Electromagnetic

```julia
β, μ = 0.1, 1 / 100                                        # β = 2μ₀ N T_i / B², μ = m_e/m_i
ion = Species()                                            # hot Maxwellian, GordeyevSeries()
electron = Species(q = -1, m = μ, response = DriftKinetic())
model = Model((ion, electron), Darwin(beta = β))           # or Maxwell(beta = β, lambda2 = (v_A/c)²)
k = Wavevector(kx = 0.5, kz = 0.2)                         # k = (kx, 0, kz), b̂ = ẑ
dispersion_matrix(model, 0.8 + 0.01im, k)                  # 𝒟 (3×3), 𝒟 Ê = 0
find_root(model, k, 0.8 + 0.01im)                          # roots of det(ω² 𝒟)
```

𝒟 = vacuum + Σ_s χ̃_s − (2/β)(k²/ω²)(I − k̂k̂) in units of ω_pi²/Ω_i², with (c/ω_pi)² = (2/β)ρ_i²
(Darwin: no vacuum term, the longitudinal part is quasineutrality; `Maxwell`: vacuum term
`lambda2` = (v_A/c)² = (λ_D/ρ_i)²). The species tensor is χ̃_s = (q²N/m)/ω² · M_s, with M_s from
`susceptibility(response, ω, k)` in the species' own units:

| Response | Tensor |
|---|---|
| `GordeyevSeries` | Stix hot-Maxwellian tensor, all harmonics |
| `GordeyevIntegral` | the same as an orbit integral (validation; Im ω > 0 or k∥ ≠ 0) |
| `Cold` | Stix S, D, P |
| `DriftKinetic(; polarization = false)` | m → 0: Hall (E×B) + parallel Landau −ζ²Z′(ζ); `polarization = true` adds χ_xx = χ_yy = m/m_i |

Electron sign (mirror y → −y) and kx < 0 (rotation by π) are handled by the species method.
`dispersion` returns det(ω²𝒟), kept analytic in ω (no ω-dependent scaling), so all ω-root finders
apply; it is entire for k∥ ≠ 0 with kinetic species, `Cold` species keep their poles at ω = |Ω_s|.
The tensor's longitudinal projection k·χ̃_s·k equals −`charge_response(s, ω, k)` of the electrostatic path.
det(ω²𝒟) can vanish like ω² at ω = 0 (k∥ = 0, and with `DriftKinetic` electrons also for k∥ ≠ 0):
seeds or branch tracking at low frequency can converge to this spurious root, so discard roots with ω ≈ 0.

Limits: no gradients (a species with κ ≠ 0 throws), ky must be 0, and `Boltzmann`/`Gyrokinetic`
have no tensor. Solve for ω at real k; complex k is not supported. For a kinetic species with
0 < k∥ ≪ ω (|ζ| ≫ 1), Z(ζ) ∝ e^{ζ²} below the real ω axis, so root searches there overflow: use
k∥ = 0 exactly for perpendicular propagation.

## Sign convention

PRL convention for all species: κn = +d ln N/dx, κT = +d ln T/dx, ω_*s = +(T_s/q_s) k_y κn_s, i.e. the
drive numerator ω − ω_* with ω_* = +k_y(κn + κT(v²−3)/2) for ions. Conventions with 1/L = −d ln N/dx
(e.g. Maeyama) enter as κ = −1/L.

## Layout

```
src/KineticDR.jl   module, exports
src/special.jl     plasma Z function, Γ_p = I_p e^{-x}
src/types.jl       Wavevector, abstract types
src/responses.jl   GordeyevSeries, Gyrokinetic, Boltzmann, GordeyevIntegral, Cold, DriftKinetic;
                   nonadiabatic_response kernel, susceptibility tensors
src/species.jl     Species, Model, charge_response, susceptibility, with_gradients
src/fields.jl      Quasineutrality, Poisson, Darwin, Maxwell, dispersion, dispersion_matrix
src/solve.jl       root finders and gradient scans
test/runtests.jl   tests
```

Not implemented: drift and toroidal terms, gradients in the electromagnetic tensor, k_y ≠ 0 for
electromagnetic models, k_x ≠ 0 with shear, radial eigenvalue problem.
