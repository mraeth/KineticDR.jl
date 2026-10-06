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

## Magnetic drift and the radial eigenvalue problem

```julia
ion = Species(κn = -0.04, κT = -0.16, response = GyrokineticDrift(c = 1/50, alpha = 0.0))   # ω_D = -k_y (c v∥² + c⊥ v⊥²/2)
find_modes(Model((ion, electron), Quasineutrality()), Wavevector(ky = 0.5); re = -0.06:0.01:0.0)

g = radial_grid(-20, 20, 80; bc = :dirichlet)              # PlasmaCore Grid, one radial axis
m = sheared_slab(; ky = 0.5, Ls = 200.0, κT = -0.3)        # or RadialModel(; ky, kpar, n, Ti, Te, κn, κT, B, drift, ...)
r = eigenmode(g, m, -0.024 + 0.0034im)                     # (; ω, φ::ScalarField, λ, converged)
find_eigenmodes(g, m; re = -0.07:0.002:0.0, im = 0.0005:0.001:0.012, starts = local_roots(m, 1:3:19; re = -0.08:0.01:0, imag = (0.002, 0.01)))
count_eigenvalues(g, m, -0.04, 0.0, 0.0005, 0.006)         # argument principle on det A(ω)
```

- `GyrokineticDrift(; c, cperp = c, alpha = 0, nmu = 64)`: J₀² response with ω_D; the v∥ average is two Z
  functions (`quadratic_average`), exact for any Im ω, so root finders work below the real axis. `alpha = 1` is
  bslLD's prescribed drift. **Converge `nmu`** when ω_D ~ ω: the μ integrand has a pole at distance ~|ω/(k_y c⊥)| from
  the real axis, error 1e-3 (nmu = 24), 1.4e-5 (80), 1e-9 (320) for k_y c = 0.01.
- `RadialModel`: ion (q = m = 1) with profiles n(x), T_i(x), κn(x), κT(x), B(x), k∥(x), c(x); Boltzmann electrons T_e(x).
  κ = +d ln/dx as everywhere here (bslLD's `add_kappaT!` uses the opposite sign). `local_model(m, x)` is the
  homogeneous model at x; at uniform profiles the operator on the k_x = 0 mode is `dispersion` exactly.
- Boundaries: `:dirichlet` (φ = 0 on the walls: unknowns on the N - 1 nodes after a, sine basis; the kinetic wall of
  the bslLD runs) and `:mirror` (cell centres, cosine basis, even about both walls). Eigenvectors are `ScalarField`s
  with one value per grid node.
- `find_eigenmodes` is a seed search and can miss nearly degenerate modes; `count_eigenvalues` counts them
  (L_s = 200 slab: 5 eigenvalues in a rectangle where the search finds 3).
- Not exact: the radial operator drops cyclotron harmonics (≲ 1e-3 in γ in local tests); `operator` reuses a
  workspace, so it is not thread-safe.

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
src/drift.jl       quadratic_average (v∥ average with drift), GyrokineticDrift
src/radial.jl      RadialModel, RadialGrid, operator, eigenmode, find_eigenmodes, count_eigenvalues
src/solve.jl       root finders and gradient scans
test/runtests.jl   tests
```

Not implemented: drift in the full-orbit (`GordeyevSeries`) response, kinetic electrons in the radial problem, complex k_x.
