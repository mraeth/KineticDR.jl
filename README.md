# KineticDR.jl

Kinetic dispersion relations of a magnetised plasma. You describe the plasma as a set of species
(ions and electrons are the same type) and a field equation, and solve for the complex frequency ω:

- **local electrostatic**: full-orbit (all cyclotron harmonics), gyrokinetic, drift-kinetic, cold or Boltzmann
  species with density and temperature gradients, closed by quasineutrality or Poisson;
- **local electromagnetic**: the same species with gradients, closed by Darwin or full Maxwell;
- **radial**: gyrokinetic, full-orbit, drift-kinetic, cold or Boltzmann species with radial profiles n(x), T(x); the
  eigenvalue problem for φ(x).

Validation studies (Raeth & Hallatschek, PRL **133**, 195101 (2024); Maeyama, Phys. Plasmas **33**,
082505 (2026); bsl6d) live in the separate repository `ibw_dispersion`.

## Install and test

The package is registered in [BSLRegistry](https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry):

```julia
pkg> registry add https://gitlab.mpcdf.mpg.de/bsl6d/BSLRegistry.git
pkg> add KineticDR
```

From a checkout: `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'`.

## Units and conventions

- m_i = e = B = T_i = N_i = 1: frequencies in Ω_i, lengths in ρ_i = v_ti/Ω_i, φ in T_i/e.
- Fields vary as exp(i k·x − iωt): **Im ω > 0 is growth**.
- b̂ = ẑ, gradients along x, so the diamagnetic drive is ∝ k_y.
- κn = +d ln N/dx and κT = +d ln T/dx for every species, and ω_* = (T/q) k_y (κn + κT(v²/v_t² − 3)/2).
  For a convention with 1/L = −d ln N/dx (e.g. Maeyama) pass κ = −1/L. bslLD's `add_kappaT!` also uses
  the opposite sign.

## The building blocks

```julia
using KineticDR

ion = Species(κn = 0.23, κT = 0.34)                    # q = m = T = N = 1, response = GordeyevSeries()
ele = Species(q = -1, response = Boltzmann())          # adiabatic electrons, T_e = T_i
model = Model((ion, ele), Quasineutrality())           # species tuple + field equation
k = Wavevector(ky = 4.0, kz = 0.025)                   # kx = 0 by default
dispersion(model, 1.05 + 0.01im, k)                    # D(ω); the modes are its roots
ω, ok = find_root(model, k, 1.05 + 0.01im)             # 1.0501 + 0.0104i, true
```

- `Species(; q, m, T, N, κn, κT, response)` gives charge, mass, temperature, density and gradients in
  the units above, plus the response. Electrons are `q = -1, m = 1/1836` (or a reduced mass).
- `Model(species, field)` takes any number of species.
- `dispersion(model, ω, k)` is a scalar, analytic in ω. Every solver below only needs this function.

Which response works with which model:

| Response | Physics | Local electrostatic | Local EM | Radial |
|---|---|:-:|:-:|:-:|
| `GordeyevSeries(; pmax = nothing)` | full orbit, all cyclotron harmonics (`pmax` truncates) | ✓ | ✓ | ✓ (fixed `pmax`) |
| `GordeyevIntegral()` | the same as an orbit integral (validation) | ✓ | ✓ | (not planned) |
| `Gyrokinetic(; c = 0, cperp = c, prescribed = false, nmu = 128)` | J₀², no harmonics (slab ITG/ETG); EM: p = 0 exactly, polarisation and E×B to lowest order in ω/Ω. `c`, `cperp`: magnetic drift (toroidal ITG, ∇B), μ by quadrature | ✓ | ✓ (drift: `prescribed = false`) | ✓ |
| `Boltzmann()` | adiabatic, δn = −qNφ/T; EM: the ζ → 0 limit of `DriftKinetic`, δn = −qNψ/T (E∥ = −ik∥ψ) | ✓ | ✓ (k∥ ≠ 0) | ✓ |
| `Cold()` | cold fluid (Stix S, D, P); E×B density-gradient drive, no κT | ✓ | ✓ | ✓ |
| `DriftKinetic(; polarization = false)` | no Larmor radius: p = 0 at Γ₀ = 1; EM the m → 0 limit of the hot tensor | ✓ | ✓ | ✓ |

Field equations: `Quasineutrality()` and `Poisson(lambda2 = (λ_D/ρ_i)²)` are electrostatic, local and
radial. `Darwin(beta)` and `Maxwell(beta, lambda2)` are electromagnetic and local only. A combination
that is not supported throws an `ArgumentError`.

## Local electrostatic modes

Ion Bernstein / cyclotron-harmonic modes with full-orbit ions (the PRL case):

```julia
model = Model((Species(κn = 0.23, κT = 0.34), Species(q = -1, response = Boltzmann())), Quasineutrality())
k = Wavevector(ky = 4.0, kz = 0.025)

find_modes(model, k; re = 0:0.25:4.5)                  # all roots found from a grid of seeds
harmonic_root(model, k, 1)                             # most unstable root next to harmonic p = 1
ks = [Wavevector(ky = ky, kz = 0.025) for ky in 4.0:0.25:6.0]
track_branch(model, ks, 1.0501 + 0.0104im)             # follow that root in k_y (NaN where lost)
count_roots(ω -> dispersion(model, ω, k), 0.8, 1.3, 0.001, 0.05)   # argument principle: roots in a box
```

Slab ITG with gyrokinetic ions, kinetic electrons and Poisson:

```julia
ion = Species(κT = -0.3, response = Gyrokinetic())
ele = Species(q = -1, m = 1 / 1836, κT = -0.3, response = Gyrokinetic())
itg = Model((ion, ele), Poisson(lambda2 = 0.01))
find_modes(itg, Wavevector(ky = 0.5, kz = 0.01); re = -0.1:0.02:0.0, imag = (0.005, 0.02))
```

Stability boundaries. D is affine in the gradients of each species, and these act on species 1 unless
`species = i` is given:

```julia
critical_kappaT(model, k, 0.23, 1)                     # smallest κT for an unstable p = 1 root at κn = 0.23
neutral_gradients(model, 1.05, k)                      # (κn, κT) that put a root exactly at ω = 1.05
with_gradients(model; κn = 0.1, κT = 0.5)              # the same model with other gradients
```

Toroidal ITG: `Gyrokinetic(c = …, cperp = …)` adds the curvature drift (`c`, ∝ v∥²) and the ∇B drift (`cperp`, ∝ μ),
ω_D = −k_y (T/q)(c v∥²/v_t² + c⊥ μ), c = c⊥ = 1/(BR) in 1/ρ_i for a low-β torus (c > 0 is bad curvature, ω_D along
the ion diamagnetic direction for κ < 0). Without drift (the default) both velocity integrals are exact; with drift μ
becomes a quadrature, so the two settings differ numerically:

```julia
ion = Species(κn = -2 / 50, κT = -8 / 50, response = Gyrokinetic(c = 1 / 50))      # R = 50 ρ_i, R/L_T = 8
find_modes(Model((ion, Species(q = -1, response = Boltzmann())), Quasineutrality()), Wavevector(ky = 0.5);
           re = -0.06:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)                 # -0.02899 + 0.01309i
```

- The v∥ average is exact for any Im ω (two Z functions, `quadratic_average`); v⊥ is an `nmu`-point
  Gauss–Laguerre quadrature in μ = v⊥²/2v_t². **Converge `nmu`** when ω_D ~ ω: the μ integrand has a pole at a
  distance ~|ω/(k_y c⊥)| from the real axis (error 1e-3 at `nmu` = 24, 1.4e-5 at 80, 1e-9 at 320 for k_y c = 0.01),
  and at large k⊥ρ the oscillation of J₀² needs `nmu` ≳ (k⊥ρ_s)² (at k⊥ρ_s = 10: 64 nodes 15% off `Gyrokinetic`, 128 exact).
- Below the real axis the quadrature turns the continuum of drift resonances into `nmu` cuts at real
  ω = k_z² v_t²/(4 k_y τ c) − k_y τ c⊥ μ_k (τ = T/q): damped drift modes there depend on `nmu`. Trust Im ω > 0, converge `nmu` otherwise.
- `prescribed = true` puts ω_D in the drive numerator as well (bslLD's prescribed drift, which does no work against
  E); `false` (the default) is gyrokinetic. `prescribed = true` is electrostatic only.
- **Equilibrium ∇B.** c⊥ = −∂x ln B. A slab in pressure balance has c = 0 and
  c⊥ = `pressure_balance_cperp(species, β)` = (β/2) Σ_s N_s T_s (κn_s + κT_s), a ∇B along −∇p (good-curvature-like
  drift); pass the same β to the field. The diamagnetic drive is the gradient at fixed energy (flux-tube convention):
  ∇B enters only through ω_D.
- **Electromagnetic.** The tensor carries the gyrokinetic currents with the drift: the gyrocentre current of h,
  h = (ω − ω_*)⟨χ⟩/(ω − k∥v∥ − ω_D), including its drift current J₀ v_D h, while the non-resonant part (pull-back and
  magnetisation) does not drift; φ in Coulomb gauge. So the longitudinal row is the electrostatic charge
  (continuity), c = 0 is the electromagnetic `Gyrokinetic`, and under the gyrokinetic ordering (ω, k∥, κ, c ∝ ε → 0)
  Darwin with it converges to the gyrokinetic field equations in (φ, A∥, δB∥) linearly in ε (tested, β = 0.03,
  δB∥ moving ω by 10%). Without the pull-back correction the tensor misses continuity by O(ω_D/ω).

```julia
β, g = 0.05, (κn = -0.05, κT = -0.5)
cp = pressure_balance_cperp((Species(; g...), Species(; q = -1, g...)), β)   # slab in pressure balance: -0.0275
gb = Gyrokinetic(c = 0, cperp = cp)
m = Model((Species(; g..., response = gb), Species(; q = -1, m = 1 / 1836, g..., response = gb)), Darwin(beta = β))
find_modes(m, Wavevector(ky = 0.5, kz = 0.02); re = -0.3:0.03:0.3, imag = (0.02, 0.1))
```

## Local electromagnetic modes

```julia
using LinearAlgebra: svd

β, μ = 0.1, 1 / 100                                    # β = 2μ₀ N T_i/B², μ = m_e/m_i
em = Model((Species(), Species(q = -1, m = μ, response = DriftKinetic())), Darwin(beta = β))
k = Wavevector(ky = 0.5, kz = 0.2)                     # any k⊥ = (kx, ky)
ω, ok = find_root(em, k, 0.8 + 0.01im)                 # root of det(ω²𝒟): 0.6472 - 0.1140i
D = dispersion_matrix(em, ω, k)                        # 3×3 𝒟, with 𝒟 Ê = 0 at a root
Ê = svd(D).V[:, end]                                   # polarisation: null vector of 𝒟
```

- `Darwin(beta)` is quasineutral Darwin: there is no vacuum term, and the longitudinal part is quasineutrality.
- `Maxwell(beta, lambda2 = (v_A/c)²)` adds the vacuum term; its longitudinal part is Poisson.
- 𝒟 = Σ_s χ_s − (2/β)(k²/ω²)(I − k̂k̂) (+ vacuum) in units of ω_pi²/Ω_i², and `dispersion` returns det(ω²𝒟).
- `dispersion` is a scalar of arbitrary size: only its zeros mean anything. All the solvers above apply
  to it.
- Discard roots at ω ≈ 0. det(ω²𝒟) vanishes there for k∥ = 0, and with `DriftKinetic` electrons also for k∥ ≠ 0.
- **`Gyrokinetic` and `Boltzmann`.** `Gyrokinetic` is the hot tensor's p = 0 term plus the polarisation and E×B
  currents of p ≠ 0 to lowest order in ω/Ω. `Boltzmann` is the ζ → 0 limit of `DriftKinetic`: an isothermal
  massless fluid along the field line, δn = −qNψ/T (E∥ = −ik∥ψ), with the E×B and the diamagnetic current. It
  lacks only the transit-time damping of δB∥, O(k⊥²ρ²).
- **Gradients.** The drive enters the current as in the electrostatic model: the Maxwellian is weighted by
  1 − ω_*(v)/ω, so M → (1 − ωn/ω)M − (ωT/ω)(M + T∂_T M) (`susceptibility(r, ω, k, ωn, ωT)`, from
  `susceptibility_T`). The local current misses the divergence of the current across the density gradient,
  κn j_x, so each species adds that charge to the longitudinal row only. That row is then the full charge: its
  electrostatic limit −k·χ·k is `charge_response` exactly, and the EM roots tend to the electrostatic ones as β → 0.
  `Boltzmann`, the ζ → 0 limit of `DriftKinetic`, gets the same drive: its density response carries the magnetic
  flutter across the gradient, the kinetic −(ω_*/ω)(φ − ψ).
- **Polarisation and the density gradient.** The polarisation of `DriftKinetic(polarization = true)` and `Cold` is
  Boussinesq: the electrostatic −w N m k⊥², not weighted by the gradient drive. The density gradient is advected by
  the E×B drift only, as in the local kinetic theory: the polarisation current across the gradient is dropped. So
  `Cold` is the T → 0 limit of `GordeyevSeries` also with κn ≠ 0 (locally; for the radial problem see "No Larmor
  radius" below).
- Limits:
  - Real k only.
  - For a kinetic species with 0 < k∥ ≪ ω, Z(ζ) overflows below the real axis; use k∥ = 0 exactly for
    perpendicular propagation.

## Radial eigenmodes

Gyrokinetic (with or without magnetic drift), full-orbit, drift-kinetic, cold or Boltzmann species with radial profiles in a shearless slab (B = 1, k∥ = k_z everywhere). Use the same
`Model`, but give a species as a function x → `Species`. The radial grid turns the problem into an
operator A(ω) acting on φ(x), and an eigenmode is an ω at which A(ω) is singular.

```julia
T(x) = exp(-1.2tanh(x / 4))                            # κT = -0.3 sech²(x/4)
ion = species_profile(; T, response = Gyrokinetic())   # x -> Species, κn and κT from N(x), T(x)
ele = species_profile(; q = -1, N = particle_density(ion), response = Boltzmann())
m = Model((ion, ele), Quasineutrality())               # or Poisson(lambda2 = ...)
k = Wavevector(ky = 0.5, kz = 0.01)                    # kx = 0: it comes from the radial basis
g = radial_grid(-20, 20, 80; bc = :dirichlet)          # φ = 0 at x = ±20; or bc = :mirror (B = 1 grids only)

seeds = local_roots(m, k, -8:2:8; re = -0.08:0.01:0, imag = (0.002, 0.01))  # local roots along x
modes = find_eigenmodes(g, m, k; re = -0.06:0.005:0.0, imag = 0.001:0.002:0.015, starts = seeds)
r = first(modes)                                       # most unstable: ω = -0.02799 + 0.00682i
r.φ                                                    # φ(x) on the grid nodes, a PlasmaCore ScalarField
eigenmode(g, m, k, r.ω + 0.001)                        # refine one mode from a guess: (; ω, φ, λ, converged)
count_eigenvalues(g, m, k, -0.06, 0.0, 0.001, 0.015)   # how many eigenvalues the box holds (here 4)
dispersion(m(0.0), r.ω, k)                             # m(x) is the local model at x
```

Full-orbit species (ion Bernstein waves) keep the Larmor harmonics |p| ≤ `pmax` and need the periodic basis:

```julia
ion = species_profile(; N = x -> exp(0.46sin(x / 2)), T = x -> exp(0.68sin(x / 2)),    # κn = 0.23cos(x/2), κT = 0.34cos(x/2)
                      response = GordeyevSeries(pmax = 16))   # harmonics |p| ≤ 16
ele = species_profile(; q = -1, N = particle_density(ion), response = Boltzmann())
m = Model((ion, ele), Quasineutrality())
k = Wavevector(ky = 5.0, kz = 0.025)
g = radial_grid(-2π, 2π, 64; bc = :periodic)                   # periodic profiles; plane-wave basis
seeds = local_roots(m, k, -2:1:2; re = 0.8:0.1:1.2, imag = (0.002, 0.01))
r = find_eigenmode(g, m, k; re = 1.0:0.02:1.1, imag = 0.001:0.002:0.011, starts = seeds)   # ω = 1.04348 + 0.00733i
```

- **Profiles.** `species_profile(; N, T, q, m, response = Gyrokinetic())` derives κn, κT from N(x), T(x).
  A hand-written `x -> Species(...)` (any callable) uses its κn, κT exactly as given. A plain `Species`
  is uniform.
- **Guiding-centre convention (bsl6d).** The N(x), T(x) of a kinetic species are guiding-centre profiles.
  - Its adiabatic term is evaluated at the particle position, as an average over the gyroradius, which
    needs the profiles about 12 ρ beyond the walls.
  - `particle_density(ion)` is the ion particle density; pass it as the Boltzmann electrons' `N` for
    bsl6d's quasineutral equilibrium.
  - A `Boltzmann` species' N is taken at the node.
- **Exact integrals.** The v∥ and μ integrals are exact (Z functions and Weber's integral); only x and, for
  full-orbit species, the harmonics are truncated. Converge in N and `pmax` (the IBW example: N = 64 → 96 changes
  ω by 5e-14, `pmax` = 16 → 24 by 1.4e-5). The workspace holds 2N²(`pmax` + 1)·n_ρ floats per kinetic species,
  n_ρ the number of distinct gyroradii on the grid: 1 for uniform T, N otherwise (840 MB at N = 128, `pmax` = 24);
  above 4 GB it warns.
- **Full orbit.** `GordeyevSeries` needs a fixed `pmax ≥ 0`: the adaptive cut-off of the local model, taken at the
  grid's highest wavenumber, needs far more harmonics than the mode (P = 132 at N = 64, 252 at N = 128 in the IBW
  example, which changes by 1.4e-5 from P = 16 to 24). It also needs `bc = :periodic` with periodic profiles (checked at
  the walls): harmonic p carries the orbit phase e^{−i sgn(q) pθ_k}, θ_k = atan2(k_x, k_y), which leaves a
  standing-wave basis. With guiding-centre profiles the operator is exact up to the cut-off: its symbol for an
  exponential density is `exponential_dispersion(ω, kx′, ky, kz; κn, T, Te)`, the exact local dispersion relation
  of that profile (ions). Negative charges rest on the mirror symmetry y → −y (charge −q at k_y = charge q at
  −k_y), which the tests check, plus the uniform-profile limit; there is no exact non-uniform electron reference.
  `Gyrokinetic` and `GordeyevSeries(pmax = 0)` are the same kernel.
- **Magnetic drift.** A `Gyrokinetic` species with a drift anywhere on the grid takes the response's `nmu`
  Gauss–Laguerre nodes in μ instead of Weber's integral (W is no longer affine in μ), with one gyroaverage matrix per
  node, at every x; c, c⊥ and `prescribed` may depend on x and vanish somewhere (`response = x -> Gyrokinetic(c = 1 / (R0 + x))`),
  `nmu` may not. Without drift at every x the exact Weber kernel is used. The quadrature under-resolves the
  high-k_x columns of A (‖ΔA‖/‖A‖ = 3% at k_y = 0.5, 40% at k_y = 5 for 64 nodes, N = 128), which localised
  modes hardly feel: converge `nmu` as well as N. B = 1 and k∥ = k_z everywhere still: c is a prescribed drift,
  not a field-line geometry.
- **No Larmor radius.** `DriftKinetic` and `Cold` species are local in x except for the polarisation charge
  −w ∂x(N m ∂x φ), w = 1 (`DriftKinetic(polarization = true)`, else 0) or Ω²/(Ω² − ω²) (`Cold`). The flux form
  carries the polarisation current across the density gradient (the local models' Boussinesq −w N m k⊥² cannot), so
  it is the ρ → 0 limit of the radial gyrokinetic and full-orbit kernels. Its N(x) is the profile: a uniform `Species`
  with κn ≠ 0 drives ω_* but has a uniform N, so no flux term, and reduces to the local model. The density gradient is advected by the E×B
  drift only, as in the local models; the radial full-orbit operator as T → 0 has the exact cold drift
  w(E_y − iωE_x/Ω)/B, and the tests add back its non-E×B drive, (w − 1) q N κn k_y/ω, and then agree to 1e-5.
- **Walls.** `:dirichlet` puts φ = 0 on both walls (the kinetic wall of the bslLD runs); `:mirror` makes
  φ even about both walls. A mode localised away from the walls must give the same ω with both.
- **Finding modes.**
  - `find_eigenmodes` is a seed search and can miss nearly degenerate modes. It returns only modes
    inside the `re × imag` rectangle. Seed it with `local_roots` and check with `count_eigenvalues`. In
    the example the grid search alone finds 3 of the 4.
  - `eigenmode` reports `converged` when the step is small and |λ_min(A)| is small and resolved. That is
    convergence on this grid. Below the real axis the discretised continuum also has eigenvalues, which
    move with N, so check damped modes by refining the grid. Far below the axis, where Z overflows, it
    returns `converged = false`.
  - `count_eigenvalues` builds A on about 1000 points of the rectangle's boundary: seconds at N = 80,
    minutes at N = 240.
- **Checks.** With uniform profiles the radial eigenvalue on the k_x = 0 mode is the local root.
  `dispersion(m(x), ω, k)` compares with the local model at any x.

## Solvers

| Function | Purpose |
|---|---|
| `find_root(model, k, ω0)`, `find_root(f, ω0)` | Muller iteration from a guess → `(ω, converged)`. `f` is any analytic function of ω. |
| `find_modes(model, k; re, imag)` | roots from a grid of seeds, de-duplicated |
| `harmonic_root(model, k, p)` | most unstable root within ±0.4 of harmonic p |
| `track_branch(model, ks, ω0)` | continuation of one root along a list of wavevectors |
| `count_roots(f, re0, re1, im0, im1)` | number of roots in a rectangle (argument principle) |
| `gradient_coefficients`, `neutral_gradients`, `critical_kappaT`, `with_gradients` | gradient scans (electrostatic) |
| `eigenmode`, `find_eigenmodes`, `count_eigenvalues`, `local_roots`, `operator` | radial problem |

A root counts as converged when the Newton step |D/D′| is below `ftol` (default 1e-8) relative to
|ω|. That does not depend on the scale of D.

## Extending

- A new **response** is a subtype of `AbstractResponse` with any of:
  - `gordeyev(r, ω, k) -> (S, S_T)` for the electrostatic path: the Gordeyev sum and its temperature
    derivative, in the species' own units (ω in |Ω_s|, k in 1/ρ_s);
  - `susceptibility_T(r, ω, k) -> (M, T∂_T M)` for the electromagnetic path: the 3×3 tensor M in the frame where
    k⊥ ∥ x̂ and its temperature derivative (the drive of a temperature gradient). Mind the convention: the
    ω_*T weight is M + T∂_T M here, but S_T itself in the electrostatic drive −(ω − ωn)S + ωT S_T. A response
    without a temperature (`Cold`) defines only `KineticDR._susceptibility(r, ω, k)`, M without polarisation;
    its `Species` has no ω_*.
  The `Species` methods add the gradient drive, the units, the electron sign, the rotation to (kx, ky) and the
  gradient charge; override `KineticDR.charge_vector(::Species{R}, ω, k)` if j_x is not the E×B current.
  A fluid response without a Gordeyev sum (`Cold`) defines `charge_response(::Species{R}, ω, k)` instead of
  `gordeyev`: at the `Species` level, in lab units, with its own gradient drive. A polarisation outside the
  kinetic response comes from `polarization_weight(r, ω)` (species units), which sets both the charge
  −w N m k⊥² and M_xx = M_yy = ω² w, so the electrostatic and electromagnetic paths stay consistent.
- A new electrostatic **field equation** is a subtype of `AbstractFieldEquation` with
  `residual(field, ρs, k)`, where ρs is the tuple of the species' q_s δn_s/φ.

## Not implemented

- Magnetic drift in the full-orbit and drift-kinetic responses (only `Gyrokinetic`), and in the radial EM problem.
- Magnetic shear, B(x) and a consistent equilibrium ∇B in the radial problem.
- Electromagnetic fields in the radial problem.
- Complex k.

## Layout

```
src/KineticDR.jl   module, exports
src/special.jl     plasma Z function, Γ_p = I_p e^{-x}
src/types.jl       Wavevector, abstract types
src/responses.jl   GordeyevSeries, Gyrokinetic, Boltzmann, GordeyevIntegral, Cold, DriftKinetic
src/species.jl     Species, Model, charge_response, susceptibility, with_gradients
src/drift.jl       Gyrokinetic (with optional drift; electrostatic and electromagnetic), polynomial_average (v∥ average),
                   pressure_balance_cperp, Gauss–Laguerre nodes
src/fields.jl      Quasineutrality, Poisson, Darwin, Maxwell, dispersion, dispersion_matrix
src/solve.jl       root finders and gradient scans
src/radial.jl      profile species, radial grid and operator, eigenmode solvers
test/runtests.jl   tests
```
