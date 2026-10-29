# Radially resolved dispersion relation: for one (k_y, k_z), the field operator A(ω) acting on φ(x),
# built from a `Model` whose species may be radial profiles x -> `Species`. An eigenmode is an ω at
# which A(ω) is singular. Shearless slab, B = 1: k∥ = k_z everywhere.
#
# Each species contributes its charge response q_s δn_s/φ = (q_s² N_s/T_s)(W_s - 1) as an operator,
#
#   A(ω) = Σ_s diag(a_s) - Σ_{s kinetic} Σ_{|p| ≤ P_s} ∫dμ e^{-μ} J_spμ† diag[(q_s² N_s/T_s) W_spμ(x, ω)] J_spμ
#          + Σ_{s without Larmor radius} [-diag(q_s δn_s/φ at k_x = 0) + w_s(ω) X_s]  (+ field terms)
#
#   W_spμ = ⟨(ω - ω_*) / (ω - p|Ω_s| - k_z v∥)⟩_{v∥},
#
# the residual -Σ_s ρ_s of `Quasineutrality` (`Poisson` adds λ²k²). J_spμ is harmonic p of the gyroaverage
# at magnetic moment μ: J_p(k⊥ρ_sμ) with the orbit phase e^{-i p sgn(q_s) θ_k}, θ_k = atan2(k_x, k_y),
# ρ_sμ(x) = √(2μ T_s m_s)/|q_s|, |Ω_s| = |q_s|/m_s (B = 1). `Gyrokinetic` is P = 0, `GordeyevSeries(pmax = P)`
# the full orbit up to |p| ≤ P (P > 0 needs the periodic basis: the orbit phase leaves a standing-wave basis).
# ω_* = (T_s/q_s) k_y (κn + κT(μ + s²/2 - 3/2)) with κ = +d ln/dx (as everywhere in KineticDR), s = v∥/v_ts.
# Both integrals are exact: the v∥ average is two Z functions (any Im ω), and W is affine in μ, so the μ
# integral is Weber's second integral (`weber`). With guiding-centre profiles (below) the operator is exact
# up to the cut-off P (checked against `exponential_dispersion`). `GyrokineticDrift` adds the magnetic drift
# ω_D = -k_y (T_s/q_s)(c s² + c⊥ μ) to the p = 0 denominator, W = ⟨(ω - α ω_D - ω_*)/(ω - k_z v∥ - ω_D)⟩: the v∥ average
# is still exact (`quadratic_average`), the μ integral is the response's Gauss–Laguerre quadrature
# (`QuadratureOperator`). N_s, T_s of a kinetic species are guiding-centre profiles, so its
# adiabatic term is a_s = q_s² ∫F_s d³v/T_s at the particle position (`gc_average`); a `Boltzmann` species
# has W = 0 and a_s = q_s² N_s/T_s at the node. Species without finite Larmor radius (`DriftKinetic`, `Cold`)
# are local in x except for their polarisation charge w_s X_s (`PolarizationOperator`). Units as in the local model
# (Ω_i, ρ_i), φ in T_i/e.

# A species at radius x: a `Species` is uniform, anything else is a profile x -> `Species`.
species_at(s::Species, x) = s
species_at(s, x) = s(x)

"""
    (model::Model)(x) -> Model

The local (homogeneous) model at radius `x`: every profile species x -> `Species` evaluated there.
`dispersion(model(x), ω, k)` is the radial operator at uniform profiles on its k_x = 0 mode.
"""
(m::Model)(x) = Model(map(s -> species_at(s, x), m.species), m.field)

# A local solver handed a model with profiles (any species that is not a `Species`)
charge_response(s, ω, k::Wavevector) = _profile_error()
susceptibility(s, ω, k::Wavevector) = _profile_error()
_profile_error() = throw(ArgumentError("the model has radial profiles x -> Species: evaluate it at a radius, model(x), " *
                                       "or solve the radial problem (eigenmode)"))

"d ln f/dx of a profile, by central differences."
log_gradient(f; h = 1e-4) = x -> (log(f(x + h)) - log(f(x - h))) / (2h)

"""
    species_profile(; N = x -> 1.0, T = x -> 1.0, q = 1, m = 1, response = Gyrokinetic()) -> x -> Species

A species with profiles N(x), T(x) and gradients κn = d ln N/dx, κT = d ln T/dx derived from them
([`log_gradient`](@ref)). `response` is a response or a function x -> response.
"""
function species_profile(; N = x -> 1.0, T = x -> 1.0, q = 1.0, m = 1.0, response = Gyrokinetic())
    κn, κT = log_gradient(N), log_gradient(T)
    r = response isa AbstractResponse ? (x -> response) : response
    return x -> Species(; q, m, T = T(x), N = N(x), κn = κn(x), κT = κT(x), response = r(x))
end

# Guiding-centre profile f(species) seen at the particle position x: ∫dd g(d; ρ²(x + d)) f(s(x + d)), g the
# Gaussian of variance ρ² = T m/q² (squared thermal gyroradius, d = v_y m/q). Cut at 12 ρ_max, ρ_max the
# largest ρ within 12 ρ(x) of x (profiles may grow without bound); needs the profile there, beyond the walls.
function gc_average(f, sp, x)
    ρ2(s) = s.T * s.m / s.q^2
    w = 12 * sqrt(ρ2(species_at(sp, x)))
    dmax = 12 * sqrt(maximum(ρ2(species_at(sp, x + d)) for d in range(-w, w; length = 41)))
    function g(d)
        s = species_at(sp, x + d)
        return f(s) * exp(-d^2 / (2ρ2(s))) / sqrt(2π * ρ2(s))
    end
    return quadgk(g, -dmax, 0, dmax; rtol = 1e-12)[1]
end

"""
    particle_density(species) -> x -> n_p(x)

Particle density of a kinetic species whose N(x), T(x) are guiding-centre profiles (Gaussian
average over the gyroradius, `gc_average`). Electrons with `N = particle_density(ion)` are bsl6d's
quasineutral equilibrium (guiding-centre profiles, ion particle density in quasineutrality).
"""
particle_density(sp) = x -> gc_average(s -> s.N, sp, x)

"""
    exponential_dispersion(ω, kx′, ky, kz; κn, T = 1, Te = 1) -> D

Exact local dispersion function (the reference of the radial operator with profiles) of full-orbit ions with an exponential guiding-centre density
n ∝ e^{κn X} (κn = d ln n/dX, any size) and uniform T, Boltzmann electrons with the ion particle
density. The equilibrium is translation invariant, so an eigenmode is e^{i k_x x} = e^{-κn x/2} e^{i kx′ x} with real kx′
(the shift k_x = kx′ + iκn/2 makes D even in kx′). Same normalisation and sign as
`dispersion` of full-orbit ions with Boltzmann electrons of density n_p, i.e. (δn_e - δn_i)/(n_p φ), to which it reduces
for κn → 0. At O(κn),
D = D_Boussinesq + κn ω ky Σ_p p Γ_p(k⊥²) a Z((ω - p)a) / k⊥² + O(κn²), a = 1/(√2|kz|) (T = 1): the orbit phase
e^{-ipθ_k} varies across the gradient; the p = 0 term vanishes. Series over all harmonics with the
Bessel argument √(Q² - R²), Q = kx′² + ky² + κn²/4, R = -κn ky (in ρ at temperature T); needs kz ≠ 0 and Q ≠ |R|.
"""
function exponential_dispersion(ω, kxp, ky, kz; κn, T = 1.0, Te = 1.0)
    kz == 0 && throw(ArgumentError("exponential_dispersion needs kz ≠ 0"))
    s = sqrt(T)                                        # scale to T = 1: k → √T k, κ → √T κ, response / T
    kxp, ky, kz, κ = s * kxp, s * ky, s * kz, s * κn
    Q, R = kxp^2 + ky^2 + κ^2 / 4, -κ * ky
    x = sqrt(Q^2 - R^2)
    r = (Q - R) / (Q + R)
    a = 1 / (sqrt(2.0) * abs(kz))
    mmax = ceil(Int, Q + abs(R) + 10 * sqrt(Q + abs(R) + 1) + 20)
    S = sum(r^(m / 2) * besselix(abs(m), x) * plasma_Z((ω - m) * a) for m in -mmax:mmax)
    G = -im * a * exp(x - Q + κ^2 / 2) * S             # ∫_{-∞}^0 e^{A_n - iωt} dt
    δn = (-exp(κ^2 / 2) - im * (ω + R) * G) / T          # ion density response per φ, gyrocentre density 1 at x = 0
    return 1 / Te - exp(-κ^2 / 2) * δn                  # (δn_e - δn_i)/(n_p φ), n_p = e^{κ²T/2}
end

# Adiabatic term q² ∫F d³v / T at the node: exact for a guiding-centre profile, N/T at the node for a
# uniform or a Boltzmann species.
adiabatic_term(s::Species, x) = s.q^2 * s.N / s.T
function adiabatic_term(sp, x)
    s = sp(x)
    s.response isa Boltzmann && return adiabatic_term(s, x)
    return s.q^2 * gc_average(t -> t.N / t.T, sp, x)
end

"""
    RadialGrid(grid::PlasmaCore.Grid; bc = :dirichlet)
    radial_grid(a, b, N; kw...)

Radial discretisation on the first spatial axis of a PlasmaCore `Grid` (nodes a + (j-1)δ,
δ = (b - a)/N, so N nodes on [a, b)); `radial_grid` builds the one-axis grid. The grid must have B = 1
along z.
- `bc = :dirichlet`: φ = 0 on the walls a and b; unknowns on the N - 1 nodes after a, sine basis
  sin(π m (x - a)/L). This is the kinetic wall of the bslLD runs, whose eigenmodes vanish there.
- `bc = :mirror`: unknowns at the cell centres x_j + δ/2 (N of them), cosine basis, even about both
  walls (bslLD's `Mirror()` halo). Admits wall-localised modes the simulation does not have.
- `bc = :periodic`: unknowns at the N nodes (N even), plane waves e^{ik(x - a)}, k = 2πm/L, m = -N/2 … N/2 - 1.
  Needs periodic profiles (checked: N, T, κn, κT equal at a and b). The only basis that carries the Larmor
  harmonics p ≠ 0 (`GordeyevSeries`). The Nyquist mode m = -N/2 is the same on the nodes as m = +N/2 but carries
  the orbit phase of -N/2 only; its weight vanishes with N (converge in N).
Eigenvectors are returned as `ScalarField`s with one value per grid node (zero at the wall node for Dirichlet; cell
centres for mirror; the nodes for periodic).
"""
struct RadialGrid{G,T<:Number}
    grid::G
    bc::Symbol
    x::Vector{Float64}          # nodes carrying unknowns
    k::Vector{Float64}          # basis wavenumbers (π m / L standing waves, 2π m / L periodic)
    C::Matrix{T}                # C[j, m] = basis_m(x_j), complex for :periodic
    Cinv::Matrix{T}
end

function RadialGrid(grid; bc = :dirichlet)
    (grid.b0 == 1 && grid.Bdir == 3) ||
        throw(ArgumentError("the radial problem assumes B = 1 along z (got b0 = $(grid.b0), Bdir = $(grid.Bdir))"))
    ax = grid.xaxes[1]
    N, h, a = length(ax), step(ax), first(ax)
    L = N * h
    if bc === :mirror
        x = collect(ax) .+ h / 2
        k = [π * m / L for m in 0:(N - 1)]
        C = [cos(k[m] * (x[j] - a)) for j in 1:N, m in 1:N]
    elseif bc === :dirichlet
        x = collect(ax)[2:end]
        k = [π * m / L for m in 1:(N - 1)]
        C = [sin(k[m] * (x[j] - a)) for j in 1:(N - 1), m in 1:(N - 1)]
    elseif bc === :periodic
        iseven(N) || throw(ArgumentError("bc = :periodic needs an even N"))
        x = collect(ax)
        k = [2π * m / L for m in (-N ÷ 2):(N ÷ 2 - 1)]
        C = [cis(k[m] * (x[j] - a)) for j in 1:N, m in 1:N]
        return RadialGrid(grid, bc, x, k, C, C' / N)
    else
        throw(ArgumentError("bc must be :mirror, :dirichlet or :periodic, got $bc"))
    end
    return RadialGrid(grid, bc, x, k, C, inv(C))
end

radial_grid(a, b, N; kw...) = RadialGrid(Grid([float(a)], [float(b)], [Int(N)], 1); kw...)

"φ on the unknown nodes as a `ScalarField` over the grid axis."
function radial_field(g::RadialGrid, φ::AbstractVector)
    return ScalarField(g.bc === :dirichlet ? vcat(zero(eltype(φ)), φ) : Vector(φ))
end

# With the magnetic drift (p = 0): the local `drift_vpar` in the species' own units.
function vpar_response(k::Wavevector, s::Species{GyrokineticDrift}, ω, mu)
    Ω, ks = _species_units(s, k)
    return drift_vpar(own_units(s.response, s), ω / Ω, ks, _drive(s, k, Ω)..., mu)
end

# W_p = ⟨(ω - ω_*)/(ω - p|Ω| - k_z v∥)⟩ over v∥ for local species `s` at one μ, harmonic p: with
# s = v∥/v_t ~ N(0, 1), ⟨1/(ω' - b s)⟩ = -a Z(ζ), ⟨s²/(ω' - b s)⟩ = -2a ζ (1 + ζZ(ζ)), ω' = ω - p|Ω|,
# b = |k_z| v_t, a = 1/(√2 b), ζ = aω'.
function vpar_response(k::Wavevector, s::Species, ω, mu, p::Integer = 0)
    dia = s.T / s.q * k.ky
    n0 = ω - dia * (s.κn + s.κT * (mu - 1.5))
    n2 = -dia * s.κT / 2
    I0, I2 = linear_moments(ω - p * abs(cyclotron_frequency(s)), abs(k.kz) * thermal_speed(s))
    return n0 * I0 + n2 * I2
end

"""
    weber(p, a, b) -> (W0, W1)

∫₀^∞ dμ e^{-μ} μ^k J_p(a√(2μ)) J_p(b√(2μ)) for k = 0, 1 (Weber's second exponential integral):
W0 = e^{-s} I_p(ab), W1 = e^{-s} [(1 - s) I_p(ab) + ab I_p'(ab)], s = (a² + b²)/2. For a = b,
W0 = Γ_p(a²). Valid for complex a, b (the scaled Bessel function is unscaled by e^{|Re ab|}).
"""
function weber(p::Integer, a, b)
    c = a * b
    d = (a - b)^2 / 2                                  # s = c + d
    e = exp(abs(real(c)) - c - d)                      # e^{-s} I_p(c) = e · besselix(p, c)
    I = besselix(abs(p), c)
    dI = (besselix(abs(p - 1), c) + besselix(abs(p + 1), c)) / 2
    # (1 - s) I + c I' = (1 - d) I + c (I' - I): no cancellation of the large terms s I and c I'
    return e * I, e * ((1 - d) * I + c * (dI - I))
end

# Harmonic cut-off P of a radial kernel: J₀ only for `Gyrokinetic`, |p| ≤ pmax for `GordeyevSeries`.
harmonics(::Gyrokinetic) = 0
function harmonics(r::GordeyevSeries)
    r.pmax === nothing && throw(ArgumentError("radial problem: GordeyevSeries needs a fixed cut-off, GordeyevSeries(pmax = P)"))
    r.pmax >= 0 || throw(ArgumentError("GordeyevSeries: pmax must be ≥ 0, got $(r.pmax)"))
    return r.pmax
end

# Kinetic species: the μ integral in closed form. With G_p[j, l] = C[j, l] e^{-iσpθ_l} J_p(k⊥_l ρ_jμ), σ = sgn(q),
#   Σ_p ∫dμ e^{-μ} G_p† diag(w (r0_p + r1_p μ)) G_p = M,
#   M[l, l'] = Σ_p e^{iσp(θ_l - θ_l')} Σ_j conj(C[j, l]) w_j (r0_pj W0_|p|[ρ_j, l, l'] + r1_pj W1_|p|[ρ_j, l, l']) C[j, l'],
# W0, W1 = weber(p, k⊥_l ρ_j, k⊥_l' ρ_j), ρ_j = √(T m)/|q| at node j: even in p, ω-independent, one table per
# |p| and distinct ρ_j (a single one for uniform T).
abstract type RadialKernel end

struct WeberOperator{S} <: RadialKernel
    ss::Vector{S}
    P::Int
    σ::Int
    iρ::Vector{Int}                # node j -> index of its gyroradius in the tables
    W0::Vector{Array{Float64,3}}   # [iρ, l, l'], index |p| + 1
    W1::Vector{Array{Float64,3}}
    θ::Vector{Float64}
    u0::Matrix{ComplexF64}         # buffers for +p and -p
    u1::Matrix{ComplexF64}
    v0::Matrix{ComplexF64}
    v1::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
end

# Warn above this size of the Weber tables of one species (bytes)
const WEBER_TABLE_WARN = 4e9

check_charge_mass(ss) = all(s -> s.q == ss[1].q && s.m == ss[1].m, ss) ||
    throw(ArgumentError("radial problem: a species' charge and mass must not depend on x"))

function WeberOperator(g::RadialGrid, k::Wavevector, ss, P)
    N = length(g.x)
    check_charge_mass(ss)
    ρs = [sqrt(s.T * s.m) / abs(s.q) for s in ss]
    ρu = unique(ρs)
    iρ = [findfirst(==(ρ), ρu) for ρ in ρs]
    bytes = 16.0 * (P + 1) * length(ρu) * N^2
    bytes > WEBER_TABLE_WARN && @warn "radial problem: Weber tables of $(round(bytes / 1e9, digits = 1)) GB for one species " *
                                       "(N = $N, pmax = $P, $(length(ρu)) distinct gyroradii)"
    kp = [hypot(q, k.ky) for q in g.k]
    W0 = [zeros(length(ρu), N, N) for _ in 0:P]
    W1 = [zeros(length(ρu), N, N) for _ in 0:P]
    for p in 0:P, (u, ρ) in enumerate(ρu), l in 1:N, l2 in l:N
        W0[p + 1][u, l, l2], W1[p + 1][u, l, l2] = weber(p, kp[l] * ρ, kp[l2] * ρ)
        W0[p + 1][u, l2, l], W1[p + 1][u, l2, l] = W0[p + 1][u, l, l2], W1[p + 1][u, l, l2]
    end
    θ = [atan(q, k.ky) for q in g.k]
    z() = zeros(ComplexF64, N, N)
    return WeberOperator(ss, P, Int(sign(ss[1].q)), iρ, W0, W1, θ, z(), z(), z(), z(), z())
end

# Gyrokinetic species with the magnetic drift: W is not affine in μ, so the μ integral is the response's
# Gauss–Laguerre quadrature, Σ_μ w_μ J_μ† diag((q² N/T) W_μ) J_μ with the gyroaverage J_μ = C diag(J₀(k⊥ ρ_jμ)) Cinv
# (one matrix per node μ). It under-resolves the high-k_x columns (‖ΔA‖/‖A‖ = 3% at k_y = 0.5, 40% at k_y = 5 for
# 64 nodes, N = 128), which localised modes hardly feel: converge `nmu`.
struct QuadratureOperator{S,T} <: RadialKernel
    ss::Vector{S}
    J::Vector{Matrix{T}}
    d::Vector{ComplexF64}        # buffers: the diagonal, D J and the sum
    DJ::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
end

function QuadratureOperator(g::RadialGrid, k::Wavevector, ss)
    check_charge_mass(ss)
    J = map(ss[1].response.mu) do mu
        G = [g.C[j, l] * besselj0(hypot(g.k[l], k.ky) * sqrt(2mu * s.T * s.m) / abs(s.q)) for (j, s) in enumerate(ss), l in eachindex(g.k)]
        G * g.Cinv
    end
    N = length(g.x)
    return QuadratureOperator(ss, J, zeros(ComplexF64, N), zeros(ComplexF64, N, N), zeros(ComplexF64, N, N))
end

function kinetic_operator!(K::QuadratureOperator, g::RadialGrid, k::Wavevector, ω)
    fill!(K.M, 0)
    for (J, mu, w) in zip(K.J, K.ss[1].response.mu, K.ss[1].response.wmu)
        for (j, s) in enumerate(K.ss)
            K.d[j] = w * s.q^2 * s.N / s.T * vpar_response(k, s, ω, mu)
        end
        mul!(K.DJ, Diagonal(K.d), J)
        mul!(K.M, J', K.DJ, true, true)
    end
    return K.M
end

# Species without finite Larmor radius (`DriftKinetic`, `Cold`): local in x except for the polarisation charge,
#   A_s(ω) = -diag(charge_response(s(x_j), ω, k_x = 0)) + w_s(ω) X,   X = -∂x c ∂x = -c ∂x² - c' ∂x,  c = N m,
# w_s = `polarization_weight`, ∂x and ∂x² spectral on the basis. c' = m dN/dx is the slope of the profile N(x), not
# the drive κn: a uniform `Species` with κn ≠ 0 has c' = 0 and gives -charge_response at k_x = k_m on the basis, and
# for a profile X is the flux form, the ρ → 0 limit of the gyrokinetic kernel (which also takes N(x) from the profile
# and κn only for ω_*). X = nothing without polarisation (w ≡ 0).
struct PolarizationOperator{S} <: RadialKernel
    ss::Vector{S}
    X::Union{Nothing,Matrix{Float64}}
end

function PolarizationOperator(g::RadialGrid, ss, sp; h = 1e-4)
    check_charge_mass(ss)
    iszero(polarization_weight(ss[1].response, 0.0)) && return PolarizationOperator(ss, nothing)
    D1, D2 = derivative_matrices(g)
    c = [s.N * s.m for s in ss]
    c′ = [ss[1].m * (species_at(sp, x + h).N - species_at(sp, x - h).N) / (2h) for x in g.x]   # 0 for a `Species`
    return PolarizationOperator(ss, -Diagonal(c) * D2 - Diagonal(c′) * D1)
end

# Spectral ∂x and ∂x² at the nodes: derivatives of the basis interpolant (cos ↔ sin for the standing waves; the
# periodic Nyquist mode has no real derivative and gets none).
function derivative_matrices(g::RadialGrid)
    a, J, M = first(g.grid.xaxes[1]), eachindex(g.x), eachindex(g.k)
    dC = g.bc === :periodic ? g.C * Diagonal(im .* g.k) :
         g.bc === :dirichlet ? [g.k[m] * cos(g.k[m] * (g.x[j] - a)) for j in J, m in M] :
         [-g.k[m] * sin(g.k[m] * (g.x[j] - a)) for j in J, m in M]
    return real.(dC * g.Cinv), real.(g.C * Diagonal(-g.k .^ 2) * g.Cinv)
end

# Adds the kernel's -charge on the nodes to A
add_kernel!(A, K::Union{WeberOperator,QuadratureOperator}, g::RadialGrid, k::Wavevector, ω) = (A .-= kinetic_operator!(K, g, k, ω))
function add_kernel!(A, L::PolarizationOperator, g::RadialGrid, k::Wavevector, ω)
    L.X === nothing || (A .+= polarization_weight(L.ss[1], ω) .* L.X)
    k0 = Wavevector(0.0, k.ky, k.kz)
    for (j, s) in enumerate(L.ss)
        A[j, j] -= charge_response(s, ω, k0)
    end
    return A
end

# Adiabatic terms a_s the workspace puts on the diagonal (Boltzmann: no kernel); `charge_response` holds them for the
# species without Larmor radius
adiabatic_terms(K, sp, g::RadialGrid) = [adiabatic_term(sp, x) for x in g.x]
adiabatic_terms(::PolarizationOperator, sp, g::RadialGrid) = zeros(length(g.x))

# ω-independent size of what a kernel adds outside `ws.diag` (convergence test in `eigenmode`): for the species without
# Larmor radius, the polarisation charge at k_x ~ 1 and their adiabatic term q²N/T (none for a cold fluid)
kernel_scale(K, k::Wavevector) = 0.0
kernel_scale(L::PolarizationOperator, k::Wavevector) = maximum(s -> polarization_weight(s, 0.0) * s.N * s.m * (k.ky^2 + 1) +
                                                                    (s.response isa Cold ? 0.0 : s.q^2 * s.N / s.T), L.ss)

# Radial kernel of a species, by its response (the same at every x): `nothing` for `Boltzmann` (adiabatic term only),
# a `WeberOperator` for the kinetic responses (a `QuadratureOperator` with drift), a `PolarizationOperator` for those without Larmor radius.
# `ss` is the species at the nodes, `sp` the profile x -> `Species` (or a uniform `Species`).
radial_kernel(::Boltzmann, g::RadialGrid, k::Wavevector, ss, sp) = nothing
function radial_kernel(r::Union{Gyrokinetic,GordeyevSeries}, g::RadialGrid, k::Wavevector, ss, sp)
    P = harmonics(r)
    P > 0 && g.bc !== :periodic && throw(ArgumentError("Larmor harmonics (pmax = $P > 0) need bc = :periodic"))
    return WeberOperator(g, k, ss, P)
end
radial_kernel(::GyrokineticDrift, g::RadialGrid, k::Wavevector, ss, sp) = QuadratureOperator(g, k, ss)
radial_kernel(::Union{DriftKinetic,Cold}, g::RadialGrid, k::Wavevector, ss, sp) = PolarizationOperator(g, ss, sp)
radial_kernel(r::AbstractResponse, g::RadialGrid, k::Wavevector, ss, sp) = throw(ArgumentError(
    "radial kernels: Boltzmann, DriftKinetic, Cold, Gyrokinetic, GyrokineticDrift or GordeyevSeries(pmax = P); got $(nameof(typeof(r)))"))

# What must agree along x: the response, for the kinetic ones only the cut-off (`Gyrokinetic` = `GordeyevSeries(pmax = 0)`),
# with drift the μ nodes (c, c⊥, α may vary)
radial_key(r::AbstractResponse) = r
radial_key(r::Union{Gyrokinetic,GordeyevSeries}) = harmonics(r)
radial_key(r::GyrokineticDrift) = (GyrokineticDrift, r.mu)

# Real frequencies at which a kernel's response has a pole (the `eigenmode` guess must avoid them)
# With drift the real singular points are a continuum of branch points (one per μ node, any k_z): none to exclude
pole_frequencies(K::QuadratureOperator, k::Wavevector) = Float64[]
pole_frequencies(K::WeberOperator, k::Wavevector) =
    k.kz == 0 ? [p * abs(cyclotron_frequency(K.ss[1])) for p in -K.P:K.P] : Float64[]
# A cold species has poles at any k_z: its drive ∝ 1/ω and its polarisation weight 1/(1 - ω²/Ω²)
function pole_frequencies(L::PolarizationOperator, k::Wavevector)
    s = L.ss[1]
    s.response isa Cold && return [0.0, abs(cyclotron_frequency(s)), -abs(cyclotron_frequency(s))]
    return k.kz == 0 ? [0.0] : Float64[]
end

struct RadialWorkspace{G,M}
    g::G
    m::M
    k::Wavevector{Float64}
    kernels::Vector{RadialKernel}
    diag::Vector{Float64}    # adiabatic terms Σ_s a_s at the nodes
    fieldop::Matrix{Float64} # field terms (Poisson λ²k², real in every basis), zero for quasineutrality
    scale::Float64           # ω-independent scale of A (convergence test in `eigenmode`)
end

field_operator(::Quasineutrality, g::RadialGrid, k::Wavevector) = zeros(length(g.x), length(g.x))
field_operator(f::Poisson, g::RadialGrid, k::Wavevector) = f.lambda2 .* ((k.ky^2 + k.kz^2) * I - derivative_matrices(g)[2])
field_operator(f::AbstractFieldEquation, g::RadialGrid, k::Wavevector) = throw(ArgumentError(
    "$(nameof(typeof(f))) is not implemented for the radial problem (Quasineutrality, Poisson)"))

function RadialWorkspace(g::RadialGrid, m::Model, k::Wavevector)
    k.kx == 0 || throw(ArgumentError("radial problem: k_x comes from the basis, pass kx = 0 (got $(k.kx))"))
    fieldop = field_operator(m.field, g, k)
    N = length(g.x)
    diag = zeros(N)
    kernels = RadialKernel[]
    scale = opnorm(fieldop, 1)
    for sp in m.species
        ss = [species_at(sp, x) for x in g.x]
        allequal(radial_key(s.response) for s in ss) || throw(ArgumentError(
            "radial problem: a species needs the same response at every x; got $(unique(string.(getfield.(ss, :response))))"))
        g.bc === :periodic && check_periodic(g, sp)
        K = radial_kernel(ss[1].response, g, k, ss, sp)
        diag .+= adiabatic_terms(K, sp, g)
        scale += kernel_scale(K, k)
        K === nothing || push!(kernels, K)
    end
    return RadialWorkspace(g, m, k, kernels, diag, fieldop, scale + maximum(abs, diag))
end

# A periodic basis needs periodic profiles: the species at both walls must agree (to 1e-6: κ of a numerically
# integrated profile, e.g. `particle_density`, carries finite-difference noise ~1e-8).
function check_periodic(g::RadialGrid, sp)
    ax = g.grid.xaxes[1]
    a, b = first(ax), first(ax) + length(ax) * step(ax)
    sa, sb = species_at(sp, a), species_at(sp, b)
    all(isapprox(getfield(sa, f), getfield(sb, f); rtol = 1e-6, atol = 1e-6) for f in (:N, :T, :κn, :κT)) ||
        throw(ArgumentError("bc = :periodic needs periodic profiles: the species differs at x = $a and x = $b"))
end

"""
    operator(g, m, k, ω[, ws]) -> A::Matrix{ComplexF64}

Field operator on the unknown nodes; `A φ = 0` for an eigenmode. `k` gives k_y and k_z (k_x = 0:
it comes from the basis).
"""
function operator(g::RadialGrid, m::Model, k::Wavevector, ω::Number, ws = RadialWorkspace(g, m, k))
    (ws.g === g && ws.m === m && ws.k == k) ||
        throw(ArgumentError("the workspace was built for another grid, model or wavevector"))
    A = complex(ws.fieldop)
    for K in ws.kernels
        add_kernel!(A, K, g, ws.k, ω)
    end
    for j in eachindex(g.x)
        A[j, j] += ws.diag[j]
    end
    return A
end

# Σ_p ∫dμ e^{-μ} J_pμ† diag(w W_pμ) J_pμ = Cinv' M Cinv (see `WeberOperator`); p and -p share the tables
# and are summed in one pass.
function kinetic_operator!(K::WeberOperator, g::RadialGrid, k::Wavevector, ω)
    N = length(g.x)
    C = g.C
    fill!(K.M, 0)
    function fill_u!(u0, u1, p)
        for (j, s) in enumerate(K.ss)
            w = s.q^2 * s.N / s.T
            r0 = vpar_response(k, s, ω, 0.0, p)
            r1 = vpar_response(k, s, ω, 1.0, p) - r0
            for l in 1:N
                u0[j, l] = conj(C[j, l]) * w * r0
                u1[j, l] = conj(C[j, l]) * w * r1
            end
        end
    end
    for p in 0:K.P
        W0, W1, iρ = K.W0[p + 1], K.W1[p + 1], K.iρ
        fill_u!(K.u0, K.u1, p)
        p > 0 && fill_u!(K.v0, K.v1, -p)
        @inbounds for l2 in 1:N, l in 1:N
            a₊ = zero(ComplexF64)
            a₋ = zero(ComplexF64)
            for j in 1:N
                w0, w1, c = W0[iρ[j], l, l2], W1[iρ[j], l, l2], C[j, l2]
                a₊ += (K.u0[j, l] * w0 + K.u1[j, l] * w1) * c
                p > 0 && (a₋ += (K.v0[j, l] * w0 + K.v1[j, l] * w1) * c)
            end
            if p == 0
                K.M[l, l2] += a₊
            else
                e = cis(K.σ * p * (K.θ[l] - K.θ[l2]))
                K.M[l, l2] += e * a₊ + conj(e) * a₋
            end
        end
    end
    return g.Cinv' * K.M * g.Cinv
end

# eigenvalue of smallest modulus and its eigenvector; NaN if A is not finite (Z overflows far below the
# real axis)
function smallest(A)
    all(isfinite, A) || return complex(NaN), fill(complex(NaN), size(A, 1))
    E = eigen(A)
    i = argmin(abs.(E.values))
    return E.values[i], E.vectors[:, i]
end

"""
    eigenmode(g, m, k, ω0; tol = 1e-10, ltol = 1e-8, maxit = 60) -> (; ω, φ, λ, converged)

Secant iteration on λ_min(A(ω)) = 0 from the guess `ω0`, tracking the eigenvalue closest to zero.
`converged` needs the step below `tol |ω|` (a jump between eigenvalue branches can stall the step away from a root),
|λ| ≤ `ltol` s, with s the ω-independent scale of A: max over x of the summed adiabatic terms, plus ‖field terms‖₁,
plus for each species without Larmor radius its polarisation charge at k_x ~ 1 and q²N/T, and λ resolved: 100 eps ‖A‖₁ ≤ `ltol` s (far below the real axis ‖A‖ grows like e^{ζ²}).
It is algebraic convergence for this grid: below the real axis the discretised continuum has
eigenvalues that move with N, so check damped modes by refining the grid. An iteration that leaves
the region where A is finite returns `converged = false`. `φ` is a `ScalarField`, normalised to
max |φ| = 1 and real at the maximum.
"""
function eigenmode(g::RadialGrid, m::Model, k::Wavevector, ω0::Number; tol = 1e-10, ltol = 1e-8, maxit = 60,
                   ws = RadialWorkspace(g, m, k))
    isfinite(ω0) || throw(ArgumentError("eigenmode: the guess ω0 = $ω0 is not finite"))
    any(K -> ω0 in pole_frequencies(K, k), ws.kernels) &&
        throw(ArgumentError("eigenmode: ω0 = $ω0 is a pole of a species' response"))
    λ(ω) = first(smallest(operator(g, m, k, ω, ws)))
    a, b = complex(ω0), complex(ω0) * (1 + 1e-3) + 1e-5im
    fa, fb = λ(a), λ(b)
    converged = false
    for _ in 1:maxit
        isfinite(fb) || break
        Δ = -fb * (b - a) / (fb - fa)
        lim = 0.3 * max(abs(b), 1e-3)
        abs(Δ) > lim && (Δ *= lim / abs(Δ))
        a, fa = b, fb
        b += Δ
        fb = λ(b)
        if abs(Δ) < tol * max(abs(b), 1e-8)
            converged = true
            break
        end
    end
    A = operator(g, m, k, b, ws)
    λb, φ = smallest(A)
    scale = ws.scale
    # λ small on the ω-independent scale, and resolved: its rounding error eps‖A‖ well below that
    converged &= isfinite(λb) && abs(λb) <= ltol * scale && 100 * eps() * opnorm(A, 1) <= ltol * scale
    isfinite(λb) && (φ = φ ./ φ[argmax(abs.(φ))])
    return (; ω = b, φ = radial_field(g, φ), λ = λb, converged)
end

"""
    local_roots(m, k, xs; re, imag) -> Vector{ComplexF64}

Most unstable local root (`find_modes` on `m(x)`) at each radius, for seeding
[`find_eigenmodes`](@ref). `NaN` where there is none.
"""
function local_roots(m::Model, k::Wavevector, xs; re, imag)
    return map(xs) do x
        roots = find_modes(m(x), k; re, imag, imin = 0.0)
        isempty(roots) ? complex(NaN, NaN) : roots[argmax(Base.imag.(roots))]
    end
end

"""
    find_eigenmodes(g, m, k; re, imag, starts = []) -> Vector of `eigenmode` results

Evaluates |λ_min(A(ω))| on the rectangle `re × imag`, refines every local minimum (plus the guesses
`starts`, e.g. [`local_roots`](@ref) along x) with [`eigenmode`](@ref), and returns the distinct
converged modes inside the rectangle, most unstable first. A seed search: it can miss modes (see [`count_eigenvalues`](@ref));
nearly degenerate modes are common, so seed it generously.
"""
function find_eigenmodes(g::RadialGrid, m::Model, k::Wavevector; re, imag, starts = ComplexF64[], kw...)
    ws = RadialWorkspace(g, m, k)
    cand = [complex(a, b) for a in re, b in imag]
    L = [abs(first(smallest(operator(g, m, k, ω, ws)))) for ω in cand]
    na, nb = size(L)
    starts = vcat(ComplexF64.(filter(isfinite, starts)),
                  [cand[i, j] for i in 1:na, j in 1:nb if
                   all(L[i, j] <= L[i2, j2] for i2 in max(1, i - 1):min(na, i + 1),
                       j2 in max(1, j - 1):min(nb, j + 1))])
    results = [eigenmode(g, m, k, ω0; ws, kw...) for ω0 in starts]
    modes = empty(results)
    inbox(ω) = minimum(re) <= real(ω) <= maximum(re) && minimum(imag) <= Base.imag(ω) <= maximum(imag)
    for r in results
        (r.converged && inbox(r.ω)) || continue
        any(abs(r.ω - q.ω) < 1e-6 * abs(r.ω) for q in modes) && continue
        push!(modes, r)
    end
    return sort(modes; by = r -> -Base.imag(r.ω))
end

"Most unstable mode of [`find_eigenmodes`](@ref)."
find_eigenmode(g::RadialGrid, m::Model, k::Wavevector; kw...) = first(find_eigenmodes(g, m, k; kw...))

"""
    count_eigenvalues(g, m, k, re0, re1, im0, im1) -> Int

Number of eigenmodes inside the rectangle: zeros of det A(ω), by the argument principle
(`count_roots`); det A is entire in ω for k_z ≠ 0.
"""
function count_eigenvalues(g::RadialGrid, m::Model, k::Wavevector, re0, re1, im0, im1; kw...)
    ws = RadialWorkspace(g, m, k)
    return count_roots(ω -> det(operator(g, m, k, ω, ws)), re0, re1, im0, im1; kw...)
end
