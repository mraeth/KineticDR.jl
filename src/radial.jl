# Radially resolved dispersion relation: for one (k_y, k_z), the field operator A(ω) acting on φ(x),
# built from a `Model` whose species may be radial profiles x -> `Species`. An eigenmode is an ω at
# which A(ω) is singular. Shearless slab, B = 1: k∥ = k_z everywhere.
#
# Each species contributes its charge response q_s δn_s/φ = (q_s² N_s/T_s)(W_s - 1) as an operator,
#
#   A(ω) = Σ_s diag(a_s) - Σ_{s kinetic} ∫dμ e^{-μ} J_sμ† diag[(q_s² N_s/T_s) W_sμ(x, ω)] J_sμ  (+ field terms)
#
#   W_sμ = ⟨(ω - ω_*) / (ω - k_z v∥)⟩_{v∥},
#
# the residual -Σ_s ρ_s of `Quasineutrality` (`Poisson` adds λ²k²). J_sμ is the gyroaverage at magnetic
# moment μ, ρ_sμ(x) = √(2μ T_s m_s)/|q_s|; ω_* = (T_s/q_s) k_y (κn + κT(μ + s²/2 - 3/2)) with κ = +d ln/dx
# (as everywhere in KineticDR), s = v∥/v_ts. Both integrals are exact: the v∥ average is two Z functions
# (any Im ω), and W is affine in μ, so the μ integral is Weber's second integral (`weber`). No magnetic
# drift (branch `radial-geometry`). N_s, T_s of a kinetic species are guiding-centre profiles, so its
# adiabatic term is a_s = q_s² ∫F_s d³v/T_s at the particle position (`gc_average`); a `Boltzmann` species
# has W = 0 and a_s = q_s² N_s/T_s at the node. Units as in the local model (Ω_i, ρ_i), φ in T_i/e.

# A species at radius x: a `Species` is uniform, anything else is a profile x -> `Species`.
species_at(s::Species, x) = s
species_at(s, x) = s(x)

"""
    (model::Model)(x) -> Model

The local (homogeneous) model at radius `x`: every profile species x -> `Species` evaluated there.
`dispersion(model(x), ω, k)` is the radial operator at uniform profiles on its k_x = 0 mode.
"""
(m::Model)(x) = Model(map(s -> species_at(s, x), m.species), m.field)

# A local solver handed a model with profiles
charge_response(s::Function, ω, k::Wavevector) = _profile_error()
susceptibility(s::Function, ω, k::Wavevector) = _profile_error()
_profile_error() = throw(ArgumentError("the model has radial profiles x -> Species: evaluate it at a radius, model(x), " *
                                       "or solve the radial problem (eigenmode)"))

"d ln f/dx of a profile, by central differences."
log_gradient(f; h = 1e-4) = x -> (log(f(x + h)) - log(f(x - h))) / (2h)

"""
    species_profile(; N = x -> 1.0, T = x -> 1.0, q = 1, m = 1, response = GordeyevSeries()) -> x -> Species

A species with profiles N(x), T(x) and gradients κn = d ln N/dx, κT = d ln T/dx derived from them
([`log_gradient`](@ref)). `response` is a response or a function x -> response.
"""
function species_profile(; N = x -> 1.0, T = x -> 1.0, q = 1.0, m = 1.0, response = GordeyevSeries())
    κn, κT = log_gradient(N), log_gradient(T)
    r = response isa AbstractResponse ? (x -> response) : response
    return x -> Species(; q, m, T = T(x), N = N(x), κn = κn(x), κT = κT(x), response = r(x))
end

# Guiding-centre profile f(species) seen at the particle position x: ∫dd g(d; ρ²(x + d)) f(s(x + d)), g the
# Gaussian of variance ρ² = T m/q² (squared thermal gyroradius, d = v_y m/q). Cut at 12 ρ of the largest ρ
# within ±20 (profiles may grow without bound); needs the profile there, beyond the walls.
function gc_average(f, sp, x)
    ρ2(s) = s.T * s.m / s.q^2
    dmax = 12 * sqrt(maximum(ρ2(species_at(sp, x + d)) for d in -20:20))
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
δ = (b - a)/N, so N nodes on [a, b)); `radial_grid` builds the one-axis grid.
- `bc = :dirichlet`: φ = 0 on the walls a and b; unknowns on the N - 1 nodes after a, sine basis
  sin(π m (x - a)/L). This is the kinetic wall of the bslLD runs, whose eigenmodes vanish there.
- `bc = :mirror`: unknowns at the cell centres x_j + δ/2 (N of them), cosine basis, even about both
  walls (bslLD's `Mirror()` halo). Admits wall-localised modes the simulation does not have.
Eigenvectors are returned as `ScalarField`s with one value per grid node (zero at the wall node for Dirichlet; cell
centres for mirror).
"""
struct RadialGrid{G}
    grid::G
    bc::Symbol
    x::Vector{Float64}          # nodes carrying unknowns
    k::Vector{Float64}          # basis wavenumbers π m / L
    C::Matrix{Float64}          # C[j, m] = basis_m(x_j)
    Cinv::Matrix{Float64}
end

function RadialGrid(grid; bc = :dirichlet)
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
    else
        throw(ArgumentError("bc must be :mirror or :dirichlet, got $bc"))
    end
    return RadialGrid(grid, bc, x, k, C, inv(C))
end

radial_grid(a, b, N; kw...) = RadialGrid(Grid([float(a)], [float(b)], [Int(N)], 1); kw...)

"φ on the unknown nodes as a `ScalarField` over the grid axis."
function radial_field(g::RadialGrid, φ::AbstractVector)
    return ScalarField(g.bc === :mirror ? Vector(φ) : vcat(zero(eltype(φ)), φ))
end

# W = ⟨(ω - ω_*)/(ω - k_z v∥)⟩ over v∥ for local species `s` at one μ: with s = v∥/v_t ~ N(0, 1),
# ⟨1/(ω - b s)⟩ = -a Z(ζ), ⟨s²/(ω - b s)⟩ = -2a ζ (1 + ζZ(ζ)), b = |k_z| v_t, a = 1/(√2 b), ζ = aω.
function vpar_response(k::Wavevector, s::Species, ω, mu)
    dia = s.T / s.q * k.ky
    n0 = ω - dia * (s.κn + s.κT * (mu - 1.5))
    n2 = -dia * s.κT / 2
    b = abs(k.kz) * thermal_speed(s)
    b == 0 && return (n0 + n2) / ω
    a = 1 / (sqrt(2.0) * b)
    ζ = a * ω
    return -a * (n0 * plasma_Z(ζ) + 2n2 * ζ * one_plus_zZ(ζ))
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

# Kinetic species: the μ integral in closed form. With G[j, l] = C[j, l] J0(k⊥_l ρ_jμ),
#   ∫dμ e^{-μ} G† diag(w (r0 + r1 μ)) G = M,  M[l, l'] = Σ_j C[j, l] w_j (r0_j W0[j, l, l'] + r1_j W1[j, l, l']) C[j, l'],
# W0, W1 = weber(0, k⊥_l ρ_j, k⊥_l' ρ_j), ρ_j = √(T m)/|q| at node j (ω-independent tables).
struct WeberOperator{S}
    ss::Vector{S}
    W0::Array{Float64,3}       # [j, l, l']
    W1::Array{Float64,3}
    u0::Matrix{ComplexF64}
    u1::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
end

function WeberOperator(g::RadialGrid, k::Wavevector, ss)
    N = length(g.x)
    kp = [hypot(q, k.ky) for q in g.k]
    W0, W1 = zeros(N, N, N), zeros(N, N, N)
    for j in 1:N
        ρ = sqrt(ss[j].T * ss[j].m) / abs(ss[j].q)
        for l in 1:N, l2 in l:N
            W0[j, l, l2], W1[j, l, l2] = weber(0, kp[l] * ρ, kp[l2] * ρ)
            W0[j, l2, l], W1[j, l2, l] = W0[j, l, l2], W1[j, l, l2]
        end
    end
    return WeberOperator(ss, W0, W1, zeros(ComplexF64, N, N), zeros(ComplexF64, N, N), zeros(ComplexF64, N, N))
end

struct RadialWorkspace
    k::Wavevector{Float64}
    kinetic::Vector{WeberOperator}
    diag::Vector{Float64}    # adiabatic terms Σ_s a_s at the nodes
    fieldop::Matrix{Float64} # field terms (Poisson λ²k²), zero for quasineutrality
end

field_operator(::Quasineutrality, g::RadialGrid, k::Wavevector) = zeros(length(g.x), length(g.x))
field_operator(f::Poisson, g::RadialGrid, k::Wavevector) =
    f.lambda2 .* (g.C * Diagonal(g.k .^ 2 .+ k.ky^2 .+ k.kz^2) * g.Cinv)
field_operator(f::AbstractFieldEquation, g::RadialGrid, k::Wavevector) = throw(ArgumentError(
    "$(nameof(typeof(f))) is not implemented for the radial problem (Quasineutrality, Poisson)"))

function RadialWorkspace(g::RadialGrid, m::Model, k::Wavevector)
    k.kx == 0 || throw(ArgumentError("radial problem: k_x comes from the basis, pass kx = 0 (got $(k.kx))"))
    fieldop = field_operator(m.field, g, k)
    N = length(g.x)
    diag = zeros(N)
    kinetic = WeberOperator[]
    for sp in m.species
        ss = [species_at(sp, x) for x in g.x]
        all(s -> s.response isa Boltzmann, ss) || all(s -> s.response isa Gyrokinetic, ss) || throw(ArgumentError(
            "radial kernels: Gyrokinetic or Boltzmann at every x, got $(unique(nameof.(typeof.(getfield.(ss, :response)))))"))
        diag .+= [adiabatic_term(sp, x) for x in g.x]
        ss[1].response isa Gyrokinetic && push!(kinetic, WeberOperator(g, k, ss))
    end
    return RadialWorkspace(k, kinetic, diag, fieldop)
end

"""
    operator(g, m, k, ω[, ws]) -> A::Matrix{ComplexF64}

Field operator on the unknown nodes; `A φ = 0` for an eigenmode. `k` gives k_y and k_z (k_x = 0:
it comes from the basis).
"""
function operator(g::RadialGrid, m::Model, k::Wavevector, ω::Number, ws = RadialWorkspace(g, m, k))
    A = complex(ws.fieldop)
    for K in ws.kinetic
        A .-= kinetic_operator!(K, g, ws.k, ω)
    end
    for j in eachindex(g.x)
        A[j, j] += ws.diag[j]
    end
    return A
end

# ∫dμ e^{-μ} J_μ† diag(w W_μ) J_μ = Cinv' M Cinv (see `WeberOperator`)
function kinetic_operator!(K::WeberOperator, g::RadialGrid, k::Wavevector, ω)
    N = length(g.x)
    C = g.C
    for (j, s) in enumerate(K.ss)
        w = s.q^2 * s.N / s.T
        r0 = vpar_response(k, s, ω, 0.0)
        r1 = vpar_response(k, s, ω, 1.0) - r0
        for l in 1:N
            K.u0[j, l] = C[j, l] * w * r0
            K.u1[j, l] = C[j, l] * w * r1
        end
    end
    @inbounds for l2 in 1:N, l in 1:N
        acc = zero(ComplexF64)
        @simd for j in 1:N
            acc += (K.u0[j, l] * K.W0[j, l, l2] + K.u1[j, l] * K.W1[j, l, l2]) * C[j, l2]
        end
        K.M[l, l2] = acc
    end
    return g.Cinv' * K.M * g.Cinv
end

# eigenvalue of smallest modulus and its eigenvector
function smallest(A)
    E = eigen(A)
    i = argmin(abs.(E.values))
    return E.values[i], E.vectors[:, i]
end

"""
    eigenmode(g, m, k, ω0; tol = 1e-10, ltol = 1e-8, maxit = 60) -> (; ω, φ, λ, converged)

Secant iteration on λ_min(A(ω)) = 0 from the guess `ω0`, tracking the eigenvalue closest to zero.
`converged` needs both the step below `tol |ω|` and |λ| ≤ `ltol` ‖A‖₁ (a jump between eigenvalue
branches can stall the step away from a root). `φ` is a `ScalarField`, normalised to max |φ| = 1
and real at the maximum.
"""
function eigenmode(g::RadialGrid, m::Model, k::Wavevector, ω0::Number; tol = 1e-10, ltol = 1e-8, maxit = 60,
                   ws = RadialWorkspace(g, m, k))
    isfinite(ω0) || throw(ArgumentError("eigenmode: the guess ω0 = $ω0 is not finite"))
    λ(ω) = first(smallest(operator(g, m, k, ω, ws)))
    a, b = complex(ω0), complex(ω0) * (1 + 1e-3) + 1e-5im
    fa, fb = λ(a), λ(b)
    converged = false
    for _ in 1:maxit
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
    converged &= abs(λb) <= ltol * opnorm(A, 1)
    φ = φ ./ φ[argmax(abs.(φ))]
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
converged modes, most unstable first. A seed search: it can miss modes (see [`count_eigenvalues`](@ref));
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
    for r in results
        r.converged || continue
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
