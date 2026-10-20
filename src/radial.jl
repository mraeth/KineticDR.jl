# Radially resolved dispersion relation: for one k_y, the field operator A(ω) acting on φ(x), built
# from the same `Species` and field equations as the local `Model`, with radial profiles. An
# eigenmode is an ω at which A(ω) is singular.
#
# Each species contributes its charge response q_s δn_s/φ = (q_s² N_s/T_s)(W_s - 1) as an operator,
#
#   A(ω) = Σ_s diag(q_s² N_s/T_s) - Σ_{s kinetic} Σ_μ w_μ J_sμ† diag[(q_s² N_s/T_s) W_sμ(x, ω)] J_sμ  (+ field terms)
#
#   W_sμ = ⟨(ω - α ω_D - ω_*) / (ω - k∥ v∥ - ω_D)⟩_{v∥},
#
# the residual -Σ_s ρ_s of `Quasineutrality` (`Poisson` adds λ²k²). J_sμ is the gyroaverage at magnetic
# moment μ (Gauss–Laguerre), ρ_sμ(x) = √(2μ T_s m_s)/(|q_s| B(x)); ω_* = (T_s/q_s)(k_y/B_drive)(κn + κT(μ +
# s²/2 - 3/2)) with κ = +d ln/dx (as everywhere in KineticDR); ω_D = -k_y (T_s/q_s)(c s² + c⊥ μ), s = v∥/v_ts.
# The v∥ average is `quadratic_average`: exact for any Im ω; with drift the μ quadrature is not, below
# the real axis (see `GyrokineticDrift`). `Boltzmann` species have W = 0. Units as in the local model
# (Ω_i, ρ_i at B = 1), φ in T_i/e.

"""
    RadialModel(species, field = Quasineutrality(); ky, kpar = x -> 0.0, B = x -> 1.0, Bdrive = B)
    RadialModel(; ky, kpar, n, Ti, Te, κn, κT, B, Bdrive, drift, drift_perp, alpha)

A radially resolved model: `species` is a tuple whose entries are `Species` (uniform) or functions
x -> `Species` (profiles of N, T, κn, κT and the response's lab parameters, e.g. the drift c(x) of a
`GyrokineticDrift`), closed by `field` (`Quasineutrality` or `Poisson`; electromagnetic fields are not
implemented radially). Responses with a radial kernel: `Gyrokinetic`, `GyrokineticDrift`, `Boltzmann`.
`kpar(x)` is k∥. `B` sets the local gyroradius and, unless `Bdrive` is given, the E×B speed of the
gradient drive (bslLD's `add_kappaT!` uses B = 1: pass `Bdrive = x -> 1.0`).

The keyword form is the ion + Boltzmann-electron model: profiles n(x), Ti(x), Te(x), κn(x), κT(x)
(d ln n/dx, d ln T_i/dx; [`log_gradient`](@ref) derives them), and `drift(x)` = c(x) in ω_D (e.g.
`x -> 1/R0`) with ∇B coefficient `drift_perp(x)` (default `drift`) and `alpha` as in [`GyrokineticDrift`](@ref).
"""
struct RadialModel{S<:Tuple,F<:AbstractFieldEquation,K,B1,B2}
    species::S
    field::F
    ky::Float64
    kpar::K
    B::B1
    Bdrive::B2
end

function RadialModel(species::Tuple, field::AbstractFieldEquation = Quasineutrality();
                     ky, kpar = x -> 0.0, B = x -> 1.0, Bdrive = B)
    field isa AbstractEMField &&
        throw(ArgumentError("electromagnetic fields are not implemented for the radial problem"))
    sp = map(s -> s isa Species ? (x -> s) : s, species)
    return RadialModel(sp, field, float(ky), kpar, B, Bdrive)
end

function RadialModel(; ky, kpar = x -> 0.0, n = x -> 1.0, Ti = x -> 1.0, Te = x -> 1.0, κn = x -> 0.0,
                     κT = x -> 0.0, B = x -> 1.0, Bdrive = B, drift = x -> 0.0, drift_perp = drift, alpha = 0.0)
    function ion(x)
        c, cp = drift(x), drift_perp(x)
        response = c == 0 && cp == 0 ? Gyrokinetic() : GyrokineticDrift(; c, cperp = cp, alpha)
        return Species(; T = Ti(x), N = n(x), κn = κn(x), κT = κT(x), response)
    end
    electron(x) = Species(; q = -1, T = Te(x), N = n(x), response = Boltzmann())
    return RadialModel((ion, electron), Quasineutrality(); ky, kpar, B, Bdrive)
end

"Reduced magnetic shear: k∥ = k_y x / L_s + k_z, uniform gradients."
sheared_slab(; ky, Ls, kz = 0.0, κT = 0.0, κn = 0.0, kw...) =
    RadialModel(; ky, kpar = x -> ky * x / Ls + kz, κT = x -> κT, κn = x -> κn, kw...)

"d ln f/dx of a profile, by central differences."
log_gradient(f; h = 1e-4) = x -> (log(f(x + h)) - log(f(x - h))) / (2h)

"""
    local_model(m::RadialModel, x) -> (model, k)

The local (homogeneous) `Model` and `Wavevector` at radius `x`, so that `dispersion(model, ω, k)`
is the radial operator at uniform profiles. Needs B = B_drive = 1 (the local model has B = 1).
"""
function local_model(m::RadialModel, x)
    (m.B(x) == 1 && m.Bdrive(x) == 1) || throw(ArgumentError("local_model needs B(x) = Bdrive(x) = 1"))
    return Model(map(s -> s(x), m.species), m.field), Wavevector(ky = m.ky, kz = m.kpar(x))
end

"""
    RadialGrid(grid::PlasmaCore.Grid; bc = :dirichlet, nmu = 64)
    radial_grid(a, b, N; kw...)

Radial discretisation on the first spatial axis of a PlasmaCore `Grid` (nodes a + (j-1)δ,
δ = (b - a)/N, so N nodes on [a, b)); `radial_grid` builds the one-axis grid.
- `bc = :dirichlet`: φ = 0 on the walls a and b; unknowns on the N - 1 nodes after a, sine basis
  sin(π m (x - a)/L). This is the kinetic wall of the bslLD runs, whose eigenmodes vanish there.
- `bc = :mirror`: unknowns at the cell centres x_j + δ/2 (N of them), cosine basis, even about both
  walls (bslLD's `Mirror()` halo). Admits wall-localised modes the simulation does not have.
`nmu` Gauss–Laguerre nodes in μ for every kinetic species (the radial problem uses these, not a
response's own `nmu`; converge it when ω_D ~ ω, see [`GyrokineticDrift`](@ref)). Eigenvectors are
returned as `ScalarField`s with one value per grid node (zero at the wall node for Dirichlet; cell
centres for mirror).
"""
struct RadialGrid{G}
    grid::G
    bc::Symbol
    x::Vector{Float64}          # nodes carrying unknowns
    k::Vector{Float64}          # basis wavenumbers π m / L
    C::Matrix{Float64}          # C[j, m] = basis_m(x_j)
    Cinv::Matrix{Float64}
    w::Vector{Float64}          # quadrature weights (uniform: the bases are orthogonal on the nodes)
    mu::Vector{Float64}
    wmu::Vector{Float64}
end

function RadialGrid(grid; bc = :dirichlet, nmu = 64)
    ax = grid.xaxes[1]
    N, h, a = length(ax), step(ax), first(ax)
    L = N * h
    mu, wmu = gauss_laguerre(nmu)
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
    return RadialGrid(grid, bc, x, k, C, inv(C), fill(h, length(x)), mu, wmu)
end

radial_grid(a, b, N; kw...) = RadialGrid(Grid([float(a)], [float(b)], [Int(N)], 1); kw...)

"φ on the unknown nodes as a `ScalarField` over the grid axis."
function radial_field(g::RadialGrid, φ::AbstractVector)
    return ScalarField(g.bc === :mirror ? Vector(φ) : vcat(zero(eltype(φ)), φ))
end

# Drift coefficients (c, c⊥, α) of a gyrokinetic response, lab units.
drift_coefficients(::Gyrokinetic) = (0.0, 0.0, 0.0)
drift_coefficients(r::GyrokineticDrift) = (r.c, r.cperp, r.alpha)

# Gyroaverage J_μ (nodes -> guiding centres) of species `ss` (local species at the nodes) and its adjoint.
function gyroaverage(g::RadialGrid, m::RadialModel, ss, mu)
    N = length(g.x)
    G = similar(g.C)
    for j in 1:N
        s = ss[j]
        ρ = sqrt(2 * mu * s.T * s.m) / (abs(s.q) * m.B(g.x[j]))
        for i in 1:N
            G[j, i] = g.C[j, i] * besselj0(sqrt(g.k[i]^2 + m.ky^2) * ρ)
        end
    end
    J = G * g.Cinv
    return J, Matrix(J')
end

# W = ⟨(ω - α ω_D - ω_*)/(ω - k∥ v∥ - ω_D)⟩ over v∥ for local species `s` at one μ and radius X
function vpar_response(m::RadialModel, s::Species, X, ω, mu)
    c, cp, alpha = drift_coefficients(s.response)
    τ = s.T / s.q
    dia = τ * m.ky / m.Bdrive(X)
    a = m.ky * τ * c
    n0 = ω - dia * (s.κn + s.κT * (mu - 1.5)) + alpha * m.ky * τ * cp * mu
    n2 = -dia * s.κT / 2 + alpha * a
    return quadratic_average(n0, n2, a, abs(m.kpar(X)) * thermal_speed(s), ω + m.ky * τ * cp * mu)
end

# Per kinetic species: local species at the nodes and the gyroaverages at each μ.
struct KineticOperator{S}
    ss::Vector{S}
    J::Vector{Matrix{Float64}}
    Jt::Vector{Matrix{Float64}}
end

struct RadialWorkspace
    kinetic::Vector{KineticOperator}
    diag::Vector{Float64}    # Σ_s q_s² N_s/T_s at the nodes
    fieldop::Matrix{Float64} # field terms (Poisson λ²k²), zero for quasineutrality
    re::Matrix{Float64}      # real/imaginary parts of the accumulating operator (J real: two real gemms)
    im::Matrix{Float64}
    t1::Matrix{Float64}
end

field_operator(::Quasineutrality, g::RadialGrid, m::RadialModel) = zeros(length(g.x), length(g.x))
function field_operator(f::Poisson, g::RadialGrid, m::RadialModel)
    K = g.C * Diagonal(g.k .^ 2 .+ m.ky^2) * g.Cinv
    return f.lambda2 .* (K + Diagonal(m.kpar.(g.x) .^ 2))
end

function RadialWorkspace(g::RadialGrid, m::RadialModel)
    N = length(g.x)
    diag = zeros(N)
    kinetic = KineticOperator[]
    for sp in m.species
        ss = [sp(x) for x in g.x]
        diag .+= [s.q^2 * s.N / s.T for s in ss]
        all(s -> s.response isa Boltzmann, ss) && continue
        for s in ss
            s.response isa Union{Gyrokinetic,GyrokineticDrift} || throw(ArgumentError(
                "$(nameof(typeof(s.response))) has no radial kernel (Gyrokinetic, GyrokineticDrift, Boltzmann at every x)"))
        end
        pairs = [gyroaverage(g, m, ss, mu) for mu in g.mu]
        push!(kinetic, KineticOperator(ss, first.(pairs), last.(pairs)))
    end
    return RadialWorkspace(kinetic, diag, field_operator(m.field, g, m), zeros(N, N), zeros(N, N), zeros(N, N))
end

"""
    operator(g, m, ω[, ws]) -> A::Matrix{ComplexF64}

Field operator on the unknown nodes; `A φ = 0` for an eigenmode.
"""
function operator(g::RadialGrid, m::RadialModel, ω::Number, ws = RadialWorkspace(g, m))
    X = g.x
    ws.re .= ws.fieldop
    fill!(ws.im, 0)
    for K in ws.kinetic, (q, mu) in enumerate(g.mu)
        d = [g.wmu[q] * s.q^2 * s.N / s.T * vpar_response(m, s, x, ω, mu) for (s, x) in zip(K.ss, X)]
        ws.t1 .= real.(d) .* K.J[q]
        mul!(ws.re, K.Jt[q], ws.t1, -1, 1)
        ws.t1 .= Base.imag.(d) .* K.J[q]
        mul!(ws.im, K.Jt[q], ws.t1, -1, 1)
    end
    A = complex.(ws.re, ws.im)
    for j in eachindex(X)
        A[j, j] += ws.diag[j]
    end
    return A
end

# eigenvalue of smallest modulus and its eigenvector
function smallest(A)
    E = eigen(A)
    i = argmin(abs.(E.values))
    return E.values[i], E.vectors[:, i]
end

"""
    eigenmode(g, m, ω0; tol = 1e-10, ltol = 1e-8, maxit = 60) -> (; ω, φ, λ, converged)

Secant iteration on λ_min(A(ω)) = 0 from the guess `ω0`, tracking the eigenvalue closest to zero.
`converged` needs both the step below `tol |ω|` and |λ| ≤ `ltol` ‖A‖₁ (a jump between eigenvalue
branches can stall the step away from a root). `φ` is a `ScalarField`, normalised to max |φ| = 1
and real at the maximum. With drift, a root at Im ω < 0 needs an `nmu` convergence check (see
[`GyrokineticDrift`](@ref)).
"""
function eigenmode(g::RadialGrid, m::RadialModel, ω0::Number; tol = 1e-10, ltol = 1e-8, maxit = 60,
                   ws = RadialWorkspace(g, m))
    λ(ω) = first(smallest(operator(g, m, ω, ws)))
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
    A = operator(g, m, b, ws)
    λb, φ = smallest(A)
    converged &= abs(λb) <= ltol * opnorm(A, 1)
    φ = φ ./ φ[argmax(abs.(φ))]
    return (; ω = b, φ = radial_field(g, φ), λ = λb, converged)
end

"""
    local_roots(m, xs; re, imag) -> Vector{ComplexF64}

Most unstable local root (`find_modes` on `local_model`) at each radius, for seeding
[`find_eigenmodes`](@ref). `NaN` where there is none.
"""
function local_roots(m::RadialModel, xs; re, imag)
    return map(xs) do x
        roots = find_modes(local_model(m, x)...; re, imag, imin = 0.0)
        isempty(roots) ? complex(NaN, NaN) : roots[argmax(Base.imag.(roots))]
    end
end

"""
    find_eigenmodes(g, m; re, imag, starts = []) -> Vector of `eigenmode` results

Evaluates |λ_min(A(ω))| on the rectangle `re × imag`, refines every local minimum (plus the guesses
`starts`, e.g. [`local_roots`](@ref) along x) with [`eigenmode`](@ref), and returns the distinct
converged modes, most unstable first. A seed search: it can miss modes (see [`count_eigenvalues`](@ref)).
Nearly degenerate modes are common in sheared geometry, so seed it generously.
"""
function find_eigenmodes(g::RadialGrid, m::RadialModel; re, imag, starts = ComplexF64[], kw...)
    ws = RadialWorkspace(g, m)
    cand = [complex(a, b) for a in re, b in imag]
    L = [abs(first(smallest(operator(g, m, ω, ws)))) for ω in cand]
    na, nb = size(L)
    starts = vcat(ComplexF64.(filter(isfinite, starts)),
                  [cand[i, j] for i in 1:na, j in 1:nb if
                   all(L[i, j] <= L[i2, j2] for i2 in max(1, i - 1):min(na, i + 1),
                       j2 in max(1, j - 1):min(nb, j + 1))])
    results = [eigenmode(g, m, ω0; ws, kw...) for ω0 in starts]
    modes = empty(results)
    for r in results
        r.converged || continue
        any(abs(r.ω - q.ω) < 1e-6 * abs(r.ω) for q in modes) && continue
        push!(modes, r)
    end
    return sort(modes; by = r -> -Base.imag(r.ω))
end

"Most unstable mode of [`find_eigenmodes`](@ref)."
find_eigenmode(g::RadialGrid, m::RadialModel; kw...) = first(find_eigenmodes(g, m; kw...))

"""
    count_eigenvalues(g, m, re0, re1, im0, im1) -> Int

Number of eigenmodes inside the rectangle: zeros of det A(ω), by the argument principle
(`count_roots`). Valid when det A has no branch cut inside, i.e. for ω_D = 0 everywhere or a
rectangle not touching the cut along Im ω < 0 under the drift resonance.
"""
function count_eigenvalues(g::RadialGrid, m::RadialModel, re0, re1, im0, im1; kw...)
    ws = RadialWorkspace(g, m)
    return count_roots(ω -> det(operator(g, m, ω, ws)), re0, re1, im0, im1; kw...)
end
