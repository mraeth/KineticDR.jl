# Radially resolved gyrokinetic dispersion relation: for one k_y, the quasineutrality operator
# A(ω) acting on φ(x) for ions with arbitrary radial profiles and adiabatic electrons. An
# eigenmode is an ω at which A(ω) is singular.
#
# For each ω,
#
#   A(ω) φ = (n/T_e + n/T_i) φ - Σ_μ w_μ J_μ† [ (n/T_i) ⟨(ω - α ω_D - ω_*) / (ω - k∥ v∥ - ω_D)⟩_{v∥} ] J_μ φ
#
# J_μ is the gyroaverage at magnetic moment μ (Gauss–Laguerre), ρ_μ(x) = √(2 μ T_i(x)) / B(x),
# ω_* = (k_y T_i/B_drive) (κn + κT (μ + s²/2 - 3/2)) with κ = +d ln/dx (as everywhere in KineticDR),
# ω_D = -k_y T_i (c s² + c⊥ μ), s = v∥/√T_i. The v∥ average is `quadratic_average`: exact, valid for
# any Im ω. Units: Ω_i, ρ_i, v_ti at the reference point (B = T_i = n = 1), φ in T_ref/e.

"""
    RadialModel(; ky, kpar, n, Ti, Te, κn, κT, B, Bdrive, drift, drift_perp, alpha)

Profiles are functions of x. `B` sets the local gyroradius and, unless `Bdrive` is given, the E×B
speed of the gradient drive (bslLD's `add_kappaT!` uses B = 1: pass `Bdrive = x -> 1.0`).
`kpar(x)` is k∥; `drift(x)` is c(x) in ω_D (e.g. `x -> 1/R0`, or `x -> 1/(B(x)(Rc + x))`);
`drift_perp(x)` is a separate ∇B coefficient c⊥ (default `drift`). `κn`, `κT` are d ln n/dx and
d ln T_i/dx; [`log_gradient`](@ref) derives them from profiles. `alpha` as in [`GyrokineticDrift`](@ref).
"""
Base.@kwdef struct RadialModel{F1,F2,F3,F4,F5,F6,F7,F8,F9,F10}
    ky::Float64
    kpar::F1 = x -> 0.0
    n::F2 = x -> 1.0
    Ti::F3 = x -> 1.0
    Te::F4 = x -> 1.0
    κn::F5 = x -> 0.0
    κT::F6 = x -> 0.0
    B::F7 = x -> 1.0
    drift::F8 = x -> 0.0
    Bdrive::F9 = B
    drift_perp::F10 = drift
    alpha::Float64 = 0.0
end

"Reduced magnetic shear: k∥ = k_y x / L_s + k_z, uniform gradients."
sheared_slab(; ky, Ls, kz = 0.0, κT = 0.0, κn = 0.0, kw...) =
    RadialModel(; ky, kpar = x -> ky * x / Ls + kz, κT = x -> κT, κn = x -> κn, kw...)

"d ln f/dx of a profile, by central differences."
log_gradient(f; h = 1e-4) = x -> (log(f(x + h)) - log(f(x - h))) / (2h)

"""
    local_model(m::RadialModel, x) -> (model, k)

The local (homogeneous) `Model` and `Wavevector` at radius `x`, so that `dispersion(model, ω, k)`
is the radial operator at uniform profiles. Needs B = B_drive = 1 (KineticDR has no B(x)).
"""
function local_model(m::RadialModel, x)
    (m.B(x) == 1 && m.Bdrive(x) == 1) || throw(ArgumentError("local_model needs B(x) = Bdrive(x) = 1"))
    c, cp = m.drift(x), m.drift_perp(x)
    response = c == 0 && cp == 0 ? Gyrokinetic() : GyrokineticDrift(; c, cperp = cp, alpha = m.alpha)
    ion = Species(; T = m.Ti(x), N = m.n(x), κn = m.κn(x), κT = m.κT(x), response)
    electron = Species(; q = -1, T = m.Te(x), N = m.n(x), response = Boltzmann())
    return Model((ion, electron), Quasineutrality()), Wavevector(ky = m.ky, kz = m.kpar(x))
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
`nmu` Gauss–Laguerre nodes in μ (converge it when ω_D ~ ω, see [`GyrokineticDrift`](@ref)). Eigenvectors are returned as `ScalarField`s with one value per grid node
(zero at the wall node for Dirichlet; cell centres for mirror).
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

# Gyroaverage J_μ (nodes -> guiding centres) and its adjoint in the quadrature inner product.
function gyroaverage(g::RadialGrid, m::RadialModel, mu)
    N = length(g.x)
    G = similar(g.C)
    for j in 1:N
        ρ = sqrt(2 * mu * m.Ti(g.x[j])) / m.B(g.x[j])
        for i in 1:N
            G[j, i] = g.C[j, i] * besselj0(sqrt(g.k[i]^2 + m.ky^2) * ρ)
        end
    end
    J = G * g.Cinv
    return J, J'
end

# ⟨(ω - α ω_D - ω_*)/(ω - k∥ v∥ - ω_D)⟩ over v∥ at one μ and radius X
function vpar_response(m::RadialModel, X, ω, mu)
    T = m.Ti(X)
    dia = m.ky * T / m.Bdrive(X)
    c, cp = m.drift(X), m.drift_perp(X)
    a = m.ky * T * c
    n0 = ω - dia * (m.κn(X) + m.κT(X) * (mu - 1.5)) + m.alpha * m.ky * T * cp * mu
    n2 = -dia * m.κT(X) / 2 + m.alpha * a
    return quadratic_average(n0, n2, a, abs(m.kpar(X)) * sqrt(T), ω + m.ky * T * cp * mu)
end

struct RadialWorkspace
    J::Vector{Matrix{Float64}}
    Jt::Vector{Matrix{Float64}}
    re::Matrix{Float64}      # real/imaginary parts of the accumulating operator (J real: two real gemms)
    im::Matrix{Float64}
    t1::Matrix{Float64}
end

function RadialWorkspace(g::RadialGrid, m::RadialModel)
    pairs = [gyroaverage(g, m, mu) for mu in g.mu]
    N = length(g.x)
    return RadialWorkspace(first.(pairs), last.(pairs), zeros(N, N), zeros(N, N), zeros(N, N))
end

"""
    operator(g, m, ω[, ws]) -> A::Matrix{ComplexF64}

Quasineutrality operator on the unknown nodes; `A φ = 0` for an eigenmode.
"""
function operator(g::RadialGrid, m::RadialModel, ω::Number, ws = RadialWorkspace(g, m))
    X = g.x
    fill!(ws.re, 0)
    fill!(ws.im, 0)
    for (q, mu) in enumerate(g.mu)
        d = [g.wmu[q] * m.n(x) / m.Ti(x) * vpar_response(m, x, ω, mu) for x in X]
        ws.t1 .= real.(d) .* ws.J[q]
        mul!(ws.re, ws.Jt[q], ws.t1, -1, 1)
        ws.t1 .= Base.imag.(d) .* ws.J[q]
        mul!(ws.im, ws.Jt[q], ws.t1, -1, 1)
    end
    A = complex.(ws.re, ws.im)
    for (j, x) in enumerate(X)
        A[j, j] += m.n(x) / m.Te(x) + m.n(x) / m.Ti(x)
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
    eigenmode(g, m, ω0; tol = 1e-10, maxit = 60) -> (; ω, φ, λ, converged)

Secant iteration on λ_min(A(ω)) = 0 from the guess `ω0`, tracking the eigenvalue closest to zero.
`φ` is a `ScalarField`, normalised to max |φ| = 1 and real at the maximum.
"""
function eigenmode(g::RadialGrid, m::RadialModel, ω0::Number; tol = 1e-10, maxit = 60,
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
    λb, φ = smallest(operator(g, m, b, ws))
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
    find_eigenmodes(g, m; re, im, starts = []) -> Vector of `eigenmode` results

Evaluates |λ_min(A(ω))| on the rectangle `re × im`, refines every local minimum (plus the guesses
`starts`, e.g. [`local_roots`](@ref) along x) with [`eigenmode`](@ref), and returns the distinct
converged modes, most unstable first. A seed search: it can miss modes (see [`count_eigenvalues`](@ref)).
Nearly degenerate modes are common in sheared geometry, so seed it generously.
"""
function find_eigenmodes(g::RadialGrid, m::RadialModel; re, im, starts = ComplexF64[], kw...)
    ws = RadialWorkspace(g, m)
    cand = [complex(a, b) for a in re, b in im]
    L = [abs(first(smallest(operator(g, m, ω, ws)))) for ω in cand]
    na, nb = size(L)
    starts = vcat(ComplexF64.(filter(isfinite, starts)),
                  [cand[i, j] for i in 1:na, j in 1:nb if
                   all(L[i, j] <= L[i2, j2] for i2 in max(1, i - 1):min(na, i + 1),
                       j2 in max(1, j - 1):min(nb, j + 1))])
    modes = []
    for ω0 in starts
        r = eigenmode(g, m, ω0; ws, kw...)
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
