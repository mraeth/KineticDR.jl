# Radially resolved gyrokinetic dispersion relation: for one k_y, the quasineutrality operator
# A(ω) acting on φ(x) for ions with arbitrary radial profiles and adiabatic electrons. An
# eigenmode is an ω at which A(ω) is singular.
#
# For each ω,
#
#   A(ω) φ = (n/T_e + n/T_i) φ - Σ_μ w_μ J_μ† [ (n/T_i) ⟨(ω - α ω_D - ω_*) / (ω - k∥ v∥ - ω_D)⟩_{v∥} ] J_μ φ
#
# J_μ is the gyroaverage at magnetic moment μ (quadrature, or the μ integral in closed form for
# `quad = :analytic`), ρ_μ(x) = √(2 μ T_i(x)) / B(x),
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

`pmax = 0` is the gyrokinetic operator (J₀ only). `pmax > 0` adds the Larmor harmonics |p| ≤ pmax
(full-orbit ions, the Gordeyev series of the PRL): harmonic p has the gyroaverage J_p(k⊥ρ_μ) with the
orbit phase e^{-ip atan2(k_x, k_y)} and the resonant denominator ω - pB - k∥v∥; the `1` of
quasineutrality is not truncated. Needs `drift = 0`.

`adiabatic = :exact` (default) takes the adiabatic ion term and the electron density at the particle
position, which is bsl6d's convention (guiding-centre profiles, ion particle density in quasineutrality): ∫F_ε d³v = -∫dv_y (n/T_i)(x + v_y) g(v_y; T_i(x + v_y)) and, for the Boltzmann electrons, the
ion particle density n_p(x) = ∫dv_y n(x + v_y) g(v_y; T_i(x + v_y)), g the 1-D Maxwellian. With B = 1 the
operator is then exact up to the cut-off |p| ≤ pmax. `adiabatic = :local` uses n(x)/T_i(x) and n(x) at the node
instead (error O((κρ)²), ~15% on steep profiles).
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
    pmax::Int = 0
    adiabatic::Symbol = :exact
end

"Reduced magnetic shear: k∥ = k_y x / L_s + k_z, uniform gradients."
sheared_slab(; ky, Ls, kz = 0.0, κT = 0.0, κn = 0.0, kw...) =
    RadialModel(; ky, kpar = x -> ky * x / Ls + kz, κT = x -> κT, κn = x -> κn, kw...)

"d ln f/dx of a profile, by central differences."
log_gradient(f; h = 1e-4) = x -> (log(f(x + h)) - log(f(x - h))) / (2h)

"""
    local_model(m::RadialModel, x) -> (model, k)

The local (homogeneous) `Model` and `Wavevector` at radius `x`, so that `dispersion(model, ω, k)`
is the radial operator at uniform profiles. Needs B = B_drive = 1 (KineticDR has no B(x)). With
`adiabatic = :exact` the electron density is the ion particle density n_p(x) (the ion adiabatic term
stays n(x)/T_i(x)).
"""
function local_model(m::RadialModel, x)
    (m.B(x) == 1 && m.Bdrive(x) == 1) || throw(ArgumentError("local_model needs B(x) = Bdrive(x) = 1"))
    c, cp = m.drift(x), m.drift_perp(x)
    if m.pmax > 0
        (c == 0 && cp == 0) || throw(ArgumentError("pmax > 0 (full orbit) needs drift = 0"))
        response = GordeyevSeries(; pmax = m.pmax)
    else
        response = c == 0 && cp == 0 ? Gyrokinetic() : GyrokineticDrift(; c, cperp = cp, alpha = m.alpha)
    end
    ion = Species(; T = m.Ti(x), N = m.n(x), κn = m.κn(x), κT = m.κT(x), response)
    ne = m.adiabatic === :exact ? gc_average(m.n, m.Ti, x) : m.n(x)
    electron = Species(; q = -1, T = m.Te(x), N = ne, response = Boltzmann())
    return Model((ion, electron), Quasineutrality()), Wavevector(ky = m.ky, kz = m.kpar(x))
end

"""
    exponential_dispersion(ω, kx′, ky, kz; κn, T = 1, Te = 1) -> D

Exact local dispersion function of full-orbit ions with an exponential guiding-centre density
n ∝ e^{κn X} (κn = d ln n/dX, any size) and uniform T, Boltzmann electrons with the ion particle
density. The equilibrium is translation invariant, so an eigenmode is e^{κn x/2} e^{i kx′ x} with real kx′
(the shift k_x = kx′ + iκn/2 makes D even in kx′). Same normalisation and sign as
`dispersion(local_model(...)...)`, i.e. (δn_e - δn_i)/(n_p φ), to which it reduces for κn → 0. At O(κn),
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

"""
    RadialGrid(grid::PlasmaCore.Grid; bc = :dirichlet, nmu = 64)
    radial_grid(a, b, N; kw...)

Radial discretisation on the first spatial axis of a PlasmaCore `Grid` (nodes a + (j-1)δ,
δ = (b - a)/N, so N nodes on [a, b)); `radial_grid` builds the one-axis grid.
- `bc = :dirichlet`: φ = 0 on the walls a and b; unknowns on the N - 1 nodes after a, sine basis
  sin(π m (x - a)/L). This is the kinetic wall of the bslLD runs, whose eigenmodes vanish there.
- `bc = :periodic`: N unknowns at the nodes a + (j-1)δ, plane-wave basis exp(i k (x - a)), k = 2π m/L (needs an
  even N and periodic profiles). The only basis that carries the orbit phase of the Larmor harmonics p ≠ 0
  (`RadialModel.pmax > 0`): in a standing-wave basis (`:dirichlet`, `:mirror`) it leaves the basis and aliases.
- `bc = :mirror`: unknowns at the cell centres x_j + δ/2 (N of them), cosine basis, even about both
  walls (bslLD's `Mirror()` halo). Admits wall-localised modes the simulation does not have.
`quad = :analytic` does the μ integral in closed form (Weber's second integral, see [`weber`](@ref)): exact,
no `nmu`, and much cheaper per ω for large `pmax`; it needs `drift = 0` (the v∥ response is then affine in μ).
Otherwise `nmu` nodes in μ: `quad = :laguerre` (Gauss–Laguerre; converge it when ω_D ~ ω, see [`GyrokineticDrift`](@ref), and for the harmonics at large k⊥ρ) or `quad = :legendre` (Gauss–Legendre in v⊥ = √(2μ), much better for oscillatory J_p²). Eigenvectors are returned as `ScalarField`s with one value per grid node
(zero at the wall node for Dirichlet; cell centres for mirror; the nodes for periodic).
"""
# ∫₀^∞ e^{-μ} f(μ) dμ nodes and weights: Gauss–Laguerre in μ, or Gauss–Legendre in u = √(2μ) on [0, 7.5]
# (∫ e^{-u²/2} u f du; resolves the oscillation of J_p²(k√(2μ T)) at large k⊥ρ with far fewer nodes).
function perp_quadrature(n, quad)
    quad === :analytic && return Float64[], Float64[]
    quad === :laguerre && return gauss_laguerre(n)
    quad === :legendre || throw(ArgumentError("quad must be :analytic, :laguerre or :legendre, got $quad"))
    E = eigen(SymTridiagonal(zeros(n), [k / sqrt(4k^2 - 1) for k in 1:(n - 1)]))
    umax = 7.5
    u = umax / 2 .* (E.values .+ 1)
    return u .^ 2 ./ 2, umax / 2 .* E.vectors[1, :] .^ 2 .* 2 .* exp.(-u .^ 2 ./ 2) .* u
end

struct RadialGrid{G}
    grid::G
    bc::Symbol
    x::Vector{Float64}          # nodes carrying unknowns
    k::Vector{Float64}          # basis wavenumbers π m / L
    C::Matrix                   # C[j, m] = basis_m(x_j) (complex for :periodic)
    Cinv::Matrix
    w::Vector{Float64}          # quadrature weights (uniform: the bases are orthogonal on the nodes)
    mu::Vector{Float64}
    wmu::Vector{Float64}
    quad::Symbol
end

function RadialGrid(grid; bc = :dirichlet, nmu = 64, quad = :laguerre)
    ax = grid.xaxes[1]
    N, h, a = length(ax), step(ax), first(ax)
    L = N * h
    mu, wmu = perp_quadrature(nmu, quad)
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
        return RadialGrid(grid, bc, x, k, C, C' / N, fill(h, N), mu, wmu, quad)
    else
        throw(ArgumentError("bc must be :mirror, :dirichlet or :periodic, got $bc"))
    end
    return RadialGrid(grid, bc, x, k, C, inv(C), fill(h, length(x)), mu, wmu, quad)
end

radial_grid(a, b, N; kw...) = RadialGrid(Grid([float(a)], [float(b)], [Int(N)], 1); kw...)

"φ on the unknown nodes as a `ScalarField` over the grid axis."
function radial_field(g::RadialGrid, φ::AbstractVector)
    return ScalarField(g.bc === :dirichlet ? vcat(zero(eltype(φ)), φ) : Vector(φ))
end

# Gyroaverage J_{μ,p} (nodes -> guiding centres) and its adjoint in the quadrature inner product.
# Orbit of an ion in B = +z: ρ = ρ(-sin ϑ, cos ϑ), ϑ̇ = -Ω, so harmonic p of e^{i k·ρ} is i^p J_p(k⊥ρ) e^{-ip φ_k}
# with φ_k = atan2(k_x, k_y) (denominator ω - pΩ after relabelling p -> -p; the i^p cancels with the
# adjoint). A standing-wave basis function sin/cos(k (x - a)) is the pair k_x = ±k, φ_{-k} = -φ_k, which
# gives sin/cos(k (x - a) - p φ_k): real, so J stays a real matrix.
function gyroaverage(g::RadialGrid, m::RadialModel, mu, p::Integer = 0)
    N = length(g.x)
    a = first(g.grid.xaxes[1])
    trig = g.bc === :mirror ? cos : sin
    G = similar(g.C)
    p != 0 && g.bc !== :periodic && throw(ArgumentError("Larmor harmonics p ≠ 0 need bc = :periodic"))
    for j in 1:N
        ρ = sqrt(2 * mu * m.Ti(g.x[j])) / m.B(g.x[j])
        for i in 1:N
            z = sqrt(g.k[i]^2 + m.ky^2) * ρ
            G[j, i] = p == 0 ? g.C[j, i] * besselj0(z) :
                      g.bc === :periodic ? g.C[j, i] * cis(-p * atan(g.k[i], m.ky)) * besselj(p, z) :
                      trig(g.k[i] * (g.x[j] - a) - p * atan(g.k[i], m.ky)) * besselj(p, z)
        end
    end
    J = G * g.Cinv
    return J, J'
end

# ⟨(ω - α ω_D - ω_*)/(ω - pB - k∥ v∥ - ω_D)⟩ over v∥ at one μ and radius X, harmonic p
function vpar_response(m::RadialModel, X, ω, mu, p::Integer = 0)
    T = m.Ti(X)
    dia = m.ky * T / m.Bdrive(X)
    c, cp = m.drift(X), m.drift_perp(X)
    a = m.ky * T * c
    n0 = ω - dia * (m.κn(X) + m.κT(X) * (mu - 1.5)) + m.alpha * m.ky * T * cp * mu
    n2 = -dia * m.κT(X) / 2 + m.alpha * a
    return quadratic_average(n0, n2, a, abs(m.kpar(X)) * sqrt(T), ω - p * m.B(X) + m.ky * T * cp * mu)
end

"""
    weber(p, a, b) -> (W0, W1)

∫₀^∞ dμ e^{-μ} μ^k J_p(a√(2μ)) J_p(b√(2μ)) for k = 0, 1 (Weber's second exponential integral):
W0 = e^{-s} I_p(ab), W1 = e^{-s} [(1 - s) I_p(ab) + ab I_p'(ab)], s = (a² + b²)/2. For a = b,
W0 = Γ_p(a²). Valid for complex a, b (analytic continuation; the scaled Bessel function is
unscaled by e^{|Re ab|}, not e^{ab}).
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

# Is the drift zero at all nodes (needed by the harmonics and by quad = :analytic)?
nodrift(g::RadialGrid, m::RadialModel) = all(m.drift(x) == 0 && m.drift_perp(x) == 0 for x in g.x)

# 1-D Maxwellian average ∫dv g(v; T(x + v)) f(x + v): the velocity integral at fixed v_y of a gyrocentre
# Maxwellian F(x + v_y, ε) at the particle position x.
# Cut at 12 thermal speeds of the hottest point within ±20 (finite range: profiles may grow without bound).
function gc_average(f, T, x)
    vmax = 12 * sqrt(maximum(T(x + v) for v in -20:20))
    return quadgk(v -> f(x + v) * exp(-v^2 / (2T(x + v))) / sqrt(2π * T(x + v)), -vmax, 0, vmax; rtol = 1e-12)[1]
end

"Diagonal of A(ω): electron Boltzmann term plus the ion adiabatic term (`m.adiabatic`)."
function adiabatic_diagonal(g::RadialGrid, m::RadialModel)
    m.adiabatic === :local && return [m.n(x) / m.Te(x) + m.n(x) / m.Ti(x) for x in g.x]
    m.adiabatic === :exact || throw(ArgumentError("adiabatic must be :local or :exact, got $(m.adiabatic)"))
    return [gc_average(m.n, m.Ti, x) / m.Te(x) + gc_average(X -> m.n(X) / m.Ti(X), m.Ti, x) for x in g.x]
end

struct RadialWorkspace{T}
    J::Vector{Matrix{T}}
    Jt::Vector{Matrix{T}}
    re::Matrix{Float64}      # real J: real/imaginary parts of the accumulating operator (two real gemms)
    im::Matrix{Float64}
    t1::Matrix{T}
    acc::Matrix{ComplexF64}  # complex J: accumulator
    diag::Vector{Float64}
end

# quad = :analytic. A = diag - Cinv† M Cinv with
#   M[l, l'] = Σ_p e^{ip(θ_l - θ_l')} Σ_j conj(C[j, l]) C[j, l'] (n/T)_j [r0_pj W0_p(j, l, l') + r1_pj W1_p(j, l, l')],
# where the v∥ response at node j is r0 + r1 μ and W0, W1 = weber(p, a_l, a_l') with a_l = k⊥_l √T_j / B_j.
# W is even in p, so one table per |p|; it does not depend on ω.
struct WeberWorkspace
    W0::Vector{Array{Float64,3}}   # [j, l, l'], index |p| + 1
    W1::Vector{Array{Float64,3}}
    phase::Vector{Float64}         # θ_l = atan2(k_l, k_y)
    diag::Vector{Float64}
end

function WeberWorkspace(g::RadialGrid, m::RadialModel)
    nodrift(g, m) || throw(ArgumentError("quad = :analytic needs drift = 0"))
    N = length(g.x)
    kp = [hypot(k, m.ky) for k in g.k]
    uk = unique(kp)                                    # ±k share k⊥: Bessel values on the distinct ones only
    iu = [findfirst(==(q), uk) for q in kp]
    W0 = [zeros(N, N, N) for _ in 0:m.pmax]
    W1 = [zeros(N, N, N) for _ in 0:m.pmax]
    T0, T1 = zeros(length(uk), length(uk)), zeros(length(uk), length(uk))
    for p in 0:m.pmax, j in 1:N
        s = sqrt(m.Ti(g.x[j])) / m.B(g.x[j])
        for u in eachindex(uk), v in u:length(uk)
            T0[u, v], T1[u, v] = weber(p, uk[u] * s, uk[v] * s)
            T0[v, u], T1[v, u] = T0[u, v], T1[u, v]
        end
        for l2 in 1:N, l in 1:N
            W0[p + 1][j, l, l2] = T0[iu[l], iu[l2]]
            W1[p + 1][j, l, l2] = T1[iu[l], iu[l2]]
        end
    end
    return WeberWorkspace(W0, W1, [atan(k, m.ky) for k in g.k], adiabatic_diagonal(g, m))
end

function RadialWorkspace(g::RadialGrid, m::RadialModel)
    g.quad === :analytic && return WeberWorkspace(g, m)
    pairs = [gyroaverage(g, m, mu, p) for p in -m.pmax:m.pmax for mu in g.mu]   # index (p + pmax) * nmu + q
    N = length(g.x)
    J = [Matrix(first(q)) for q in pairs]
    Jt = [Matrix(last(q)) for q in pairs]
    T = eltype(first(J))
    return RadialWorkspace{T}(J, Jt, zeros(N, N), zeros(N, N), zeros(T, N, N), zeros(ComplexF64, N, N),
                              adiabatic_diagonal(g, m))
end

# ws += -J† diag(d) J
function accumulate!(ws::RadialWorkspace{Float64}, d, i)
    ws.t1 .= real.(d) .* ws.J[i]
    mul!(ws.re, ws.Jt[i], ws.t1, -1, 1)
    ws.t1 .= Base.imag.(d) .* ws.J[i]
    mul!(ws.im, ws.Jt[i], ws.t1, -1, 1)
end
function accumulate!(ws::RadialWorkspace{ComplexF64}, d, i)
    ws.t1 .= d .* ws.J[i]
    mul!(ws.acc, ws.Jt[i], ws.t1, 1, 1)
end

"""
    operator(g, m, ω[, ws]) -> A::Matrix{ComplexF64}

Quasineutrality operator on the unknown nodes; `A φ = 0` for an eigenmode.
"""
function operator(g::RadialGrid, m::RadialModel, ω::Number, ws = RadialWorkspace(g, m))
    X = g.x
    fill!(ws.re, 0)
    fill!(ws.im, 0)
    fill!(ws.acc, 0)
    nmu = length(g.mu)
    for p in -m.pmax:m.pmax, (q, mu) in enumerate(g.mu)
        i = (p + m.pmax) * nmu + q
        d = [g.wmu[q] * m.n(x) / m.Ti(x) * vpar_response(m, x, ω, mu, p) for x in X]
        accumulate!(ws, d, i)
    end
    A = eltype(ws.J[1]) <: Complex ? -ws.acc : complex.(ws.re, ws.im)
    for j in eachindex(X)
        A[j, j] += ws.diag[j]
    end
    return A
end

# M[l, l'] += e^{ip(θ_l - θ_l')} Σ_j (u0[j, l] W0[j, l, l'] + u1[j, l] W1[j, l, l']) C[j, l']
function weber_accumulate!(M, u0, u1, W0, W1, C, phase, p)
    N = size(M, 1)
    @inbounds for l2 in 1:N, l in 1:N
        acc = zero(ComplexF64)
        @simd for j in 1:N
            acc += (u0[j, l] * W0[j, l, l2] + u1[j, l] * W1[j, l, l2]) * C[j, l2]
        end
        M[l, l2] += p == 0 ? acc : cis(p * (phase[l] - phase[l2])) * acc
    end
    return M
end

function operator(g::RadialGrid, m::RadialModel, ω::Number, ws::WeberWorkspace)
    N = length(g.x)
    C = g.C
    M = zeros(ComplexF64, N, N)
    u0, u1 = zeros(ComplexF64, N, N), zeros(ComplexF64, N, N)
    for p in 0:m.pmax, sp in (p == 0 ? (0,) : (p, -p))
        # v∥ response r0 + r1 μ (affine in μ without drift), times n/T, folded into conj(C[j, l])
        for (j, x) in enumerate(g.x)
            r0 = vpar_response(m, x, ω, 0.0, sp)
            r1 = vpar_response(m, x, ω, 1.0, sp) - r0
            w = m.n(x) / m.Ti(x)
            for l in 1:N
                u0[j, l] = conj(C[j, l]) * w * r0
                u1[j, l] = conj(C[j, l]) * w * r1
            end
        end
        weber_accumulate!(M, u0, u1, ws.W0[p + 1], ws.W1[p + 1], C, ws.phase, sp)
    end
    A = -(g.Cinv' * M * g.Cinv)
    for j in 1:N
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
