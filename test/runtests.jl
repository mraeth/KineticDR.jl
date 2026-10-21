using Test
using KineticDR
using QuadGK: quadgk
using SpecialFunctions: besselj
using LinearAlgebra: LinearAlgebra, eigen, SymTridiagonal, Diagonal, diag
using PlasmaCore: ScalarField

const KD = KineticDR

adiabatic_electrons(τ = 1.0) = Species(q = -1, T = τ, response = Boltzmann())
struct ConstProfile                                    # a profile that is a callable struct, not a Function
    s::Species
end
(p::ConstProfile)(x) = p.s

# The radial operator with the μ integral by 128-node Gauss–Laguerre quadrature, built independently of the package:
# gyroaverage J_p with the orbit phase, adjoint J', Poisson term
function quadrature_operator(g, m, k, ω; n = 128)
    E = eigen(SymTridiagonal([2j + 1.0 for j in 0:(n - 1)], [Float64(j) for j in 1:(n - 1)]))
    A = Matrix{ComplexF64}(Diagonal([sum(KD.adiabatic_term(sp, x) for sp in m.species) for x in g.x]))
    m.field isa Poisson && (A += m.field.lambda2 * g.C * Diagonal(g.k .^ 2 .+ k.ky^2 .+ k.kz^2) * g.Cinv)
    for sp in m.species
        ss = [KD.species_at(sp, x) for x in g.x]
        ss[1].response isa Boltzmann && continue
        P = ss[1].response isa Gyrokinetic ? 0 : ss[1].response.pmax
        σ = sign(ss[1].q)
        for p in -P:P, (μ, w) in zip(E.values, E.vectors[1, :] .^ 2)
            # harmonic p of e^{ik·ρ}: J_p(k⊥ρ) e^{-iσpθ_k}, θ_k = atan2(k_x, k_y)
            G = [g.C[j, l] * cis(-σ * p * atan(g.k[l], k.ky)) * besselj(p, hypot(g.k[l], k.ky) * sqrt(2μ * s.T * s.m) / abs(s.q))
                 for (j, s) in enumerate(ss), l in eachindex(g.k)]
            J = G * g.Cinv
            A -= J' * Diagonal([w * s.q^2 * s.N / s.T * KD.vpar_response(k, s, ω, μ, p) for s in ss]) * J
        end
    end
    return A
end
paper_model(; κn = 0.23, κT = 0.34, ion = GordeyevSeries(), τ = 1.0, field = Quasineutrality()) =
    Model((Species(; κn, κT, response = ion), adiabatic_electrons(τ)), field)
ion_W(m, ω, k) = nonadiabatic_response(m.species[1], ω, k)

@testset "special functions" begin
    for ζ in (0.3 + 0.5im, -1.2 + 0.1im, 2.0 + 3.0im)
        ref = quadgk(t -> exp(-t^2) / (t - ζ), -Inf, Inf; rtol = 1e-12)[1] / sqrt(π)
        @test plasma_Z(ζ) ≈ ref rtol = 1e-9
    end
    @test plasma_Z(0.0) ≈ im * sqrt(π)
    for x in (0.0, 0.3, 4.0, 50.0)
        @test sum(Gamma_p(p, x) for p in -400:400) ≈ 1 atol = 1e-12
    end
    for ζ in (9.0 + 0.5im, 9.0 - 0.5im, 6.0 - 7.0im, -8.5 - 2.0im)       # below the axis: Landau term kept
        @test KD.one_plus_zZ(ζ) ≈ 1 + ζ * plasma_Z(ζ) rtol = 1e-8
    end
end

@testset "ion response" begin
    k = Wavevector(ky = 0.5, kz = 0.1)
    ω = 0.3 + 0.05im
    S, ST = gordeyev(GordeyevSeries(), ω, k)
    Si, STi = gordeyev(GordeyevIntegral(), ω, k)
    @test S ≈ Si rtol = 1e-8
    @test ST ≈ STi rtol = 1e-8

    # also with kz = 0 (needs Im ω > 0 for the time integral)
    k0 = Wavevector(ky = 0.7, kz = 0.0)
    ω0 = 0.4 + 0.08im
    S, ST = gordeyev(GordeyevSeries(), ω0, k0)
    Si, STi = gordeyev(GordeyevIntegral(), ω0, k0)
    @test S ≈ Si rtol = 1e-7
    @test ST ≈ STi rtol = 1e-7

    # kz -> 0 limit of the Z-series is the kz = 0 formula
    kz_small = Wavevector(ky = 0.7, kz = 1e-7)
    @test all(isapprox.(gordeyev(GordeyevSeries(), ω0, kz_small), gordeyev(GordeyevSeries(), ω0, k0); rtol = 1e-6))

    # static limit: ∫h/φ -> Γ0 without gradients (Notes: n̂/φ = Γ0 - 1)
    kk = Wavevector(ky = 1.3, kz = 0.0)
    @test nonadiabatic_response(Species(), 1e-7im, kk) ≈ Gamma_p(0, 1.3^2) atol = 1e-6

    # gyrokinetic limit is the p = 0 term
    @test gordeyev(Gyrokinetic(), ω, k) == gordeyev(GordeyevSeries(pmax = 0), ω, k)

    # T-derivative against a finite difference of S(T): rescale k and ω-argument
    # S(T) = Σ Γ_p(T k⊥²) a(T) Z((ω-p) a(T)), a = 1/(|kz|√(2T)): check with explicit T
    function S_of_T(T)
        x = T * 0.5^2
        a = 1 / (0.1 * sqrt(2T))
        sum(Gamma_p(p, x) * a * plasma_Z((ω - p) * a) for p in -40:40)
    end
    h = 1e-5
    @test gordeyev(GordeyevSeries(), ω, k)[2] ≈ (S_of_T(1 + h) - S_of_T(1 - h)) / 2h rtol = 1e-6
end

@testset "model and swap" begin
    k = Wavevector(ky = 0.4, kz = 0.025)
    m = paper_model()
    ω = 1.003 + 0.01im
    @test dispersion(m, ω, k) ≈ 2 - ion_W(m, ω, k)
    # electron temperature enters as 1 + 1/τ
    @test dispersion(paper_model(τ = 2.0), ω, k) ≈ 1.5 - ion_W(m, ω, k)
    # Poisson with λ² -> 0 reduces to quasineutrality
    m3 = paper_model(field = Poisson(lambda2 = 0.0))
    @test dispersion(m3, ω, k) ≈ dispersion(m, ω, k)
    # reality condition: D(-ω*, -ky) = conj(D(ω, ky))
    kn = Wavevector(ky = -0.4, kz = 0.025)
    @test dispersion(m, -conj(ω), kn) ≈ conj(dispersion(m, ω, k)) rtol = 1e-10
end

@testset "root finding" begin
    r, ok = muller(z -> (z - 1 - 2im) * (z + 3) * (z - 0.5), 0.5 + 0im, 1 + 1im, 2 + 0im)
    @test ok && any(isapprox(r, z; atol = 1e-9) for z in (1 + 2im, -3, 0.5))
    # PRL Fig. 1 parameters; k_y = 3 is near the maximum of the p = 1, 2 branches
    k = Wavevector(ky = 3.0, kz = 0.025)
    m = paper_model()
    roots = find_modes(m, k; re = 0.0:0.25:4.5)
    @test !isempty(roots)
    @test all(abs(dispersion(m, ω, k)) < 1e-8 for ω in roots)
    # unstable branch near each of the first harmonics exists at these gradients
    for p in 1:2
        @test any(ω -> abs(real(ω) - p) < 0.3 && imag(ω) > 0, roots)
    end
    # argument principle: number of roots in the upper-half strip equals what the search found
    unstable = filter(ω -> imag(ω) > 1e-4 && 0.0 < real(ω) < 3.5, roots)
    n = count_roots(ω -> dispersion(m, ω, k), 0.0, 3.5, 1e-4, 0.1)
    @test n == length(unstable)
end

@testset "neutral gradients" begin
    k = Wavevector(ky = 0.4, kz = 0.025)
    m = paper_model()
    g = neutral_gradients(m, 1.02, k)
    @test g !== nothing
    mm = with_gradients(m; κn = g.κn, κT = g.κT)
    @test abs(dispersion(mm, 1.02 + 0im, k)) < 1e-10
end

@testset "PRL regression values" begin
    # computed with this package, checked against the PRL by eye (Figs. 1-3) and the table (p = 1)
    m = paper_model(κn = 0.25, κT = 0.34)
    ω = harmonic_root(m, Wavevector(kx = 3.0, ky = 4.0, kz = 0.025), 1)
    @test imag(ω) ≈ 0.00687 rtol = 2e-2        # PRL table: 0.0070
    # critical κT at κn = 0.8, ky = 1 for p = 0 (PRL Fig. 2 middle: ≈ 0.75-0.8)
    m0 = paper_model(κn = 0.0, κT = 0.0)
    κc = critical_kappaT(m0, Wavevector(ky = 1.0, kz = 0.025), 0.8, 0)
    @test 0.7 < κc < 0.85
    # the neutral point is a root: D(ω_c; κc) = 0 at some real ω
    @test isfinite(κc)
end

@testset "species model" begin
    # (1) D agrees with the pre-refactor code (separate ion/electron Model and MultiSpeciesModel in the
    # Maeyama sign convention), values from the pre-refactor code (ibw_dispersion study, _temp/pre_refactor_reference.jl)
    pts = [(1.0 + 0.02im, Wavevector(ky = 1.0, kz = 0.025)), (1.04 + 0.007im, Wavevector(kx = 3.0, ky = 4.0, kz = 0.025)),
           (0.3 + 0.05im, Wavevector(ky = 0.5, kz = 0.1))]
    models = Dict("base" => paper_model(κn = 0.25), "tau2" => paper_model(κn = 0.25, τ = 2.0),
                  "poisson" => paper_model(κn = 0.25, field = Poisson(lambda2 = 1e-2)),
                  "gk" => paper_model(κn = 0.25, ion = Gyrokinetic()),
                  "maeyama" => Model((Species(κn = -0.1, κT = -0.3, response = GordeyevSeries(pmax = 12)),
                                      Species(q = -1, m = 1 / 1836, κn = -0.1, κT = -0.5, response = Gyrokinetic())),
                                     Poisson(lambda2 = 0.25 / 1836)))
    reference = [
        ("base", 1, 1.383473706754165 + 4.62849967921451im),
        ("base", 2, -0.1635118940887783 + 0.16352587821644216im),
        ("base", 3, 1.3961692854637715 + -0.06501131105030791im),
        ("tau2", 1, 0.883473706754165 + 4.62849967921451im),
        ("tau2", 2, -0.6635118940887783 + 0.16352587821644216im),
        ("tau2", 3, 0.8961692854637715 + -0.06501131105030791im),
        ("poisson", 1, 1.393479956754165 + 4.62849967921451im),
        ("poisson", 2, 0.08649435591122168 + 0.16352587821644216im),
        ("poisson", 3, 1.3987692854637714 + -0.06501131105030791im),
        ("gk", 1, 1.5628258488413893 + -0.0005705636280353239im),
        ("gk", 2, 1.9439596353013897 + -0.00016328740785479082im),
        ("gk", 3, 1.3944522730894682 + -0.06883877683086781im),
        ("maeyama", 1, 0.5465532265262126 + 7.319230158700353im),
        ("maeyama", 2, -0.27925108511906627 + 2.1164335478699816im),
        ("maeyama", 3, 1.0453998696832425 + 0.26584606045102066im),
    ]
    for (name, j, D) in reference
        ω, k = pts[j]
        @test dispersion(models[name], ω, k) ≈ D rtol = 1e-12
    end

    # (2) Poisson with λ² and electron T_e = 2: Boltzmann electrons give q²N/T = 1/2
    ion = Species(κn = 0.25, κT = 0.34)
    ω = 1.04 + 0.007im
    m2 = Model((ion, adiabatic_electrons(2.0)), Poisson(lambda2 = 0.1))
    k = Wavevector(ky = 4.0, kz = 0.025)
    @test dispersion(m2, ω, k) ≈ 0.1 * (16 + 0.025^2) - (charge_response(ion, ω, k) - 0.5) rtol = 1e-12

    # (3) kinetic electron in its own units: k∥ = 0 closed form (Maeyama Eq. 18) for the gyrokinetic ℓ = 0 term
    e = Species(q = -1, m = 1 / 3670, κn = -0.05, κT = -0.18, response = Gyrokinetic())
    kk = Wavevector(ky = 20.0, kz = 0.0)
    ωe = 0.3 + 0.1im
    x = (kk.ky * thermal_gyroradius(e))^2
    C0, C1 = Gamma_p(0, x), Gamma_p(1, x)
    ω_s = (e.T / e.q) * kk.ky * e.κn          # ω_*e
    ω_T = (e.T / e.q) * kk.ky * e.κT          # g ω_*e
    W0 = C0 / ωe * (ωe - ω_s - ω_T * (0 - x * (1 - C1 / C0)))
    @test nonadiabatic_response(e, ωe, kk) ≈ W0 rtol = 1e-10

    # (4) parallel-kinetic electrons without FLR (k⊥ = 0): charge response -(1 + ζZ(ζ))/τ,
    # ζ = ω/(√2 |kz| v_te), v_te = √(τ/μ); its μ -> 0 limit is the Boltzmann species
    μ, τ = 1 / 1836, 2.0
    el = Species(q = -1, m = μ, T = τ, response = Gyrokinetic())
    kz = Wavevector(ky = 0.0, kz = 0.025)
    ζ = ω / (sqrt(2) * 0.025 * sqrt(τ / μ))
    @test charge_response(el, ω, kz) ≈ -(1 + ζ * plasma_Z(ζ)) / τ rtol = 1e-12
    @test charge_response(Species(q = -1, m = 1e-14, T = τ, response = Gyrokinetic()), ω, kz) ≈
          charge_response(adiabatic_electrons(τ), ω, kz) rtol = 1e-5

    # (5) gradient scans act on the chosen species only
    mm = with_gradients(m2; species = 2, κn = 0.7, κT = 0.1)
    @test mm.species[1] === m2.species[1] && mm.species[2].κn == 0.7

    # (6) root finding with a scaled function
    mp = paper_model(κn = 0.25)
    kp = Wavevector(kx = 3.0, ky = 4.0, kz = 0.025)
    r, ok = find_root(mp, kp, 1.04 + 0.007im)
    @test ok && imag(r) > 0
    r2, ok2 = find_root(z -> dispersion(mp, 1e-3 * z, kp), 1040 + 7im; δ = 1.0)
    @test ok2
    @test r2 * 1e-3 ≈ r rtol = 1e-8

    # (7) track_branch: model and function forms agree on a vector of wavevectors
    ks = [Wavevector(kx = 3.0, ky = 4.0 + d, kz = 0.025) for d in (0.0, 0.01, 0.02)]
    @test track_branch(mp, ks, r) == track_branch((ω, k) -> dispersion(mp, ω, k), ks, r)
end

@testset "electromagnetic" begin
    # (1) cold Darwin against the closed-form Stix biquadratic A n⁴ - B n² + C = 0 (n̂² = (2/β) K²/ω²),
    # K² from the closed form at real ω, then the ω-root at that K must return ω;
    # ion terms S = -1/(ω²-1), D = 1/(ω(ω²-1)), P = -1/ω² and three electron closures (μ = m_e/m_i)
    β, μ = 0.1, 1 / 100
    # drift-kinetic electrons at T_e = T_i: at θ = 0 det = P (R - n²)(L - n²), so the transverse roots do
    # not depend on the (kinetic) P; at θ = π/2, k∥ = 0 and P = -1/(μω²) exactly, and the electron
    # pressure adds -2K²/ω² = -β n² to χ_yy: X mode n² = (S² - D²)/(S(1 + β)). Oblique: cold only.
    # Maxwell adds the vacuum term λ² to S and P.
    closures = [
        "cold electrons" => (Species(q = -1, m = μ, response = Cold()),
                             w -> μ / (1 - μ^2 * w^2), w -> 1 / (w * (1 - μ^2 * w^2)), (0.0, π / 2, 0.7), 0.0),
        "drift-kinetic"  => (Species(q = -1, m = μ, response = DriftKinetic()), w -> 0.0, w -> 1 / w, (0.0, π / 2), β),
        "drift-kinetic + polarisation" =>
                            (Species(q = -1, m = μ, response = DriftKinetic(polarization = true)), w -> μ, w -> 1 / w, (0.0, π / 2), β),
    ]
    nfound = 0
    for (name, (el, sE, dE, θs, βe)) in closures, θ in θs, w in (0.4, 3.0), λ2 in (0.0, 0.05)
        m = Model((Species(response = Cold()), el), λ2 == 0 ? Darwin(beta = β) : Maxwell(beta = β, lambda2 = λ2))
        S = -1 / (w^2 - 1) + sE(w) + λ2
        D = 1 / (w * (w^2 - 1)) + dE(w)
        P = -1 / w^2 - 1 / (μ * w^2) + λ2
        s2, c2 = sin(θ)^2, cos(θ)^2
        A, B, C = S * s2 + P * c2, (S^2 - D^2) * s2 + P * S * (1 + c2), P * (S^2 - D^2)
        q = sqrt(complex(B^2 - 4A * C))
        n2s = θ == π / 2 ? (P, (S^2 - D^2) / (S * (1 + βe))) : ((B + q) / 2A, (B - q) / 2A)
        for n2 in n2s
            K2 = β / 2 * w^2 * n2
            (abs(imag(K2)) < 1e-12 && 1e-3 < real(K2) < 25) || continue
            K = sqrt(real(K2))
            kz = θ == π / 2 ? 0.0 : K * cos(θ)                     # exact k∥ = 0, not K cos(π/2) ≈ 1e-17 K
            ωr, ok = find_root(m, Wavevector(kx = K * sin(θ), ky = 0.0, kz = kz), 1.001 * w; δ = 1e-4 * w, maxstep = 0.01 * w)
            @test ok
            @test ωr ≈ w rtol = 1e-10
            nfound += 1
        end
    end
    @test nfound >= 24

    # (2) series against the orbit integral (all nine elements), ions and electrons, signs of kx and kz
    ω = 0.6 + 0.05im
    for (q, m) in ((1, 1.0), (-1, 1 / 100)),
        k in (Wavevector(kx = 0.7, ky = 0.0, kz = 0.3), Wavevector(kx = -0.7, ky = 0.0, kz = 0.3), Wavevector(kx = 0.7, ky = 0.0, kz = -0.3),
              Wavevector(kx = 1.3, ky = 0.0, kz = 0.02))
        A = susceptibility(Species(; q, m), ω, k)
        B = susceptibility(Species(; q, m, response = GordeyevIntegral()), ω, k)
        @test maximum(abs.(A - B)) < 1e-11 * maximum(abs.(A))
        # (3) electrostatic projection: k·χ̃_s·k = -q_s δn_s/φ of the scalar (electrostatic) path
        kv = [k.kx, 0, k.kz]
        @test transpose(kv) * A * kv ≈ -charge_response(Species(; q, m), ω, k) rtol = 1e-12
    end
    k0 = Wavevector(kx = 0.9, ky = 0.0, kz = 0.0)
    @test transpose([0.9, 0, 0]) * susceptibility(Species(), ω, k0) * [0.9, 0, 0] ≈ -charge_response(Species(), ω, k0) rtol = 1e-12

    # (4) hot -> cold for T -> 0 (ions and finite-mass electrons; checks the mirror for q < 0):
    # thermal corrections are O(T), so the error must drop 100× from T = 1e-8 to 1e-10
    k = Wavevector(kx = 0.8, ky = 0.0, kz = 0.5)
    ωc = 0.37 + 0.01im
    for (q, m) in ((1, 1.0), (-1, 1 / 100))
        cold = susceptibility(Species(; q, m, response = Cold()), ωc, k)
        err(T) = maximum(abs.(susceptibility(Species(; q, m, T), ωc, k) - cold)) / maximum(abs.(cold))
        @test err(1e-10) < 1e-7
        @test 50 < err(1e-8) / err(1e-10) < 200
    end

    # (5) drift-kinetic parallel response = k⊥ = 0 zz element of the hot tensor
    e(r) = Species(q = -1, m = 1 / 100, response = r)
    kz = Wavevector(ky = 0.0, kz = 0.3)
    @test susceptibility(e(DriftKinetic()), ω, kz)[3, 3] ≈ susceptibility(e(GordeyevSeries()), ω, kz)[3, 3] rtol = 1e-12

    # (6) Alfvén wave: θ = 0, K -> 0 gives ω/K = v_A/v_ti = √(2/β) (ion-cyclotron correction O(ω))
    mA = Model((Species(response = Cold()), Species(q = -1, m = μ, T = 1e-12, response = DriftKinetic())), Darwin(beta = 1.0))
    KA = 1e-3
    ωA, okA = find_root(mA, Wavevector(ky = 0.0, kz = KA), KA * sqrt(2.0); δ = 1e-6)
    @test okA && abs(real(ωA) / (KA * sqrt(2.0)) - 1) < 1e-3

    # (7) Hermitian for real ω, k∥ = 0, off-harmonic (no dissipation)
    χh = susceptibility(Species(), 0.5, Wavevector(kx = 1.2, ky = 0.0))
    @test χh ≈ χh' atol = 1e-14

    # (8) Maxwell with λ² -> 0 is Darwin; the longitudinal part is Poisson
    mD = Model((Species(), Species(q = -1, m = μ)), Darwin(beta = β))
    mM = Model((Species(), Species(q = -1, m = μ)), Maxwell(beta = β, lambda2 = 0.0))
    @test dispersion(mM, ω, k) ≈ dispersion(mD, ω, k) rtol = 1e-12
    mλ = Model(mD.species, Maxwell(beta = β, lambda2 = 0.01))
    kv = [k.kx, 0, k.kz]
    @test transpose(kv) * dispersion_matrix(mλ, ω, k) * kv ≈ dispersion(Model(mD.species, Poisson(lambda2 = 0.01)), ω, k) rtol = 1e-12

    # (9) any direction of k⊥: 𝒟 rotates about b̂, so det is invariant
    for m in (mD, mλ)
        @test dispersion(m, ω, Wavevector(kx = 0.3, ky = 0.4, kz = 0.1)) ≈ dispersion(m, ω, Wavevector(kx = 0.5, ky = 0.0, kz = 0.1)) rtol = 1e-12
        @test dispersion(m, ω, Wavevector(kx = 0.0, ky = -0.5, kz = 0.1)) ≈ dispersion(m, ω, Wavevector(kx = 0.5, ky = 0.0, kz = 0.1)) rtol = 1e-12
    end
    ky = Wavevector(kx = 0.0, ky = 0.7, kz = 0.3)
    kv = [0.0, 0.7, 0.3]
    @test transpose(kv) * susceptibility(Species(q = -1, m = μ), ω, ky) * kv ≈ -charge_response(Species(q = -1, m = μ), ω, ky) rtol = 1e-12

    # (10) parallel propagation and k⊥ → 0: series against the orbit integral, also below the real axis
    for (q, m) in ((1, 1.0), (-1, 1 / 100)), kx in (0.0, 1e-9, 3e-8, 0.4), ωt in (0.6 + 0.05im, 0.6 - 0.05im, 0.3 - 0.2im)
        k = Wavevector(kx = kx, ky = 0.0, kz = 0.3)
        A = susceptibility(Species(; q, m), ωt, k)
        B = susceptibility(Species(; q, m, response = GordeyevIntegral()), ωt, k)
        @test maximum(abs.(A - B)) < 1e-10 * maximum(abs.(A))
    end
    @test abs(susceptibility(Species(), 0.6 + 0.05im, Wavevector(ky = 0.0, kz = 0.3))[1, 1]) > 0.1

    # (11) DriftKinetic is the m → 0 limit of the hot tensor at fixed ζ (k∥ ∝ √m), element-wise in χ
    function dkerr(me)
        k = Wavevector(kx = 0.5, ky = 0.0, kz = 0.05 * sqrt(me / 1e-2))
        A = susceptibility(Species(q = -1, m = me), 0.3 + 0.01im, k)
        B = susceptibility(Species(q = -1, m = me, response = DriftKinetic()), 0.3 + 0.01im, k)
        return maximum(abs.(A - B)[i, j] / max(abs(B[i, j]), 1.0) for i in 1:3, j in 1:3)
    end
    @test dkerr(1e-6) < 1e-2
    @test dkerr(1e-6) < 0.1 * dkerr(1e-4)

    # (12) guards
    @test_throws ArgumentError nonadiabatic_response(Species(response = Cold()), ω, k)          # fluid: no Gordeyev sum
    @test_throws ArgumentError charge_response(Species(κT = 0.1, response = Cold()), ω, k)      # no temperature
    @test_throws ArgumentError dispersion(Model((Species(), Species(q = -1, response = Boltzmann())), Darwin(beta = β)), ω,
                                          Wavevector(ky = 0.5, kz = 0.0))                     # Boltzmann needs k∥ ≠ 0
end

@testset "radial operator" begin
    mg = radial_grid(-20, 20, 40; bc = :mirror)
    rgrid = (-0.08:0.01:0.0, (0.002, 0.01, 0.02))
    # uniform profiles: the operator is singular at the local root of the same model, flat null vector
    m = Model((Species(κT = -0.3, response = Gyrokinetic()), adiabatic_electrons()), Quasineutrality())
    for kz in (0.01, 0.02)
        k = Wavevector(ky = 0.5, kz = kz)
        loc = find_modes(m, k; re = rgrid[1], imag = rgrid[2], imin = 0.0)[1]
        λ, φ = KD.smallest(operator(mg, m, k, loc))
        @test abs(λ) < 1e-10
        @test maximum(abs.(φ ./ φ[1] .- 1)) < 1e-9
    end
    # localised by the profile: κT = -0.3 sech²(x/4), electrons with the ion particle density (bsl6d's
    # convention), Dirichlet walls; converged against N, and the walls do not matter
    T(x) = exp(-1.2 * tanh(x / 4))
    ionp = species_profile(; T, response = Gyrokinetic())
    mp = Model((ionp, species_profile(; q = -1, N = particle_density(ionp), response = Boltzmann())), Quasineutrality())
    kp = Wavevector(ky = 0.5, kz = 0.01)
    ω0 = local_roots(mp, kp, [0.0]; re = rgrid[1], imag = rgrid[2])[1]
    ref = eigenmode(radial_grid(-20, 20, 120), mp, kp, ω0)
    @test ref.converged
    @test ref.ω ≈ -0.027990992483326 + 0.006821667129586im atol = 1e-11
    @test ref.φ isa ScalarField && length(ref.φ) == 120 && ref.φ[1] == 0
    for N in (80, 160)
        @test eigenmode(radial_grid(-20, 20, N), mp, kp, ω0).ω ≈ ref.ω atol = 1e-11
    end
    rmir = eigenmode(radial_grid(-20, 20, 120; bc = :mirror), mp, kp, ω0)
    @test rmir.converged
    @test rmir.ω ≈ ref.ω atol = 1e-11
    # the argument principle sees every eigenvalue, the seeded search only some
    g = radial_grid(-20, 20, 80)
    found = find_eigenmodes(g, mp, kp; re = -0.06:0.005:0.0, imag = 0.001:0.002:0.015)
    @test any(r -> abs(r.ω - ref.ω) < 1e-10, found)
    @test all(r -> -0.06 <= real(r.ω) <= 0 && 0.001 <= imag(r.ω) <= 0.015, found)          # inside the box only
    @test count_eigenvalues(g, mp, kp, -0.06, 0.0, 0.001, 0.015) >= length(found) >= 1
    @test find_eigenmode(g, mp, kp; re = -0.06:0.005:0.0, imag = 0.001:0.002:0.015).ω ≈ ref.ω atol = 1e-10
    @test length(rmir.φ) == 120                                                              # mirror: cell centres
    # far below the real axis Z overflows: no exception, converged = false; searches there do not abort
    g40 = radial_grid(-20, 20, 40)
    @test !eigenmode(g40, mp, kp, 1.0 - 0.5im).converged
    @test find_eigenmodes(g40, mp, kp; re = -0.3:0.1:0.0, imag = -0.4:0.1:0.0) isa Vector
    # the tolerance does not scale with ‖A(ω)‖ (≈ 4e8 at this guess)
    @test !eigenmode(radial_grid(-20, 20, 80), mp, kp, -0.3 - 0.3im).converged
    # the gyroradius average needs the profile within ~12 ρ of the node only
    nb = species_profile(; N = x -> abs(x) <= 31.5 ? exp(0.1x) : error("outside"))
    @test particle_density(nb)(19.0) ≈ exp(1.9 + 0.005) rtol = 1e-12
end

@testset "radial species interface" begin
    # any tuple of species and field: at uniform profiles the operator is singular at the local root,
    # with a flat null vector (k_x = 0 mode of the mirror basis)
    g = radial_grid(-20, 20, 40; bc = :mirror)
    k = Wavevector(ky = 0.5, kz = 0.01)
    ion = Species(κn = -0.05, κT = -0.3, response = Gyrokinetic())
    el = Species(q = -1, m = 1 / 400, κn = -0.05, κT = -0.3, response = Gyrokinetic())
    for m in (Model((ion, el), Quasineutrality()),                                        # kinetic electrons
              Model((ion, adiabatic_electrons()), Poisson(lambda2 = 0.5)))
        loc = find_modes(m, k; re = -0.08:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)[end]
        λ, φ = KD.smallest(operator(g, m, k, loc))
        @test abs(λ) < 1e-12
        @test maximum(abs.(φ ./ φ[1] .- 1)) < 1e-10
    end
    # a uniform Species and the constant profile x -> Species are the same operator
    g2 = radial_grid(-20, 20, 40)
    ionG = Species(κT = -0.3, response = Gyrokinetic())
    @test operator(g2, Model((ionG, adiabatic_electrons()), Quasineutrality()), k, -0.024 + 0.003im) ≈
          operator(g2, Model((x -> ionG, adiabatic_electrons()), Quasineutrality()), k, -0.024 + 0.003im) rtol = 1e-12
    # species_profile: κ from N(x), T(x); x-dependent response; same operator as the hand-built species
    N, T = x -> 1 + 0.2tanh(x / 4), x -> 1 + 0.4tanh(-x / 1.5)
    sp = species_profile(; N, T, response = x -> Gyrokinetic())
    hand(x) = Species(; N = N(x), T = T(x), κn = log_gradient(N)(x), κT = log_gradient(T)(x), response = Gyrokinetic())
    @test sp(1.3).κT ≈ -0.4 / 1.5 * sech(1.3 / 1.5)^2 / T(1.3) rtol = 1e-7
    @test sp(1.3).κn ≈ 0.2 / 4 * sech(1.3 / 4)^2 / N(1.3) rtol = 1e-7
    m3 = Model((sp, species_profile(; q = -1, N, response = Boltzmann())), Quasineutrality())
    m4 = Model((hand, x -> Species(q = -1, N = N(x), response = Boltzmann())), Quasineutrality())
    @test operator(g2, m3, k, -0.024 + 0.003im) == operator(g2, m4, k, -0.024 + 0.003im)
    # model(x) is the local model; local solvers reject an unevaluated profile model
    @test m3(1.3).species[2] == Species(q = -1, N = N(1.3), κn = log_gradient(N)(1.3), response = Boltzmann())
    @test (m3(1.3).species[1].T, m3(1.3).species[1].κT) == (T(1.3), sp(1.3).κT)
    @test dispersion(m3(1.3), 0.01 + 0.01im, k) isa Complex
    @test_throws ArgumentError dispersion(m3, 0.01 + 0.01im, k)
    # not (yet) radial: electromagnetic fields, full orbit without a fixed cut-off, k_x ≠ 0, non-finite guess
    @test_throws ArgumentError operator(g2, Model((ion, el), Darwin(beta = 0.1)), k, 0.1 + 0.01im)
    @test_throws ArgumentError operator(g2, Model((Species(), el), Quasineutrality()), k, 0.1 + 0.01im)
    @test_throws ArgumentError operator(g2, m3, Wavevector(kx = 0.1, ky = 0.5), 0.1 + 0.01im)
    @test_throws ArgumentError eigenmode(g2, m3, k, complex(NaN, NaN))
    @test_throws ArgumentError eigenmode(g2, m3, Wavevector(ky = 0.5), 0.0)                  # pole at ω = 0, k_z = 0
    @test_throws ArgumentError RadialGrid(KD.Grid([-10.0], [10.0], [32], 1, 2.0))              # B ≠ 1
    # a callable struct as a profile: radial works, local solvers get the explanatory error
    mc = Model((ConstProfile(ionG), adiabatic_electrons()), Quasineutrality())
    @test operator(g2, mc, k, -0.024 + 0.003im) ≈ operator(g2, Model((ionG, adiabatic_electrons()), Quasineutrality()), k, -0.024 + 0.003im)
    @test_throws ArgumentError dispersion(mc, 0.01 + 0.01im, k)
    # a workspace belongs to its grid, model and wavevector
    ws = KD.RadialWorkspace(g2, m3, k)
    @test_throws ArgumentError operator(g2, m3, Wavevector(ky = 0.7, kz = 0.01), 0.01 + 0.01im, ws)
    @test_throws ArgumentError operator(g2, m4, k, 0.01 + 0.01im, ws)
    # species_profile defaults to the radial kernel
    @test species_profile()(0.0).response isa Gyrokinetic
end

@testset "radial: closed-form μ integral and particle density" begin
    # Weber's second integral against quadrature, real and complex arguments, up to large k⊥ρ
    for p in 0:2, (a, b) in ((0.7, 1.9), (0.3 + 0.2im, 1.1 - 0.1im), (6.0, 7.5))
        W(k) = quadgk(μ -> exp(-μ) * μ^k * besselj(p, a * sqrt(2μ)) * besselj(p, b * sqrt(2μ)), 0, Inf; rtol = 1e-13)[1]
        W0, W1 = KD.weber(p, a, b)
        @test W0 ≈ W(0) atol = 1e-14
        @test W1 ≈ W(1) atol = 1e-13
    end
    @test KD.weber(1, 1.3, 1.3)[1] ≈ Gamma_p(1, 1.3^2) rtol = 1e-14
    # v∥ average against quadrature (Im ω > 0), and its k_z = 0 limit
    s = Species(T = 1.3, m = 2.0, κn = -0.1, κT = -0.3)
    kv = Wavevector(ky = 0.7, kz = 0.05)
    for ω in (0.02 + 0.01im, -0.05 + 0.002im), μ in (0.4, 2.0)
        ωs(v) = s.T / s.q * kv.ky * (s.κn + s.κT * (μ + v^2 / 2 - 1.5))
        direct = quadgk(v -> exp(-v^2 / 2) / sqrt(2π) * (ω - ωs(v)) / (ω - kv.kz * thermal_speed(s) * v), -Inf, Inf; rtol = 1e-12)[1]
        @test KD.vpar_response(kv, s, ω, μ) ≈ direct rtol = 1e-10
        @test KD.vpar_response(Wavevector(ky = 0.7), s, ω, μ) ≈ (ω - ωs(1.0)) / ω rtol = 1e-14
    end
    # the operator is the μ integral done by quadrature (`quadrature_operator`, built independently). Nonuniform
    # N, T, κ ≠ 0; ions with q = 2, m = 4 and kinetic electrons; both bases (small N: the quadrature resolves every k_x)
    N, T = x -> 1 + 0.3tanh(x / 3), x -> exp(-0.8tanh(x / 2))
    k = Wavevector(ky = 0.7, kz = 0.02)
    ion2 = species_profile(; q = 2, m = 4, N = x -> N(x) / 2, T = x -> 1.5T(x))
    kel = species_profile(; q = -1, m = 1 / 400, N, T = x -> 0.8 + 0.1x / 15)
    for mw in (Model((species_profile(; N, T), species_profile(; q = -1, N, response = Boltzmann())), Quasineutrality()),
               Model((ion2, kel), Quasineutrality()),
               Model((ion2, kel, species_profile(; q = -1, N = x -> 0.1, response = Boltzmann())), Poisson(lambda2 = 0.3))),
        bc in (:dirichlet, :mirror)
        g = radial_grid(-15, 15, 40; bc)
        Aw = operator(g, mw, k, -0.03 + 0.01im)
        @test maximum(abs.(Aw - quadrature_operator(g, mw, k, -0.03 + 0.01im))) < 1e-12 * maximum(abs.(Aw))
    end
    # uniform profiles: A is diagonal in the basis, entry l the local D at k_x = k_l (every mode, both
    # bases, q = 2 ions + kinetic electrons, Poisson with an extra Boltzmann species, k_z = 0, Im ω < 0)
    iu = Species(q = 2, m = 4, T = 1.5, N = 0.5, κn = -0.1, κT = -0.3, response = Gyrokinetic())
    eu = Species(q = -1, m = 1 / 400, κT = -0.2, response = Gyrokinetic())
    for (mu, ku, ω) in ((Model((iu, eu), Quasineutrality()), Wavevector(ky = 0.6, kz = 0.03), 0.02 + 0.01im),
                        (Model((iu, eu, Species(q = -1, N = 0.1, response = Boltzmann())), Poisson(lambda2 = 0.3)),
                         Wavevector(ky = -0.6, kz = -0.03), 0.02 - 0.005im),
                        (Model((iu, eu), Quasineutrality()), Wavevector(ky = 0.6), 0.02 + 0.01im)),
        bc in (:dirichlet, :mirror)
        g = radial_grid(-15, 15, 24; bc)
        Ad = g.Cinv * operator(g, mu, ku, ω) * g.C
        local_D = [dispersion(mu, ω, Wavevector(kx = q, ky = ku.ky, kz = ku.kz)) for q in g.k]
        @test maximum(abs.(Ad - Diagonal(Ad))) < 1e-12 * maximum(abs.(Ad))
        @test maximum(abs.(diag(Ad) - local_D)) < 1e-12 * maximum(abs.(local_D))
    end
    # particle density of an exponential guiding-centre profile: n e^{κ² ρ_t²/2}, ρ_t² = T m/q²
    @test particle_density(species_profile(; N = x -> exp(0.5x)))(0.3) ≈ exp(0.15 + 0.125) rtol = 1e-12
    @test particle_density(species_profile(; N = x -> exp(0.5x), T = x -> 2.0, m = 4.0, q = 2.0))(0.3) ≈
          exp(0.15 + 0.25) rtol = 1e-12
    # adiabatic term: q² ∫F/T d³v at the particle position for a kinetic profile, N/T at the node otherwise
    ion = species_profile(; N = x -> exp(0.5x), response = Gyrokinetic())
    @test KD.adiabatic_term(ion, 0.3) ≈ exp(0.15 + 0.125) rtol = 1e-12
    @test KD.adiabatic_term(species_profile(; q = -1, N = x -> exp(0.5x), response = Boltzmann()), 0.3) == exp(0.15)
    @test KD.adiabatic_term(Species(N = 0.8, T = 1.3), 0.3) == 0.8 / 1.3
end

@testset "radial: Larmor harmonics (full orbit)" begin
    # uniform profiles, periodic basis: A is diagonal and entry l is the local full-orbit D at k_x = k_l (k_x of
    # both signs), for full-orbit ions with Boltzmann or full-orbit electrons
    for sp in ((Species(κn = 0.23, κT = 0.34, response = GordeyevSeries(pmax = 12)), Species(q = -1, response = Boltzmann())),
               (Species(κn = 0.23, κT = 0.34, response = GordeyevSeries(pmax = 12)),
                Species(q = -1, m = 0.05, T = 0.8, κT = 0.2, response = GordeyevSeries(pmax = 6))))
        m, k, ω = Model(sp, Quasineutrality()), Wavevector(ky = 2.0, kz = 0.05), 1.05 + 0.01im
        g = radial_grid(-10, 10, 24; bc = :periodic)
        Ad = g.Cinv * operator(g, m, k, ω) * g.C
        local_D = [dispersion(m, ω, Wavevector(kx = q, ky = k.ky, kz = k.kz)) for q in g.k]
        @test maximum(abs.(Ad - Diagonal(Ad))) < 1e-13 * maximum(abs.(Ad))
        @test maximum(abs.(diag(Ad) - local_D)) < 1e-13 * maximum(abs.(local_D))
    end
    # and singular at the local IBW root (p = 1)
    mu = Model((Species(κn = 0.23, κT = 0.34, response = GordeyevSeries(pmax = 8)), adiabatic_electrons()), Quasineutrality())
    ku = Wavevector(ky = 5.0, kz = 0.025)
    loc = first(filter(z -> abs(real(z) - 1) < 0.4, find_modes(mu, ku; re = 0.7:0.1:1.3, imag = (0.002, 0.01), imin = 0.0)))
    @test abs(first(KD.smallest(operator(radial_grid(-20, 20, 24; bc = :periodic), mu, ku, loc)))) < 1e-9
    r = eigenmode(radial_grid(-20, 20, 24; bc = :periodic), mu, ku, loc * (1 + 1e-4))       # end to end
    @test r.converged
    @test r.ω ≈ loc rtol = 1e-9
    # the assembled matrix carries the symbol's orbit phase: κn(x) = ε cos(Kx) (uniform N, T) couples k' to k' ± K
    # with entry -(ε/2) Σ_p e^{iσp(θ_k - θ_k')} W0_p(k⊥ρ, k⊥'ρ) ∂W_p/∂κn, the sum of the symbol test below
    for ky in (1.3, -1.3), q in (1, -1)
        ε, L, Tc = 0.3, 10.0, 1.4
        K = 2 * 2π / L
        sp = x -> Species(; q, T = Tc, κn = ε * cos(K * x), response = GordeyevSeries(pmax = 3))
        m = Model((sp, Species(q = -q, response = Boltzmann())), Quasineutrality())
        g, k, ω = radial_grid(0, L, 24; bc = :periodic), Wavevector(; ky, kz = 0.1), 1.05 + 0.02im
        Ad = g.Cinv * operator(g, m, k, ω) * g.C
        s0, s1 = Species(; q, T = Tc), Species(; q, T = Tc, κn = 1.0)
        ρ = sqrt(Tc) / abs(q)
        kp(k) = hypot(k, ky)
        coupling(kk, k′) = -(ε / 2) / Tc * sum(cis(sign(q) * p * (atan(kk, ky) - atan(k′, ky))) * KD.weber(p, kp(kk) * ρ, kp(k′) * ρ)[1] *
                                               (KD.vpar_response(k, s1, ω, 0.0, p) - KD.vpar_response(k, s0, ω, 0.0, p)) for p in -3:3)
        l′ = 12                                          # k_l′ = -2π/L·1; k_l′ ± K inside the band, away from Nyquist
        for dl in (2, -2)
            @test Ad[l′ + dl, l′] ≈ coupling(g.k[l′ + dl], g.k[l′]) rtol = 1e-12
        end
        @test abs(Ad[l′ + 1, l′]) < 1e-14 * abs(Ad[l′, l′])     # nothing else
    end
    # non-uniform periodic profiles: the closed-form μ integral is the μ quadrature (assembly with the orbit phase)
    T(x) = 1 + 0.3 * sin(2π * x / 10)
    n(x) = 1 + 0.2 * cos(2π * x / 10)
    ion = species_profile(; N = n, T, response = GordeyevSeries(pmax = 3))
    for el in (species_profile(; q = -1, N = particle_density(ion), response = Boltzmann()),
               species_profile(; q = -1, m = 0.1, N = n, T = x -> 0.5T(x), response = GordeyevSeries(pmax = 2)))
        m, k, g = Model((ion, el), Quasineutrality()), Wavevector(ky = 1.3, kz = 0.1), radial_grid(0, 10.0, 24; bc = :periodic)
        A = operator(g, m, k, 1.05 + 0.02im)
        @test maximum(abs.(A - quadrature_operator(g, m, k, 1.05 + 0.02im))) < 1e-12 * maximum(abs.(A))
    end
    # mirror y -> -y: a species of charge -q at k_y is the species of charge q at -k_y (ω_* = (T/q) k_y κ unchanged)
    g = radial_grid(0, 10.0, 24; bc = :periodic)
    sp(q) = species_profile(; q, N = n, T, response = GordeyevSeries(pmax = 3))
    bz(q) = species_profile(; q, N = n, response = Boltzmann())
    @test operator(g, Model((sp(-1), bz(1)), Quasineutrality()), Wavevector(ky = 1.3, kz = 0.1), 1.05 + 0.02im) ≈
          operator(g, Model((sp(1), bz(-1)), Quasineutrality()), Wavevector(ky = -1.3, kz = 0.1), 1.05 + 0.02im) rtol = 1e-12
    @test length(eigenmode(g, Model((ion, bz(-1)), Quasineutrality()), Wavevector(ky = 1.3, kz = 0.1), 1.05 + 0.02im).φ) == 24
    # exponential density, uniform T: the exact local DR (all orders in κn) ...
    ω, ky, kz = 0.9 + 0.2im, 2.0, 0.1
    for (Ti, Te) in ((1.0, 1.0), (1.7, 0.8))             # ... is the full-orbit local D for κn → 0
        lm = Model((Species(T = Ti, response = GordeyevSeries(pmax = 60)), Species(q = -1, T = Te, response = Boltzmann())), Quasineutrality())
        @test exponential_dispersion(ω, 0.3, ky, kz; κn = 1e-12, T = Ti, Te) ≈ dispersion(lm, ω, Wavevector(; kx = 0.3, ky, kz)) rtol = 1e-10
    end
    # ... equals the time integral of the impulse response (exponent -2Q sin²(t/2) - iR sin t - kz² t²/2, R = -κn ky) ...
    κ, kxp = 0.5, 0.3
    Q, R = kxp^2 + ky^2 + κ^2 / 4, -κ * ky
    G = exp(κ^2 / 2) * quadgk(t -> exp(-2Q * sin(t / 2)^2 - im * R * sin(t) - kz^2 * t^2 / 2 - im * ω * t), -400, 0; rtol = 1e-13)[1]
    @test exponential_dispersion(ω, kxp, ky, kz; κn = κ) ≈ 1 + exp(-κ^2 / 2) * (exp(κ^2 / 2) + im * (ω + R) * G) rtol = 1e-12
    # ... and differs from the Boussinesq D at O(κn) by the orbit phase across the gradient, not by a k_x shift
    G0(w) = quadgk(t -> exp(-2(kxp^2 + ky^2) * sin(t / 2)^2 - kz^2 * t^2 / 2 - im * w * t), -400, 0; rtol = 1e-13)[1]
    lm = Model((Species(κn = 1e-4, response = GordeyevSeries(pmax = 60)), adiabatic_electrons()), Quasineutrality())
    d = dispersion(lm, ω, Wavevector(; kx = kxp, ky, kz)) - exponential_dispersion(ω, kxp, ky, kz; κn = 1e-4)
    @test d ≈ -1e-4 * im * ω * ky / 2 * (G0(ω - 1) - G0(ω + 1)) rtol = 1e-3
    # the radial operator's symbol on e^{i k_x x}, k_x = kx′ + iκ/2, for n = e^{κx} (guiding centre, exact adiabatic
    # terms) is the exact DR; k_y of both signs
    for (ky, Ti, Te) in ((2.0, 1.0, 1.0), (-2.0, 1.0, 1.0), (2.0, 1.7, 0.8), (-1.5, 0.6, 1.3))
        ion = species_profile(; N = x -> exp(κ * x), T = x -> Ti, response = GordeyevSeries(pmax = 40))
        k, s0 = Wavevector(; ky, kz), ion(0.0)
        kx = kxp + im * κ / 2
        q = kx - im * κ                                  # output wavenumber: n(x) e^{i k_x x}
        kp(k) = sqrt(k^2 + ky^2)
        eθ(k) = (ky + im * k) / kp(k)                    # e^{iθ_k}, θ_k = atan2(k_x, k_y), continued to complex k_x
        ρ = sqrt(Ti)
        σ = sum(KD.vpar_response(k, s0, ω, 0.0, p) * KD.weber(p, kp(q) * ρ, kp(kx) * ρ)[1] * (eθ(q) / eθ(kx))^p for p in -40:40) / Ti
        np = particle_density(ion)(0.0)
        @test (np / Te + KD.adiabatic_term(ion, 0.0) - σ) / np ≈ exponential_dispersion(ω, kxp, ky, kz; κn = κ, T = Ti, Te) rtol = 1e-12
    end
    # interface
    @test_throws ArgumentError operator(g, Model((Species(), adiabatic_electrons()), Quasineutrality()), Wavevector(ky = 1.0), 0.1 + 0.01im)
    @test_throws ArgumentError operator(radial_grid(0, 10.0, 24), mu, ku, 1.0 + 0.01im)                 # p ≠ 0 needs :periodic
    @test_throws ArgumentError radial_grid(0, 10.0, 25; bc = :periodic)
    @test_throws ArgumentError operator(g, Model((Species(response = GordeyevSeries(pmax = -1)), adiabatic_electrons()), Quasineutrality()),
                                        Wavevector(ky = 1.0, kz = 0.1), 0.1 + 0.01im)                       # pmax < 0
    nonper = species_profile(; N = x -> 1 + 0.05x, response = GordeyevSeries(pmax = 2))
    @test_throws ArgumentError operator(g, Model((nonper, adiabatic_electrons()), Quasineutrality()), Wavevector(ky = 1.0, kz = 0.1), 0.1 + 0.01im)
    @test_throws ArgumentError eigenmode(g, mu, Wavevector(ky = 5.0), 1.0 + 0.0im)                          # pole ω = |Ω| at k_z = 0
    # Gyrokinetic and GordeyevSeries(pmax = 0) are the same kernel, also mixed along x
    mix = x -> Species(κT = -0.3, response = x < 5 ? Gyrokinetic() : GordeyevSeries(pmax = 0))
    @test operator(g, Model((mix, adiabatic_electrons()), Quasineutrality()), Wavevector(ky = 0.5, kz = 0.01), 0.02 + 0.01im) ≈
          operator(g, Model((Species(κT = -0.3, response = Gyrokinetic()), adiabatic_electrons()), Quasineutrality()),
                   Wavevector(ky = 0.5, kz = 0.01), 0.02 + 0.01im) rtol = 1e-14
    # uniform T: one table per harmonic, shared by all nodes
    @test size(KD.WeberOperator(g, Wavevector(ky = 1.0), [Species() for _ in g.x], 2).W0[1], 1) == 1
end

@testset "local: DriftKinetic and Cold (electrostatic), Gyrokinetic and Boltzmann (electromagnetic)" begin
    # the electrostatic responses are -k·χ·k of the electromagnetic tensors (zero gradient), every response, both signs
    for k in (Wavevector(0.4, 0.7, 0.1), Wavevector(0.0, 1.1, 0.3)), q in (1, -1),
        r in (Cold(), DriftKinetic(), DriftKinetic(polarization = true), Gyrokinetic(), Boltzmann(), GordeyevSeries())
        s, ω, kv = Species(; q, m = q > 0 ? 1.0 : 0.3, T = 0.7, N = 0.8, response = r), 0.3 + 0.02im, [k.kx, k.ky, k.kz]
        @test charge_response(s, ω, k) ≈ -transpose(kv) * susceptibility(s, ω, k) * kv rtol = 1e-13
    end
    # drift kinetic = gyrokinetic at k⊥ → 0 with the same ω_*
    ky, ω = 0.8, 0.05 + 0.01im
    dk = Species(q = -1, m = 0.01, T = 0.5, κn = 0.2, κT = -0.4, response = DriftKinetic())
    gk = Species(q = -1, m = 0.01, T = 0.5, κn = 0.2ky / 1e-9, κT = -0.4ky / 1e-9, response = Gyrokinetic())
    @test charge_response(dk, ω, Wavevector(; ky, kz = 0.2)) ≈ charge_response(gk, ω, Wavevector(ky = 1e-9, kz = 0.2)) rtol = 1e-13
    # cold = full-orbit kinetic as T → 0, also with a density gradient (both advect it by the E×B drift only)
    k = Wavevector(0.4, 0.7, 0.1)
    for ω in (0.3 + 0.02im, 1.7 + 0.05im), q in (1, -1), κn in (0.0, 0.3)
        @test charge_response(Species(; q, κn, response = Cold()), ω, k) ≈
              charge_response(Species(; q, κn, T = 1e-8, response = GordeyevSeries()), ω, k) rtol = 1e-6
    end
    # gyrokinetic = full orbit to O(ω²) for ω ~ k∥v_t → 0 (electromagnetic, lab frame: both charge signs, k⊥ off the axes)
    gkerr(w, q) = (kk = Wavevector(0.9cos(0.7), 0.9sin(0.7), w); ω = w * (1 + 0.3im);
                   a = susceptibility(Species(; q), ω, kk);
                   maximum(abs, a - susceptibility(Species(; q, response = Gyrokinetic()), ω, kk)) / maximum(abs, a))
    for q in (1, -1)
        @test gkerr(1e-3, q) < 2e-6
        @test gkerr(1e-3, q) < 2e-4 * gkerr(1e-1, q)
    end
    # (1 - Γ₀)/b: Taylor series and the direct form agree across the switch
    P1, PT1 = KD._one_minus_Gamma0_over_b(prevfloat(0.5))
    P2, PT2 = KD._one_minus_Gamma0_over_b(0.5)
    @test P1 ≈ P2 rtol = 1e-14
    @test PT1 ≈ PT2 rtol = 1e-13
    @test all(KD._one_minus_Gamma0_over_b(1e-3) .≈ (0.9992504164844406, -0.0007491672132792669))   # BigFloat reference
    @test KD._one_minus_Gamma0_over_b(0.0) == (1, 0)
    # Boltzmann = drift-kinetic electrons as ζ = ω/(√2 k∥v_t) ∝ √m → 0: every element of χ, O(ζ)
    bz(m) = (kk = Wavevector(0.3, 0.5, 0.2); w = 0.05 + 0.01im;
             (susceptibility(Species(q = -1, m = m, response = Boltzmann()), w, kk), susceptibility(Species(q = -1, m = m, response = DriftKinetic()), w, kk)))
    bzerr(m) = (ab = bz(m); maximum(abs, ab[1] - ab[2]) / maximum(abs, ab[2]))
    @test bzerr(1e-6) < 1e-2
    @test bzerr(1e-6) < 0.12 * bzerr(1e-4)
    # finite β: Darwin roots with Boltzmann electrons = drift-kinetic electrons at m → 0 (cold ions); the electrons'
    # Landau damping, Im ω ~ ζ, is what Boltzmann drops
    kd = Wavevector(0.0, 0.4, 0.05)
    for β in (0.1, 1.0)
        ωb, okb = find_root(Model((Species(response = Cold()), Species(q = -1, response = Boltzmann())), Darwin(beta = β)), kd, 0.045 + 0.0im)
        ωd, okd = find_root(Model((Species(response = Cold()), Species(q = -1, m = 1e-8, response = DriftKinetic())), Darwin(beta = β)), kd, ωb)
        @test okb && okd
        @test real(ωb) ≈ real(ωd) rtol = 1e-8
        @test abs(imag(ωd)) < 1e-4 * abs(ωd)
    end
    # β → 0: the electromagnetic root tends to the electrostatic one, linearly in β (no gradients: an ion acoustic wave)
    ku = Wavevector(ky = 0.3, kz = 0.1)
    for ele in (Species(q = -1, T = 5.0, response = Boltzmann()), Species(q = -1, m = 1 / 1836, T = 5.0, response = DriftKinetic()))
        ion = Species(response = Gyrokinetic())
        es, ok = find_root(Model((ion, ele), Quasineutrality()), ku, 0.25 - 0.01im)
        d(β) = abs(first(find_root(Model((ion, ele), Darwin(beta = β)), ku, es; δ = 1e-4)) - es)
        @test ok && d(1e-5) < 1e-5 && d(1e-7) < 0.02 * d(1e-5)
    end
end

@testset "electromagnetic: gradients" begin
    # the longitudinal row is the full charge: -k·χ̃·k = charge_response, every response, both signs, with gradients
    ω, k = 0.6 + 0.08im, Wavevector(0.3, 0.7, 0.2)
    kv = [k.kx, k.ky, k.kz]
    for r in (GordeyevSeries(), GordeyevIntegral(), Gyrokinetic(), DriftKinetic(), DriftKinetic(polarization = true), Cold(), Boltzmann()),
        q in (1, -1)
        s = Species(; q, m = q > 0 ? 1.0 : 0.05, T = 0.8, N = 0.9, κn = 0.3, κT = r isa Cold ? 0.0 : -0.4, response = r)
        @test -(transpose(kv) * susceptibility(s, ω, k) * kv) ≈ charge_response(s, ω, k) rtol = 1e-13
    end
    @test_throws ArgumentError susceptibility(Species(κT = 0.1, response = Cold()), ω, k)
    # T∂_T M = ½ d/ds M(ω, s k) (central difference, h² ~ 1e-8)
    kf = Wavevector(0.9, 0.0, 0.3)
    for r in (GordeyevSeries(), GordeyevIntegral(), Gyrokinetic(), DriftKinetic())
        M, MT = susceptibility_T(r, ω, kf)
        Ms(s) = first(susceptibility_T(r, ω, Wavevector(s * kf.kx, 0.0, s * kf.kz)))
        @test maximum(abs, MT - (Ms(1 + 1e-4) - Ms(1 - 1e-4)) / 4e-4) < 1e-7 * maximum(abs, MT)
    end
    # the fast M-only paths equal the first tensor of susceptibility_T
    for r in (GordeyevSeries(), GordeyevIntegral())
        @test susceptibility(r, ω, kf) ≈ first(susceptibility_T(r, ω, kf)) rtol = 1e-14
    end
    # series with gradient drive against the orbit integral, whose T-derivative is taken inside the integrand
    for kz in (0.3, -0.3)
        a = susceptibility(GordeyevSeries(), ω, Wavevector(0.9, 0.0, kz), 0.2, -0.3)
        @test maximum(abs, a - susceptibility(GordeyevIntegral(), ω, Wavevector(0.9, 0.0, kz), 0.2, -0.3)) < 1e-12 * maximum(abs, a)
    end
    # the Gordeyev series does not allocate
    susceptibility(GordeyevSeries(), ω, kf, 0.2, -0.3)
    @test (@allocated susceptibility(GordeyevSeries(), ω, kf, 0.2, -0.3)) == 0
    # β → 0: the electromagnetic root of an ITG case tends to the electrostatic one, linearly in β
    ku = Wavevector(ky = 0.5, kz = 0.1)
    for ele in (Species(q = -1, κn = -0.1, response = Boltzmann()),
                Species(q = -1, m = 1 / 1836, κn = -0.1, κT = -0.4, response = DriftKinetic()))
        ion = Species(κn = -0.1, κT = -0.4, response = Gyrokinetic())
        es, ok = find_root(Model((ion, ele), Quasineutrality()), ku, -0.1 + 0.05im)
        d(β) = abs(first(find_root(Model((ion, ele), Darwin(beta = β)), ku, es; δ = 1e-4)) - es)
        @test ok && d(1e-5) < 1e-6 && d(1e-7) < 0.02 * d(1e-5)
    end
end

@testset "local models: overlap limits" begin
    # Each limit: the relative difference at ε = 1e-3 is small and drops at least like ε^p from ε = 1e-2, without and with
    # gradients (g = 0, 1; κn = 0.3, κT = -0.4 where the model has them). A ← B reads "A is the limit of B".
    relerr(a::Number, b::Number) = abs(a - b) / abs(b)
    relerr(a::AbstractArray, b::AbstractArray) = maximum(abs, a - b) / maximum(abs, b)
    function converges(f, p; tol)
        e2, e3 = relerr(f(1e-2)...), relerr(f(1e-3)...)
        return e3 < tol && e2 / e3 > 10^(p - 0.2)
    end
    me = 1 / 1836
    k0, ω0 = Wavevector(0.3, 0.7, 0.2), 0.6 + 0.08im
    sp(r, g; s = 1.0, q = 1, m = q > 0 ? 1.0 : me, T = 1.0) = Species(; q, m, T, κn = 0.3g * s, κT = -0.4g * s, response = r)
    for g in (0, 1)
        # Gyrokinetic ← full orbit for ω, k∥v_t, ω_* ~ ε ≪ Ω (electrostatic and electromagnetic)
        kg(ε) = Wavevector(0.3, 0.7, 0.2ε)
        @test converges(ε -> (charge_response(sp(Gyrokinetic(), g; s = ε), ε * ω0, kg(ε)), charge_response(sp(GordeyevSeries(), g; s = ε), ε * ω0, kg(ε))), 2; tol = 1e-6)
        @test converges(ε -> (susceptibility(sp(Gyrokinetic(), g; s = ε), ε * ω0, kg(ε)), susceptibility(sp(GordeyevSeries(), g; s = ε), ε * ω0, kg(ε))), 2; tol = 1e-6)
        # DriftKinetic ← Gyrokinetic for k⊥ρ ~ ε at fixed ω_*; with polarisation at k∥ = 0 (fluid, W = 1 - ω_*/ω)
        kd(ε, kz) = Wavevector(0.3ε, 0.7ε, kz)
        @test converges(ε -> (charge_response(sp(DriftKinetic(), g; s = 1 / ε), ω0, kd(ε, 0.2)), charge_response(sp(Gyrokinetic(), g; s = 1 / ε), ω0, kd(ε, 0.2))), 2; tol = 1e-5)
        @test converges(ε -> (charge_response(Species(κn = 0.3g / ε, response = DriftKinetic(polarization = true)), ω0, kd(ε, 0.0)),
                              charge_response(Species(κn = 0.3g / ε, response = Gyrokinetic()), ω0, kd(ε, 0.0))), 2; tol = 1e-6)
        # DriftKinetic ← full-orbit electrons, m → 0 at fixed ζ, electromagnetic
        km(m) = Wavevector(0.5, 0.3, 0.05 * sqrt(m / 1e-2))
        @test converges(ε -> (m = 1e-2ε^2; (susceptibility(sp(DriftKinetic(), g; q = -1, m), 0.3 + 0.01im, km(m)),
                                            susceptibility(sp(GordeyevSeries(), g; q = -1, m), 0.3 + 0.01im, km(m)))), 2; tol = 1e-8)
        # Cold ← DriftKinetic(polarization), T → 0 and ω, k∥ ~ ε ≪ Ω (both Boussinesq, so any k_x)
        kc(ε) = Wavevector(0.3, 0.7, 0.2ε)
        # (T_dk = 1e-8 electrostatically, where q²N/T cancels; 1e-12 in EM, where ω_* ∝ T_dk sets a floor ~ T_dk/ε)
        dkc(g, T) = Species(; T, κn = 0.3g, response = DriftKinetic(polarization = true))
        @test converges(ε -> (charge_response(Species(κn = 0.3g, response = Cold()), ε * ω0, kc(ε)), charge_response(dkc(g, 1e-8), ε * ω0, kc(ε))), 2; tol = 1e-6)
        @test converges(ε -> (susceptibility(Species(κn = 0.3g, response = Cold()), ε * ω0, kc(ε)), susceptibility(dkc(g, 1e-12), ε * ω0, kc(ε))), 2; tol = 1e-6)
        # Boltzmann ← full-orbit electrons for ζ = ω/(k∥v_t) and ω_*/(k∥v_t) ~ ε (the gradient drive is non-adiabatic)
        @test converges(ε -> (charge_response(sp(Boltzmann(), g; s = ε, q = -1), ε * ω0, k0), charge_response(sp(GordeyevSeries(), g; s = ε, q = -1), ε * ω0, k0)), 1; tol = 3e-4)
        # Boltzmann ← drift-kinetic electrons, electromagnetic, ζ, ω_* ~ ε: every element but the δB∥-δB∥ one (lab xx for k⊥
        # along y), the transit-time damping 2b ζZ(ζ), O(k⊥²ρ²) and absent from a fluid; and the density response to an
        # inductive E∥ (magnetic flutter, δB_x ≠ 0)
        kb = Wavevector(0.0, 0.5, 0.2)
        ωb(ε) = ε * (0.05 + 0.01im)
        bd(r, g, ε) = (χ = susceptibility(sp(r, g; s = ε, q = -1), ωb(ε), kb); [χ[i] for i in 2:9])
        @test converges(ε -> (bd(Boltzmann(), g, ε), bd(DriftKinetic(), g, ε)), 1; tol = 1e-4)
        ρind(r, ε) = (s = sp(r, g; s = ε, q = -1); kv = [kb.kx, kb.ky, kb.kz]; E = [0, 0, 1];
                      -im * transpose(kv) * susceptibility(s, ωb(ε), kb) * E + transpose(KD.charge_vector(s, ωb(ε), kb)) * E)
        @test converges(ε -> (ρind(Boltzmann(), ε), ρind(DriftKinetic(), ε)), 1; tol = 3e-5)
        # Boltzmann ← Gyrokinetic, ions and electrons, ζ, ω_* ~ ε: electrostatically at any k⊥ρ (here 0.76 for ions);
        # electromagnetically also k⊥ρ ~ ε (the parallel response carries Γ₀), every element but the polarisation χ_xx, χ_yy
        # of the gyrokinetic species, which a massless fluid lacks
        kbg(ε) = Wavevector(0.0, 0.5ε, 0.2)
        par(r, q, ε) = (χ = susceptibility(sp(r, g; s = ε, q), ωb(ε), kbg(ε)); [χ[i, j] for i in 1:3, j in 1:3 if i != j || i == 3])
        for q in (1, -1)
            @test converges(ε -> (charge_response(sp(Boltzmann(), g; s = ε, q), ε * ω0, k0), charge_response(sp(Gyrokinetic(), g; s = ε, q), ε * ω0, k0)), 1; tol = 3e-3)
            @test converges(ε -> (par(Boltzmann(), q, ε), par(Gyrokinetic(), q, ε)), 1; tol = 5e-4)
        end
    end
    # Cold ← full orbit, T ~ ε, with and without a density gradient (both advect it by the E×B drift only), every element
    for κn in (0.0, 0.3)
        @test converges(ε -> (charge_response(Species(; κn, response = Cold()), ω0, k0), charge_response(Species(; T = ε, κn), ω0, k0)), 1; tol = 1e-3)
        @test converges(ε -> (susceptibility(Species(; κn, response = Cold()), ω0, k0), susceptibility(Species(; T = ε, κn), ω0, k0)), 1; tol = 3e-3)
    end
    # field equations: Maxwell → Darwin as λ² → 0 (det), and the Maxwell root → the Poisson root as β → 0 (gradients)
    mfield(f) = Model((Species(κn = 0.3, κT = -0.4), Species(q = -1, m = me, κn = 0.3, response = DriftKinetic())), f)
    @test converges(ε -> (dispersion(mfield(Maxwell(beta = 0.01, lambda2 = ε)), ω0, k0), dispersion(mfield(Darwin(beta = 0.01)), ω0, k0)), 1; tol = 1e-3)
    itg = (Species(κn = -0.1, κT = -0.4, response = Gyrokinetic()), Species(q = -1, m = me, κn = -0.1, κT = -0.4, response = DriftKinetic()))
    ki = Wavevector(ky = 0.5, kz = 0.1)
    es, = find_root(Model(itg, Poisson(lambda2 = 0.01)), ki, -0.1 + 0.05im)
    @test converges(ε -> (first(find_root(Model(itg, Maxwell(beta = ε^2, lambda2 = 0.01)), ki, es; δ = 1e-4)), es), 2; tol = 1e-6)
end

@testset "radial: species without Larmor radius (DriftKinetic, Cold)" begin
    # uniform profiles: diagonal on the basis, entries the local D at k_x = k_m, every bc
    ω, k = 0.3 + 0.02im, Wavevector(ky = 0.7, kz = 0.1)
    for bc in (:periodic, :mirror, :dirichlet), r in (Cold(), DriftKinetic(polarization = true))
        g = radial_grid(-5, 5, 16; bc)
        m = Model((Species(T = 0.7, response = r), adiabatic_electrons()), Quasineutrality())
        B = g.Cinv * operator(g, m, k, ω) * g.C
        @test maximum(abs, B - Diagonal(diag(B))) < 1e-12
        @test diag(B) ≈ [dispersion(m, ω, Wavevector(q, k.ky, k.kz)) for q in g.k] rtol = 1e-13
    end
    # radial cold = radial full-orbit kinetic (exact for guiding-centre profiles) as T → 0, with a gradient, after adding
    # what the local Boussinesq/E×B model drops from the exact cold drift v_x = w(E_y - iωE_x/Ω)/B: the polarisation
    # current across the gradient, w(X_exact - X), X_exact = -∂x c ∂x = -[∂²c + c∂² - c'']/2, X = -c∂², and the
    # non-E×B part of the drive, (w - 1) q N κn k_y/ω (resolved modes; at the grid's highest wavenumbers the kinetic
    # orbit phase is aliased). This checks the polarisation operator and the gradient drive.
    L = 4π
    n(x) = exp(0.3sin(2π * x / L))
    g = radial_grid(0, L, 32; bc = :periodic)
    lo = findall(q -> abs(q) < 3, g.k)
    D2 = real.(g.C * Diagonal(-g.k .^ 2) * g.Cinv)
    c, c2 = n.(g.x), [(n(x + 1e-3) - 2n(x) + n(x - 1e-3)) / 1e-6 for x in g.x]
    dX = (Diagonal(c2) - D2 * Diagonal(c) + Diagonal(c) * D2) / 2
    for ω in (0.3 + 0.02im, 1.7 + 0.05im), q in (1, -1)
        Bc = g.Cinv * operator(g, Model((species_profile(; N = n, q, response = Cold()),), Quasineutrality()), k, ω) * g.C
        Bk = g.Cinv * operator(g, Model((species_profile(; N = n, q, T = x -> 1e-6, response = GordeyevSeries(pmax = 1)),),
                                        Quasineutrality()), k, ω) * g.C
        w = 1 / (1 - ω^2)
        drive = Diagonal([(w - 1) * q * n(x) * 0.3 * (2π / L) * cos(2π * x / L) * k.ky / ω for x in g.x])
        Bx = Bc + g.Cinv * (w * dX + drive) * g.C
        @test maximum(abs, Bx[lo, lo] - Bk[lo, lo]) < 1e-5 * maximum(abs, Bx[lo, lo])
        @test maximum(abs, Bc[lo, lo] - Bk[lo, lo]) > 1e-2 * maximum(abs, Bx[lo, lo])      # the dropped terms are visible
    end
    # end to end: drift-kinetic ions with polarisation, uniform, return the local root
    m = Model((Species(κn = -0.3, response = DriftKinetic(polarization = true)), adiabatic_electrons()), Quasineutrality())
    ku = Wavevector(ky = 0.5, kz = 0.05)
    loc, ok = find_root(m, ku, 0.1 + 0.01im)
    r = eigenmode(radial_grid(-10, 10, 16; bc = :periodic), m, ku, loc * (1 + 1e-3))
    @test ok && r.converged
    @test r.ω ≈ loc rtol = 1e-8
    # the convergence scale does not grow with the resolution, and is O(1) without polarisation
    sc(N, r) = KD.RadialWorkspace(radial_grid(-10, 10, N; bc = :periodic),
                                  Model((Species(response = r), adiabatic_electrons()), Quasineutrality()), ku).scale
    @test sc(16, DriftKinetic(polarization = true)) == sc(64, DriftKinetic(polarization = true))
    @test sc(64, DriftKinetic()) == 2
    # guards: one response at every x; a guess on a pole
    mix = x -> Species(response = x < 2 ? Cold() : DriftKinetic())
    @test_throws ArgumentError operator(g, Model((mix,), Quasineutrality()), k, ω)
    @test_throws ArgumentError eigenmode(g, Model((Species(response = Cold()),), Quasineutrality()), k, 1.0)
end

@testset "magnetic drift: quadratic average" begin
    brute(n0, n2, a, b, e) = quadgk(s -> exp(-s^2 / 2) / sqrt(2π) * (n0 + n2 * s^2) / (a * s^2 - b * s + e),
                                    -12, 12; rtol = 1e-12, atol = 0)[1]
    for (a, b, e) in ((0.01, 0.02, 0.01 + 0.01im), (-0.02, 0.05, -0.03 + 0.002im),
                      (0.05, 0.0, 0.02 + 0.02im), (0.01, 0.02, -0.04 + 0.01im))
        @test KD.quadratic_average(0.3 + 0.1im, -0.15, a, b, e) ≈ brute(0.3 + 0.1im, -0.15, a, b, e) rtol = 1e-10
    end
    # a → 0 is the linear-denominator moments, without cancellation at small a
    I0, I2 = KD.linear_moments(0.02 + 0.01im, 0.03)
    @test KD.quadratic_average(0.3, 0.1, 1e-9, 0.03, 0.02 + 0.01im) ≈ 0.3 * I0 + 0.1 * I2 rtol = 1e-6
    for a in (1e-12, 4e-8, 1e-7, 1e-6, -1e-6)
        @test KD.quadratic_average(0.3 + 0.1im, -0.15, a, 0.3, 0.1 + 0.05im) ≈
              brute(0.3 + 0.1im, -0.15, a, 0.3, 0.1 + 0.05im) rtol = 1e-12
    end
    # linear moments below the real axis are Z's continuation, also past the asymptotic switch |ζ| = 6
    direct(ω, q) = (I0 = -plasma_Z(ω / (sqrt(2) * q)) / (sqrt(2) * q); (I0, (ω / q)^2 * I0 - ω / q^2))
    for ζ in (5.999, 6.001, 9.0), arg in (-0.05π, -0.3π, -0.45π)
        ω = sqrt(2) * 0.01 * ζ * cis(arg)
        @test all(isapprox.(KD.linear_moments(ω, 0.01), direct(ω, 0.01); rtol = 1e-11))
    end
    @test KD.linear_moments(-0.1im, 0.01)[1] ≈ direct(-0.1im, 0.01)[1] rtol = 1e-12
    # analytic across Im e = 0: complex derivatives along the real and imaginary direction agree
    f(e) = KD.quadratic_average(0.3, 0.1, 0.01, 0.02, e)
    for e0 in (0.02 + 0.0im, 0.03 - 0.004im, -0.01 + 0.0im)
        h = 1e-6
        @test (f(e0 + h) - f(e0 - h)) / 2h ≈ (f(e0 + im * h) - f(e0 - im * h)) / (2im * h) rtol = 1e-5
    end
end

@testset "magnetic drift: local response" begin
    ky, R0, ω = 0.5, 50.0, -0.03 + 0.01im
    k = Wavevector(ky = ky, kz = 0.02)
    for alpha in (0.0, 1.0)
        s = Species(κn = -0.04, κT = -0.16, response = GyrokineticDrift(c = 1 / R0, alpha = alpha, nmu = 200))
        # independent: direct double quadrature in (s, μ), Im ω > 0
        W = quadgk(μ -> exp(-μ) * KD.besselj0(ky * sqrt(2μ))^2 * quadgk(v -> begin
                ωD = -ky * (v^2 + μ) / R0
                ωs = ky * (-0.04 + -0.16 * (μ + v^2 / 2 - 1.5))
                exp(-v^2 / 2) / sqrt(2π) * (ω - alpha * ωD - ωs) / (ω - k.kz * v - ωD)
            end, -12, 12; rtol = 1e-11)[1], 0, 40; rtol = 1e-9)[1]
        @test nonadiabatic_response(s, ω, k) ≈ W rtol = 1e-6
    end
    # kinetic electrons: ω_D ∝ T_s/q_s, i.e. sgn(q) enters the rescaling to own units
    e = Species(q = -1, m = 1 / 100, T = 2.0, κn = -0.04, κT = -0.16,
                response = GyrokineticDrift(c = 0.02, cperp = 0.03, alpha = 1.0, nmu = 800))
    ke, ωe, vt, ρ = Wavevector(ky = 0.5, kz = 0.002), 0.03 + 0.01im, sqrt(200.0), sqrt(0.02)
    We = quadgk(μ -> exp(-μ) * KD.besselj0(ke.ky * ρ * sqrt(2μ))^2 * quadgk(v -> begin
            ωD = 2 * ke.ky * (0.02 * v^2 + 0.03 * μ)              # -k_y (T/q)(c s² + c⊥ μ), T/q = -2
            ωs = -2 * ke.ky * (-0.04 + -0.16 * (μ + v^2 / 2 - 1.5))
            exp(-v^2 / 2) / sqrt(2π) * (ωe - ωD - ωs) / (ωe - ke.kz * vt * v - ωD)
        end, -14, 14; rtol = 1e-12)[1], 0, 5, 20, 60; rtol = 1e-11)[1]
    @test nonadiabatic_response(e, ωe, ke) ≈ We rtol = 1e-6
    # c → 0 is Gyrokinetic
    g = Species(κT = -0.3, response = Gyrokinetic())
    d = Species(κT = -0.3, response = GyrokineticDrift(c = 1e-12, nmu = 32))
    @test nonadiabatic_response(d, ω, k) ≈ nonadiabatic_response(g, ω, k) rtol = 1e-9
    # toroidal ITG roots (R/L_T = 8, R/L_n = 2, adiabatic electrons), regression values of the snapshot b5fc4f4
    for (alpha, ref) in ((0.0, -0.0289898331 + 0.0130943411im), (1.0, -0.0216328565 + 0.0162489625im))
        ion = Species(κn = -2 / 50, κT = -8 / 50, response = GyrokineticDrift(c = 1 / 50, alpha = alpha, nmu = 160))
        m = Model((ion, adiabatic_electrons()), Quasineutrality())
        ω0 = find_modes(m, Wavevector(ky = 0.5); re = -0.06:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)[1]
        @test ω0 ≈ ref atol = 1e-8
    end
    # the prescribed drift (alpha ≠ 0) is electrostatic only
    @test_throws ArgumentError dispersion(Model((Species(response = GyrokineticDrift(c = 0.02, alpha = 1.0)), Species(q = -1, m = 1 / 1836)),
                                                Darwin(beta = 0.01)), 0.1 + 0.01im, k)
end

@testset "magnetic drift: radial operator" begin
    # c = 0 with enough μ nodes is the closed-form Gyrokinetic kernel: nonuniform N, T, all three bases
    N, T = x -> 1 + 0.3 * sin(2π * x / 30), x -> exp(-0.4 * sin(2π * x / 30))
    k = Wavevector(ky = 0.7, kz = 0.02)
    el = species_profile(; q = -1, N, response = Boltzmann())
    mw = Model((species_profile(; N, T, response = Gyrokinetic()), el), Quasineutrality())
    mq = Model((species_profile(; N, T, response = GyrokineticDrift(c = 0.0, nmu = 128)), el), Quasineutrality())
    for bc in (:dirichlet, :mirror, :periodic)
        g = radial_grid(-15, 15, 40; bc)
        Aw, Aq = operator(g, mw, k, -0.03 + 0.01im), operator(g, mq, k, -0.03 + 0.01im)
        @test maximum(abs.(Aw - Aq)) < 1e-12 * maximum(abs.(Aw))
    end
    # uniform drift: the radial eigenvalue is the local root, α = 0 and 1
    mg = radial_grid(-20, 20, 40; bc = :mirror)
    k0 = Wavevector(ky = 0.5)
    for alpha in (0.0, 1.0)
        md = Model((Species(κT = -8 / 50, κn = -2 / 50, response = GyrokineticDrift(c = 1 / 50, alpha = alpha)),
                    adiabatic_electrons()), Quasineutrality())
        loc = find_modes(md, k0; re = -0.06:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)[1]
        r = eigenmode(mg, md, k0, loc * 1.02)
        @test r.converged
        @test r.ω ≈ loc rtol = 1e-9
    end
    # A is analytic through the real axis (Im ω < 0 is the continuation)
    md = Model((Species(κT = -8 / 50, response = GyrokineticDrift(c = 1 / 50)), adiabatic_electrons()), Quasineutrality())
    ω0, h = -0.0123 + 0.0im, 1e-7
    dA_re = (operator(mg, md, k0, ω0 + h) - operator(mg, md, k0, ω0 - h)) / 2h
    dA_im = (operator(mg, md, k0, ω0 + im * h) - operator(mg, md, k0, ω0 - im * h)) / (2im * h)
    @test LinearAlgebra.norm(dA_re - dA_im) < 1e-4 * LinearAlgebra.norm(dA_re)
    # profiles of the lab drift coefficient c(x) and kinetic electrons with drift: uniform limit is the local root
    ion = Species(κn = -2 / 50, κT = -8 / 50, response = GyrokineticDrift(c = 1 / 50, nmu = 160))
    ele = Species(q = -1, m = 1 / 400, κn = -2 / 50, κT = -8 / 50, response = GyrokineticDrift(c = 1 / 50, nmu = 160))
    m2 = Model((ion, ele), Quasineutrality())
    kk = Wavevector(ky = 0.5, kz = 0.01)
    loc = find_modes(m2, kk; re = -0.08:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)[end]
    r = eigenmode(mg, m2, kk, loc * 1.02)
    @test r.converged
    @test r.ω ≈ loc rtol = 1e-8
    sp = species_profile(; N, T, response = x -> GyrokineticDrift(c = 1 / (50 + x)))
    @test operator(mg, Model((sp, el), Quasineutrality()), kk, -0.02 + 0.01im) isa Matrix{ComplexF64}
    # guard: the μ nodes must agree along x
    mix = x -> Species(response = GyrokineticDrift(c = 0.02, nmu = x < 0 ? 32 : 64))
    @test_throws ArgumentError operator(mg, Model((mix, adiabatic_electrons()), Quasineutrality()), kk, 0.02 + 0.01im)
end

# Gyrokinetic field equations in (φ, A∥, δB∥) for GyrokineticDrift species, built independently of the tensor:
# h = (qF/T)(ω - ω_*)/(ω - k∥v∥ - ω_D) ⟨χ⟩, ⟨χ⟩ = J0(φ - v∥A∥) + (T/q) 2μ (J1/z) δB∥;
# Σ q(-(qN/T)φ + ∫J0 h) = 0, k⊥² A∥ = (β/2) Σ q∫v∥J0 h, δB∥ = -(β/2) Σ ∫ m v⊥² (J1/z) h
function gk_field_matrix(sp, β, ω, k)
    G = zeros(ComplexF64, 3, 3)
    G[2, 2], G[3, 3] = k.kx^2 + k.ky^2, 1
    for s in sp
        r = KD.own_units(s.response, s)
        Ω, vt = abs(s.q) / s.m, sqrt(s.T / s.m)
        ρ = vt / Ω
        ω′, kp, kz, ky = ω / Ω, hypot(k.kx, k.ky) * ρ, k.kz * ρ, k.ky * ρ
        ωn, ωT = (s.T / s.q) * k.ky * s.κn / Ω, (s.T / s.q) * k.ky * s.κT / Ω
        G[1, 1] -= s.q^2 * s.N / s.T
        for (μ, w) in zip(r.mu, r.wmu)
            sv = range(-12, 12, length = 2401)            # s average by the trapezoidal rule (exponentially accurate), Im ω > 0
            z = kp * sqrt(2μ)
            J0, J1 = KD.besselj0(z), KD.besselj1(z)
            for (i, mom) in enumerate((v -> J0, v -> J0 * vt * v, v -> 2s.T * μ * J1 / z)),
                (j, χ) in enumerate((v -> J0, v -> -J0 * vt * v, v -> (s.T / s.q) * 2μ * J1 / z))
                pre = (s.q, -β / 2 * s.q, β / 2)[i]
                g(v) = exp(-v^2 / 2) / sqrt(2π) * mom(v) * χ(v) * (ω′ - ωn - ωT * (μ + v^2 / 2 - 1.5)) /
                       (ω′ - kz * v + ky * (r.c * v^2 + r.cperp * μ))
                G[i, j] += pre * w * (s.q * s.N / s.T) * step(sv) * sum(g, sv)
            end
        end
    end
    return G
end

@testset "magnetic drift: electromagnetic" begin
    # ⟨s^j/(a s² - b s + e)⟩ up to s⁶, both signs of b, linear and quadratic denominators
    for (a, b, e) in ((0.0, 0.03, 0.02 + 0.01im), (0.0, -0.03, 0.02 + 0.01im), (0.01, -0.02, 0.01 + 0.01im), (-0.02, 0.05, -0.03 + 0.002im))
        K = KD.average_kernel(a, b, e)
        for j in 0:6
            @test K[j + 1] ≈ quadgk(s -> exp(-s^2 / 2) / sqrt(2π) * s^j / (a * s^2 - b * s + e), -14, 14; rtol = 1e-12)[1] rtol = 1e-10
        end
    end
    for k in (Wavevector(kx = 0.2, ky = 0.5, kz = 0.02), Wavevector(kx = -0.3, ky = -0.4, kz = -0.03)), ω in (-0.03 + 0.01im, 0.05 - 0.004im),
        (q, m) in ((1, 1.0), (-1, 1 / 100)), (κn, κT) in ((0.0, 0.0), (-0.04, -0.16))
        # c = 0 is the electromagnetic Gyrokinetic tensor
        χg = susceptibility(Species(; q, m, κn, κT, response = Gyrokinetic()), ω, k)
        @test susceptibility(Species(; q, m, κn, κT, response = GyrokineticDrift(c = 0.0, nmu = 128)), ω, k) ≈ χg rtol = 1e-12
        # with drift the longitudinal row is the electrostatic GyrokineticDrift charge (continuity closed by the drift current)
        d = Species(; q, m, κn, κT, response = GyrokineticDrift(c = 0.02, cperp = 0.03, nmu = 64))
        kv = [k.kx, k.ky, k.kz]
        @test -(transpose(kv) * susceptibility(d, ω, k) * kv) ≈ charge_response(d, ω, k) rtol = 1e-12
    end
    # Darwin with the tensor against the gyrokinetic field equations (independent) under the gyrokinetic ordering:
    # ω, k∥, gradients, drift ∝ ε, the difference O(ω/Ω, ρ/L) → linear in ε (finite β: δB∥ matters, k_x ≠ 0)
    sp(ε) = (Species(κn = -0.04ε, κT = -0.16ε, response = GyrokineticDrift(c = 0.02ε, nmu = 48)),
             Species(q = -1, m = 1 / 400, κn = -0.04ε, κT = -0.16ε, response = GyrokineticDrift(c = 0.02ε, nmu = 48)))
    β, ω1 = 0.03, -0.058906 + 0.043717im                 # unstable; dropping δB∥ moves it by 10%
    Δ = map((0.5, 0.25)) do ε
        k = Wavevector(kx = 0.2, ky = 0.5, kz = 0.01ε)
        wG, okG = find_root(ω -> LinearAlgebra.det(gk_field_matrix(sp(ε), β, ω, k)), ω1 * ε; δ = 1e-3ε)
        wD, okD = find_root(ω -> dispersion(Model(sp(ε), Darwin(beta = β)), ω, k), wG; δ = 1e-4ε)
        @test okG && okD
        abs(wD - wG) / abs(wG)
    end
    @test Δ[1] < 2e-3
    @test 0.45 < Δ[2] / Δ[1] < 0.55
    # β → 0: the Darwin root is the electrostatic one, linearly in β
    ion = Species(κn = -0.04, κT = -0.16, response = GyrokineticDrift(c = 0.02))
    ele = Species(q = -1, m = 1 / 400, κn = -0.04, κT = -0.16, response = GyrokineticDrift(c = 0.02))
    k = Wavevector(ky = 0.5, kz = 0.01)
    es, = find_root(Model((ion, ele), Quasineutrality()), k, -0.0287 + 0.0036im)
    d = [abs(first(find_root(Model((ion, ele), Darwin(beta = β)), k, es; δ = 1e-5)) - es) for β in (1e-6, 1e-5)]
    @test 8 < d[2] / d[1] < 12
end

@testset "magnetic drift: pressure balance" begin
    sp = (Species(N = 1.0, T = 1.0, κn = -0.05, κT = -0.3), Species(q = -1, m = 1 / 100, N = 1.0, T = 2.0, κn = -0.05, κT = -0.1))
    @test pressure_balance_cperp(sp, 0.02) ≈ 0.01 * (1.0 * (-0.35) + 2.0 * (-0.15))
    # sign: ion orbit (Boris) in B(x) ẑ from pressure balance, B² = 1 - β (p(x) - p(0)), p = e^{κ_p x}: its ∇B drift
    # v_y equals ω_D/k_y of GyrokineticDrift, -c⊥ v⊥²/2 (T μ = v⊥²/2 at m = 1)
    β, κp = 0.02, -0.5
    cperp = pressure_balance_cperp((Species(κn = κp),), β)
    Bz(x) = sqrt(1 - β * (exp(κp * x) - 1))
    x, v, dt = [0.0, 0.0], [0.4, 0.0], 2π / 400
    for n in 1:400 * 50                                    # 50 gyroperiods, Boris rotation (q = m = 1, E = 0)
        b = Bz(x[1]) * dt / 2
        v = [v[1] + 2b / (1 + b^2) * (v[2] - b * v[1]), v[2] - 2b / (1 + b^2) * (v[1] + b * v[2])]
        x += v * dt
    end
    @test x[2] / (50 * 2π) ≈ -cperp * 0.4^2 / 2 rtol = 0.02
end
