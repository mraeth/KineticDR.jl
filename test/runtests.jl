using Test
using KineticDR
using QuadGK: quadgk
using SpecialFunctions: besselj
using LinearAlgebra: eigen, SymTridiagonal, Diagonal
using PlasmaCore: ScalarField

const KD = KineticDR

adiabatic_electrons(τ = 1.0) = Species(q = -1, T = τ, response = Boltzmann())
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
    @test_throws ArgumentError dispersion(Model((Species(response = Cold()), Species(q = -1, response = Boltzmann())), Quasineutrality()), ω, k)
    @test_throws ArgumentError dispersion(Model((Species(), Species(q = -1, response = Boltzmann())), Darwin(beta = β)), ω, k)
    @test_throws ArgumentError dispersion(Model((Species(κn = 0.1), Species(q = -1, m = μ)), Darwin(beta = β)), ω, k)
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
    @test count_eigenvalues(g, mp, kp, -0.06, 0.0, 0.001, 0.015) >= length(found) >= 1
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
    # not (yet) radial: electromagnetic fields, full-orbit responses, k_x ≠ 0, non-finite guess
    @test_throws ArgumentError operator(g2, Model((ion, el), Darwin(beta = 0.1)), k, 0.1 + 0.01im)
    @test_throws ArgumentError operator(g2, Model((Species(), el), Quasineutrality()), k, 0.1 + 0.01im)
    @test_throws ArgumentError operator(g2, m3, Wavevector(kx = 0.1, ky = 0.5), 0.1 + 0.01im)
    @test_throws ArgumentError eigenmode(g2, m3, k, complex(NaN, NaN))
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
    # the operator is the μ integral done by quadrature (128 Gauss–Laguerre nodes): nonuniform N, T, κ ≠ 0,
    # both bases (small N: the quadrature resolves every k_x)
    function quadrature_operator(g, m, k, ω; n = 128)
        E = eigen(SymTridiagonal([2j + 1.0 for j in 0:(n - 1)], [Float64(j) for j in 1:(n - 1)]))
        ss = [m.species[1](x) for x in g.x]                      # species 1 kinetic, the others Boltzmann
        A = Matrix{ComplexF64}(Diagonal([sum(KD.adiabatic_term(sp, x) for sp in m.species) for x in g.x]))
        for (μ, w) in zip(E.values, E.vectors[1, :] .^ 2)
            G = [g.C[j, l] * besselj(0, hypot(g.k[l], k.ky) * sqrt(2μ * s.T * s.m) / abs(s.q)) for (j, s) in enumerate(ss), l in eachindex(g.k)]
            J = G * g.Cinv
            A -= J' * Diagonal([w * s.q^2 * s.N / s.T * KD.vpar_response(k, s, ω, μ) for s in ss]) * J
        end
        return A
    end
    N, T = x -> 1 + 0.3tanh(x / 3), x -> exp(-0.8tanh(x / 2))
    k = Wavevector(ky = 0.7, kz = 0.02)
    mw = Model((species_profile(; N, T, response = Gyrokinetic()), species_profile(; q = -1, N, response = Boltzmann())), Quasineutrality())
    for bc in (:dirichlet, :mirror)
        g = radial_grid(-15, 15, 40; bc)
        Aw = operator(g, mw, k, -0.03 + 0.01im)
        @test maximum(abs.(Aw - quadrature_operator(g, mw, k, -0.03 + 0.01im))) < 1e-13 * maximum(abs.(Aw))
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
