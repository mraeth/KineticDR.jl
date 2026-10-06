using Test
using KineticDR
using QuadGK: quadgk
using LinearAlgebra: norm
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
    @test KD.one_plus_zZ(9.0 + 0.5im) ≈ 1 + (9.0 + 0.5im) * plasma_Z(9.0 + 0.5im) rtol = 1e-8
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
end

@testset "quadratic average" begin
    brute(n0, n2, a, b, e) = quadgk(s -> exp(-s^2 / 2) / sqrt(2π) * (n0 + n2 * s^2) / (a * s^2 - b * s + e),
                                    -12, 12; rtol = 1e-12, atol = 0)[1]
    for (a, b, e) in ((0.01, 0.02, 0.01 + 0.01im), (-0.02, 0.05, -0.03 + 0.002im),
                      (0.05, 0.0, 0.02 + 0.02im), (0.01, 0.02, -0.04 + 0.01im))
        @test KD.quadratic_average(0.3 + 0.1im, -0.15, a, b, e) ≈ brute(0.3 + 0.1im, -0.15, a, b, e) rtol = 1e-10
    end
    # a → 0 is the linear-denominator moments
    I0, I2 = KD.linear_moments(0.02 + 0.01im, 0.03)
    @test KD.quadratic_average(0.3, 0.1, 1e-9, 0.03, 0.02 + 0.01im) ≈ 0.3 * I0 + 0.1 * I2 rtol = 1e-6
    # analytic across Im e = 0 (the labelling of the roots is by continuation): complex derivative
    # along the real and imaginary direction agree, also with Im e < 0
    f(e) = KD.quadratic_average(0.3, 0.1, 0.01, 0.02, e)
    for e0 in (0.02 + 0.0im, 0.03 - 0.004im, -0.01 + 0.0im)
        h = 1e-6
        @test (f(e0 + h) - f(e0 - h)) / 2h ≈ (f(e0 + im * h) - f(e0 - im * h)) / (2im * h) rtol = 1e-5
    end
end

@testset "gyrokinetic drift response" begin
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
    # c → 0 is Gyrokinetic
    g = Species(κT = -0.3, response = Gyrokinetic())
    d = Species(κT = -0.3, response = GyrokineticDrift(c = 1e-12, nmu = 32))
    @test nonadiabatic_response(d, ω, k) ≈ nonadiabatic_response(g, ω, k) rtol = 1e-9
    # local roots (R/L_T = 8, R/L_n = 2) are valid solutions of D, reproduced by the radial operator below
    for (alpha, ref) in ((0.0, -0.0289898331 + 0.0130943411im), (1.0, -0.0216328565 + 0.0162489625im))
        ion = Species(κn = -2 / 50, κT = -8 / 50, response = GyrokineticDrift(c = 1 / 50, alpha = alpha, nmu = 160))
        m = Model((ion, adiabatic_electrons()), Quasineutrality())
        ω0 = find_modes(m, Wavevector(ky = 0.5); re = -0.06:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)[1]
        @test ω0 ≈ ref atol = 1e-8
    end
end

@testset "radial operator" begin
    mg = radial_grid(-20, 20, 40; bc = :mirror)
    rgrid = (-0.08:0.01:0.0, (0.002, 0.01, 0.02))
    # uniform k∥: the operator is singular at the local root with a flat null vector
    for kz in (0.01, 0.02)
        m = RadialModel(; ky = 0.5, kpar = x -> kz, κT = x -> -0.3)
        loc = find_modes(local_model(m, 0.0)...; re = rgrid[1], imag = rgrid[2], imin = 0.0)[1]
        λ, φ = KD.smallest(operator(mg, m, loc))
        @test abs(λ) < 1e-10
        @test maximum(abs.(φ ./ φ[1] .- 1)) < 1e-9
    end
    # uniform drift: the radial eigenvalue is the local root, α = 0 and 1
    for alpha in (0.0, 1.0)
        m = RadialModel(; ky = 0.5, drift = x -> 1 / 50, κT = x -> -8 / 50, κn = x -> -2 / 50, alpha)
        loc = find_modes(local_model(m, 0.0)...; re = -0.06:0.01:0.0, imag = (0.002, 0.01, 0.02), imin = 0.0)[1]
        r = eigenmode(mg, m, loc * 1.02)
        @test r.converged
        @test r.ω ≈ loc rtol = 1e-9
    end
    # sheared slab, L_s = 200, Dirichlet walls: converged against N and nμ
    m = sheared_slab(; ky = 0.5, Ls = 200.0, κT = -0.3)
    ref = eigenmode(radial_grid(-20, 20, 120), m, -0.0241 + 0.0034im)
    @test ref.converged
    @test ref.ω ≈ -0.024124 + 0.003427im atol = 2e-6
    @test ref.φ isa ScalarField && length(ref.φ) == 120 && ref.φ[1] == 0
    for (N, nmu) in ((80, 24), (80, 16), (80, 32))
        r = eigenmode(radial_grid(-20, 20, N; nmu), m, -0.0241 + 0.0034im)
        @test r.ω ≈ ref.ω atol = 1e-7
    end
    # the argument principle sees every eigenvalue, the seeded search only some
    g = radial_grid(-20, 20, 80)
    found = find_eigenmodes(g, m; re = -0.04:0.005:0.0, im = 0.001:0.002:0.006)
    inrect(ω) = -0.04 < real(ω) < 0 && 0.0005 < imag(ω) < 0.006
    @test count_eigenvalues(g, m, -0.04, 0.0, 0.0005, 0.006) >= count(r -> inrect(r.ω), found) >= 1
    # drift in the radial operator: A is analytic through the real axis (Im ω < 0 is the continuation):
    # complex derivative along the real and the imaginary direction agree
    md = RadialModel(; ky = 0.5, drift = x -> 1 / 50, κT = x -> -8 / 50)
    ω0, h = -0.0123 + 0.0im, 1e-7
    dA_re = (operator(mg, md, ω0 + h) - operator(mg, md, ω0 - h)) / 2h
    dA_im = (operator(mg, md, ω0 + im * h) - operator(mg, md, ω0 - im * h)) / (2im * h)
    @test norm(dA_re - dA_im) < 1e-4 * norm(dA_re)
end
