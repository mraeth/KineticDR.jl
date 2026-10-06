# Plasma dispersion function and Bessel weights.

"Fried–Conte function Z(ζ) = i√π w(ζ), analytically continued to Im ζ < 0."
plasma_Z(ζ) = im * sqrt(π) * erfcx(-im * ζ)

# 1 + ζ Z(ζ) without cancellation at large |ζ| (asymptotic series, valid for Im ζ ≥ 0).
function one_plus_zZ(ζ)
    if abs(ζ) > 8 && imag(ζ) >= 0
        w = 1 / (2ζ^2)
        s = zero(ζ)
        t = one(ζ)
        for n in 1:12
            t *= (2n - 1) * w
            s -= t
        end
        # The resonant part i√π ζ e^{-ζ²} is dropped: for |ζ| > 8 and Im ζ ≥ 0 it is either
        # below e^{-32} (|Re ζ| > |Im ζ|) or absent from Z (upper half plane, Stokes sector).
        return s
    end
    return 1 + ζ * plasma_Z(ζ)
end

"Γ_p(x) = I_p(x) e^{-x} (x ≥ 0)."
Gamma_p(p::Integer, x::Real) = besselix(abs(p), x)

"dΓ_p/dx = (Γ_{p-1} + Γ_{p+1})/2 − Γ_p."
dGamma_p(p::Integer, x::Real) = (Gamma_p(p - 1, x) + Gamma_p(p + 1, x)) / 2 - Gamma_p(p, x)
