"""
    Wavevector(kx, ky, kz)

Wavevector in units of 1/ρ_i. `kperp` is the perpendicular wavenumber used by the Bessel
weights. The gradient drive acts along y (gradients in x), so ω_* ∝ ky.
"""
struct Wavevector{T<:Real}
    kx::T
    ky::T
    kz::T
end
Wavevector(; kx = 0.0, ky = 0.0, kz = 0.0) = Wavevector(promote(float(kx), float(ky), float(kz))...)
kperp(k::Wavevector) = hypot(k.kx, k.ky)

abstract type AbstractResponse end
abstract type AbstractFieldEquation end
