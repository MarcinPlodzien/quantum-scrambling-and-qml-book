#!/usr/bin/env python3
"""
###########################################################################
###########################################################################
##
##  SOLUTION: Haar Invariance on U(1)
##
##  Book:     Information Scrambling and the Haar Measure
##  Chapter:  1 -- Mathematical Foundations
##  Section:  1.6 -- The Haar Measure
##
###########################################################################
##
##  EXERCISE STATEMENT
##  ------------------
##
##  The circle group U(1) = {exp(iθ) : θ ∈ [0,2π)} is the simplest
##  compact Lie group.  Its Haar measure is dμ = dθ/(2π).
##
##  (a) Verify translation invariance:
##      ∫ f(e^{i(θ+α)}) dθ/2π = ∫ f(e^{iθ}) dθ/2π.
##
##  (b) Prove uniqueness: any continuous translation-invariant density
##      p(θ) must equal 1/(2π).
##      Argument via Fourier coefficients: translation invariance
##      forces e^{inα} p_n = p_n for all α, so p_n = 0 for n ≠ 0.
##
##  (c) For Haar-random U ∈ U(D), the phase of det(U) = exp(iΣ_k φ_k)
##      is uniformly distributed on [0,2π).
##
###########################################################################
##
##  ANALYTICAL SOLUTION
##  -------------------
##
##  (a) Substitute θ' = θ + α.  The integral over [0,2π) is invariant
##      because exp(iθ') has period 2π.
##
##  (b) Fourier expansion: p(θ) = Σ_n p_n e^{inθ}.
##      Translation invariance: p(θ+α) must yield the same integral
##      against all test functions, forcing e^{inα} p_n = p_n.
##      For n ≠ 0, choosing α such that e^{inα} ≠ 1 gives p_n = 0.
##      Only p_0 survives; normalization fixes p(θ) = 1/(2π).
##
##  (c) Left multiplication U → e^{iα}U shifts all eigenphases by α,
##      so det(U) → e^{iDα} det(U).  Since this preserves Haar measure,
##      the phase of det(U) must be shift-invariant.
##      By part (b), it is uniformly distributed on [0,2π).
##
###########################################################################
###########################################################################
"""
import sympy as sp
import numpy as np
from scipy.stats import unitary_group, kstest

# ========================================================================
#  SECTION 1 :  Symbolic derivation with SymPy
# ========================================================================

print("=" * 70)
print("  Symbolic proof of uniqueness via Fourier analysis")
print("=" * 70)

theta = sp.Symbol('theta', real=True)
alpha = sp.Symbol('alpha', real=True)
n = sp.Symbol('n', integer=True, nonzero=True)

# --- Step 1: Translation invariance of dθ/(2π) ---
print("\n  Step 1: Translation invariance of the uniform measure")
print("  " + "-" * 50)
print("    For f periodic with period 2π:")
print("    ∫₀²π f(θ+α) dθ = ∫₀²π f(θ') dθ'  (substitution θ' = θ+α)")
print("    The integration domain wraps around: [α, 2π+α) ≡ [0, 2π).")

# Verify symbolically for f = cos(kθ), k = 1,2,3
for k in [1, 2, 3]:
    f_shifted = sp.cos(k * (theta + alpha))
    integral_shifted = sp.integrate(f_shifted, (theta, 0, 2*sp.pi)) / (2*sp.pi)
    integral_base = sp.integrate(sp.cos(k * theta), (theta, 0, 2*sp.pi)) / (2*sp.pi)
    integral_shifted = sp.simplify(integral_shifted)
    integral_base = sp.simplify(integral_base)
    print(f"    f = cos({k}θ): ∫f(θ+α) = {integral_shifted}, ∫f(θ) = {integral_base}")
    assert integral_shifted == integral_base

# --- Step 2: Uniqueness via Fourier coefficients ---
print("\n  Step 2: Fourier uniqueness — only p₀ survives")
print("  " + "-" * 50)
print("    Translation invariance requires e^{inα} p_n = p_n for all α.")
print("    For n ≠ 0, e^{inα} ≠ 1 for generic α, forcing p_n = 0.")

# Show that ∫₀²π e^{inα} p_n dα ≠ p_n unless p_n = 0
condition = sp.exp(sp.I * n * alpha)
integrated = sp.integrate(condition, (alpha, 0, 2*sp.pi)) / (2*sp.pi)
integrated = sp.simplify(integrated)
print(f"    (1/2π) ∫₀²π e^{{inα}} dα = {integrated}  for n ≠ 0")

# Verify the uniform density has p_0 = 1/(2π) and all other p_n = 0
print("\n    Fourier coefficients of p(θ) = 1/(2π):")
for k_val in [0, 1, 2, 3]:
    p_n = sp.integrate(sp.exp(-sp.I * k_val * theta) / (2*sp.pi),
                        (theta, 0, 2*sp.pi)) / (2*sp.pi)
    p_n = sp.simplify(p_n)
    print(f"      p_{k_val} = {p_n}")

# --- Step 3: Eigenphase uniformity ---
print("\n  Step 3: Each eigenphase of Haar-random U(D) is uniform")
print("  " + "-" * 50)
print("    The eigenvalue set of e^{iα}U is that of U rotated by α on the circle.")
print("    Left-invariance makes the averaged spectral density rotation-invariant,")
print("    hence uniform by Step 2. Likewise arg det U = Σφ_k is uniform; the")
print("    arithmetic mean of eigenphases is branch-dependent (uniform mod 2π/D).")

# ========================================================================
#  SECTION 2 :  Numerical verification with NumPy
# ========================================================================

print("\n" + "=" * 70)
print("  Numerical verification")
print("=" * 70)

np.random.seed(42)

# Part (a): Translation invariance via Monte Carlo
print("\n  Part (a): Translation invariance of the uniform measure")
print("  " + "-" * 50)

N_samples = 100000
thetas = np.random.uniform(0, 2*np.pi, N_samples)

for f_name, f in [("cos(θ)", np.cos),
                   ("cos²(θ)", lambda t: np.cos(t)**2),
                   ("sin(3θ)", lambda t: np.sin(3*t))]:
    base_mean = np.mean(f(thetas))
    for alpha_val in [0.5, 1.7, np.pi]:
        shifted_mean = np.mean(f(thetas + alpha_val))
        assert np.isclose(base_mean, shifted_mean, atol=0.02)
    print(f"    f = {f_name:20s}: <f> = {base_mean:+.4f}  (shift-invariant)  ✓")

# Part (b): Fourier coefficients
print("\n  Part (b): Fourier coefficients of uniform vs non-uniform density")
print("  " + "-" * 50)

theta_grid = np.linspace(0, 2*np.pi, 10000, endpoint=False)
p_uniform = np.ones_like(theta_grid) / (2*np.pi)

for k_val in [1, 2, 3, 5]:
    p_n = np.trapz(p_uniform * np.exp(-1j * k_val * theta_grid),
                    theta_grid) / (2*np.pi)
    print(f"    Uniform: p_{k_val} = {abs(p_n):.6f}  (expected 0)  ✓")

p_nonuniform = (1 + 0.5*np.cos(theta_grid)) / (2*np.pi)
p1_nonuniform = np.trapz(p_nonuniform * np.exp(-1j * theta_grid),
                          theta_grid) / (2*np.pi)
print(f"    Non-uniform (1+cos)/2π: p₁ = {abs(p1_nonuniform):.6f}  (≠ 0: NOT invariant)")

# Part (c): KS test for det-phase uniformity
print("\n  Part (c): Determinant phase uniformity (KS test)")
print("  " + "-" * 50)
print("    The phase of det(U) = exp(i·Σ_k φ_k) is uniform by Haar invariance.")

for d_val in [2, 4, 8]:
    det_phases = []
    for _ in range(5000):
        U = unitary_group.rvs(d_val)
        det_phase = np.angle(np.linalg.det(U)) % (2 * np.pi)
        det_phases.append(det_phase)

    normalized = np.array(det_phases) / (2 * np.pi)
    stat, pval = kstest(normalized, 'uniform')
    assert pval > 0.001, f"D={d_val}: KS p-value = {pval}"

    print(f"    D={d_val}: KS stat = {stat:.4f}, p-value = {pval:.4f}  ✓")
