#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 4.2 — Krylov Complexity from the Geometric Distribution
#
#  Chapter 4, Scrambling Dynamics
#  Topic: Krylov basis, operator growth, Lanczos coefficients
#
#  ---------- EXERCISE STATEMENT ----------
#
#  When Lanczos coefficients grow linearly as b_n = alpha*n, the
#  Krylov amplitudes are phi_n(t) = tau^n / c, where
#  tau(t) = tanh(alpha*t) and c(t) = cosh(alpha*t).
#
#  (a) Verify |phi_n|^2 = (1-tau^2) tau^{2n} is normalized.
#  (b) Derive C_K(t) = sinh^2(alpha*t).
#  (c) Compute Var(n) = tau^2/(1-tau^2)^2.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) p_n = (1-tau^2) tau^{2n}.  Geometric series with ratio r = tau^2:
#      sum p_n = (1-tau^2) / (1-tau^2) = 1.
#
#  (b) C_K = sum n*p_n = (1-tau^2) * tau^2/(1-tau^2)^2 = tau^2/(1-tau^2).
#      Since tau=tanh and 1-tanh^2 = 1/cosh^2:
#      C_K = tanh^2 * cosh^2 = sinh^2(alpha*t).
#
#  (c) <n^2> = (1-tau^2) * tau^2(1+tau^2)/(1-tau^2)^3
#            = tau^2(1+tau^2)/(1-tau^2)^2.
#      Var = <n^2> - <n>^2 = tau^2/(1-tau^2)^2.
#      In hyperbolic functions: Var = sinh^2 * cosh^2 = (1/4)sinh^2(2*alpha*t).
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np

alpha = 1.5  # Lanczos growth rate

print(f"Krylov complexity with linear Lanczos coefficients b_n = {alpha}*n")
print("=" * 60)

for t in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]:
    tau = np.tanh(alpha * t)
    tau2 = tau**2
    n_max = 100000  # truncation for the infinite sum (needs more terms at large t)

    # --- Part (a): Normalization ---
    ns = np.arange(n_max)
    p_n = (1 - tau2) * tau2**ns
    norm = np.sum(p_n)
    assert abs(norm - 1) < 1e-4, f"Normalization = {norm} at t={t}"

    # --- Part (b): Krylov complexity ---
    C_K_numerical = np.sum(ns * p_n)
    C_K_formula = tau2 / (1 - tau2) if tau2 < 1 else np.inf
    C_K_hyperbolic = np.sinh(alpha * t)**2

    rel_err_ck = abs(C_K_numerical - C_K_formula) / max(C_K_formula, 1)
    assert rel_err_ck < 0.01, \
        f"C_K formula mismatch at t={t}: rel_err={rel_err_ck}"
    assert abs(C_K_formula - C_K_hyperbolic) < 1e-5, \
        f"C_K hyperbolic mismatch at t={t}: {C_K_formula} vs {C_K_hyperbolic}"

    # --- Part (c): Variance ---
    var_numerical = np.sum(ns**2 * p_n) - C_K_numerical**2
    var_formula = tau2 / (1 - tau2)**2
    var_hyperbolic = np.sinh(alpha*t)**2 * np.cosh(alpha*t)**2

    rel_err_var = abs(var_numerical - var_formula) / max(var_formula, 1)
    assert rel_err_var < 0.05, \
        f"Var formula mismatch at t={t}: rel_err={rel_err_var}"
    assert abs(var_formula - var_hyperbolic) < 1e-3, \
        f"Var hyperbolic mismatch at t={t}: {var_formula} vs {var_hyperbolic}"

    print(f"  t={t:.1f}: C_K = {C_K_hyperbolic:8.3f} = sinh^2({alpha}*{t:.1f}), "
          f"Var = {var_hyperbolic:10.3f} = sinh^2 cosh^2, "
          f"norm = {norm:.8f}")

print(f"\n  At late times, C_K ~ (1/4)e^{{2*alpha*t}}: exponential growth.")
print(f"  The wavepacket on the Krylov chain both translates (C_K)")
print(f"  and broadens (Var ~ C_K^2) exponentially.")

