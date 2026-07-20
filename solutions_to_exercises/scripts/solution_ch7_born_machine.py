#!/usr/bin/env python3
"""
###########################################################################
#  EXERCISE 7.3 — Born Machine Expressivity and Concentration
#
#  Chapter 7, Quantum Machine Learning
#  Topic: Born machines, IPR, Porter-Thomas, measurement correlations
#
#  ---------- EXERCISE STATEMENT ----------
#
#  (a) Show that for a product state |psi> = tensor_j |psi_j> with
#      local magnetizations m_j = <Z_j>, the IPR is
#      C_2 = 2^{-N} prod_j (1 + m_j^2).
#      Evaluate for (i) all m_j=0, (ii) all m_j=+/-1.
#
#  (b) For Haar-random states, each Born probability p(x) ~ Beta(1,D-1).
#      Show E[H] = psi(D+1) - psi(2), and the entropy deficit is
#      Delta_H = 1 - gamma_E ~ 0.42 nats.
#
#  (c) Prove I(X_A : X_Abar) <= 2 S(rho_A) via data processing.
#
#  ---------- ANALYTICAL SOLUTION ----------
#
#  (a) Born distribution factorizes: p(x) = prod_j p_j(x_j).
#      C_2 = prod_j (|alpha_j|^4 + |beta_j|^4) = prod_j (1+m_j^2)/2
#           = 2^{-N} prod_j (1+m_j^2).
#      (i)  m_j=0: C_2 = 1/D (uniform).
#      (ii) m_j=1: C_2 = 1 (delta function).
#
#  (b) E[p ln p] = (psi(2)-psi(D+1))/D using Beta integral identity.
#      E[H] = psi(D+1) - psi(2) = H_D - 1 ~ ln(D) + gamma_E - 1.
#      Delta_H = ln(D) - E[H] ~ 1 - gamma_E ~ 0.4228.
#
#  (c) Computational-basis measurement is a local CPTP map.
#      Data processing: I_Q(sigma) <= I_Q(|psi><psi|) = 2 S(rho_A).
#
#  ---------- NUMERICAL VERIFICATION ----------
###########################################################################
"""
import numpy as np
from scipy.special import digamma
from scipy.stats import unitary_group

np.random.seed(42)

# ========================================================================
# Part (a): Product state IPR
# ========================================================================
print("Part (a): IPR for product states")
print("=" * 60)

for N in [2, 4, 6]:
    D = 2**N

    # Case (i): equatorial states, all m_j = 0
    # |psi_j> = |+> = (|0> + |1>)/sqrt(2)
    C2_equatorial = 2**(-N) * np.prod([1 + 0**2 for _ in range(N)])
    assert abs(C2_equatorial - 1/D) < 1e-15

    # Case (ii): computational basis, all m_j = 1
    C2_comp = 2**(-N) * np.prod([1 + 1**2 for _ in range(N)])
    assert abs(C2_comp - 1) < 1e-15

    # Numerical verification with random product state
    ms = np.random.uniform(-1, 1, N)
    psi = np.array([1], dtype=complex)
    for m in ms:
        alpha_sq = (1 + m) / 2
        beta_sq = (1 - m) / 2
        psi = np.kron(psi, np.array([np.sqrt(alpha_sq), np.sqrt(beta_sq)]))

    born_probs = abs(psi)**2
    C2_numerical = np.sum(born_probs**2)
    C2_formula = 2**(-N) * np.prod(1 + ms**2)

    print(f"  N={N}: C2(m=0)={1/D:.4f}=1/D, C2(m=1)=1, "
          f"C2(random m)={C2_numerical:.6f} vs formula={C2_formula:.6f}")
    assert abs(C2_numerical - C2_formula) < 1e-10

# ========================================================================
# Part (b): Porter-Thomas Shannon entropy
# ========================================================================
print(f"\nPart (b): Shannon entropy of Haar-random Born distributions")
print("=" * 60)

euler_gamma = 0.5772156649

for N in [2, 3, 4, 5]:
    D = 2**N
    n_samples = 3000

    entropies = []
    for _ in range(n_samples):
        psi = unitary_group.rvs(D)[:, 0]
        probs = abs(psi)**2
        H_shannon = -np.sum(probs * np.log(probs))
        entropies.append(H_shannon)

    E_H_numerical = np.mean(entropies)
    E_H_analytical = digamma(D + 1) - digamma(2)
    H_uniform = np.log(D)
    deficit_pred = 1 - euler_gamma

    print(f"  N={N} (D={D}): E[H] = {E_H_numerical:.4f}  "
          f"(analytical = {E_H_analytical:.4f})")
    print(f"    ln(D) = {H_uniform:.4f}, deficit = {H_uniform - E_H_numerical:.4f}  "
          f"(predicted 1-gamma_E = {deficit_pred:.4f})")

# ========================================================================
# Part (c): Data processing inequality -- numerical illustration
# ========================================================================
print(f"\nPart (c): I(X_A:X_Abar) <= 2 S(rho_A)")
print("=" * 60)

N = 4
D = 2**N
m = 2**(N//2)
n_samples = 1000

for _ in range(3):
    psi = unitary_group.rvs(D)[:, 0]

    # Entanglement entropy
    psi_mat = psi.reshape(m, m)
    rho_A = psi_mat @ psi_mat.conj().T
    eigs = np.linalg.eigvalsh(rho_A)
    eigs = eigs[eigs > 1e-15]
    S_A = -np.sum(eigs * np.log(eigs))

    # Classical mutual information from Born distribution
    probs = abs(psi)**2
    probs_mat = probs.reshape(m, m)
    p_A = probs_mat.sum(axis=1)
    p_B = probs_mat.sum(axis=0)

    I_classical = 0
    for a in range(m):
        for b in range(m):
            if probs_mat[a, b] > 1e-15:
                I_classical += probs_mat[a, b] * np.log(
                    probs_mat[a, b] / (p_A[a] * p_B[b]))

    print(f"  I(X_A:X_Abar) = {I_classical:.4f} <= 2*S(rho_A) = {2*S_A:.4f}  "
          f"({'satisfied' if I_classical <= 2*S_A + 1e-10 else 'VIOLATED'})")
    assert I_classical <= 2 * S_A + 1e-10

print(f"\n  The bound is tight for product measurements on pure states.")
print(f"  Area-law states (S ~ boundary) can only produce boundary-scaling")
print(f"  classical correlations, while volume-law states permit extensive")
print(f"  multi-body correlations in the Born distribution.")

