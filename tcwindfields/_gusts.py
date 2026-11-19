"""
Wind gusts from the mean wind

WMO guideline for TCs (Harper et al. 2010, WMO/TD-No. 1555), modified ESDU method of their Appendix D
    gust = G x mean_wind,   G = G_T0 (1 + g Iu),   Iu = 1 / ln(z / z0)
G is the highest 3-s gust expected within a period T0 over the mean wind. It grows with the roughness upwind (z0,
through the turbulence intensity Iu) and with T0 (g: peak factor in 1 h, G_T0: correction for T0 shorter than 1 h).
"""
import numpy as np

gust_s = 3.0  # s, WMO gust duration

def peak_factor(tau=gust_s, z=10.0):
    """Peak factor g of a tau-s gust within 1 h (Eqns D.2-D.6)."""
    tu = 3.13*z**0.2  # s, integral time scale of the turbulence
    nu = (0.007 + 0.213*(tu/tau)**0.654)/tu  # Hz, zero-crossing rate
    s = np.sqrt(2*np.log(nu*3600.0))
    std_ratio = 1 - 0.193*(tu/tau + 0.1)**-0.68  # std of the tau-s wind / std of the wind
    return (s + 0.577/s)*std_ratio

def gust_factor(z0, period_s=600.0, tau=gust_s, z=10.0):
    """
    Gust factor G (Eqn D.10): highest tau-s gust expected within period_s (60 to 3600 s) / mean wind, for the
    upwind roughness length z0 (m). 3-s gust within 10 min: 1.66 for z0 = 0.18 m (in-land), 1.52 for 0.07 m
    (off-land), 1.38 for 0.013 m (off-sea), 1.23 for 0.0005 m (at sea), as in Table 1.1 of the guideline.
    """
    iu = 1/np.log(z/np.asarray(z0, dtype=float))
    g_t0 = 0.2193*np.log(np.log10(period_s)) + 0.7242  # 1.00 for 1 h, 0.948 for 10 min (Eqn D.9)
    return g_t0*(1 + peak_factor(tau, z)*iu)
