import numpy as np
import math


def sineGaussian(t, t_0, f_0, amplitude, q_factor, phase):
    """
    Return a Gaussian modulated sinusoid:

        ``A exp[-(t - t_0)**2 / tau**2]cos(2*pi*f_c*t + phase)``

    Parameters
    ----------

    A: amplitude given
    t_0: centered time
    tau: q_factor / (2 * \pi * f_0)
    f_0: central frequency

    Returns
    -------
    time series with sineGaussian waveform

    """
    if f_0 < 0:
        raise ValueError(f"Center frequency (f_0={f_0:.2f}) must be >=0.")
    if amplitude < 0:
        raise ValueError(f"Cannot pass amplitude below zero! (passed {amplitude})")

    # exp(-a t^2) <->  sqrt(pi/a) exp(-pi^2/a * f^2)  = g(f)

    tau = q_factor / (2* math.pi * f_0)
    pre_fact = np.exp( - (t - t_0)**2 / tau**2)
    waveform_td = amplitude * pre_fact * np.cos(2 * math.pi * f_0 * t + phase)
    return waveform_td
