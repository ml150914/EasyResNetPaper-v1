#!/home/lorenzo-mobilia/.conda/envs/myenv3/bin/python

import numpy as np

#------- Distance distribution ---------
def generate_x2_distribution(a, b):
    # PDF f(x) \sim x^2
    def pdf(x):
        return x**2

    # Inverse CDF method for sampling
    def inverse_cdf(u, a, b):
        return ((u * (b**3 - a**3)) + a**3)**(1/3)

    # Generate one uniform random number in [0, 1]
    u = np.random.uniform(0, 1)

    # Use the inverse CDF to generate a sample
    x = inverse_cdf(u, a, b)

    return x

#---->Chirp Distance
def chirp_distance(mc, d):
    return (1.4 / mc)**(- 5 / 6) * d

#--------- Mass Distribution -----------
#------> Power Law
def power_law_distribution(m_min=3, m_max=100, alpha=3.5, size = 1):
    u = np.random.uniform(0, 1, size)
    exponent = 1 - alpha
    m = ((m_max**exponent - m_min**exponent) * u + m_min**exponent) ** (1/exponent)
    return m
# -----> Gaussian Peak
def gaussian_enhancement(m, mean=34, sigma=3):
    return 1 + np.exp(-0.5 * ((m - mean) / sigma) ** 2)

def bbh_distribution_law(m_min=3, m_max=100, alpha=3.5, mean=34, sigma=3, size=1):
    
    target_size = int(size)
    samples = []
    M = 2.0
    while len(samples) < target_size:
        n_remaining = target_size - len(samples)
        candidates = power_law_distribution(m_min, m_max, alpha, size=n_remaining)
        weights = gaussian_enhancement(candidates, mean, sigma)  # in [1, 2]
        accept_prob = weights / M  # in [0.5, 1.0]
        u = np.random.uniform(0.0, 1.0, size=n_remaining)
        accepted = candidates[u < accept_prob]

        if accepted.size > 0:
            samples.extend(accepted.tolist())
    arr = np.array(samples[:target_size])
    if target_size == 1:
        return float(arr[0])
    return arr

#-----> Chirp mass
def chirp_mass(m1, m2):
    m1 = float(m1)
    m2 = float(m2)
    return ((m1 * m2) ** (3/5)) / ((m1 + m2) ** (1/5))


#-----> spin distribution
def extreme_spin_distribution(n_samples=1, mu=0.95, sigma=0.02):
    signs = np.random.choice([-1.0, 1.0], size=n_samples)
    centers = signs * mu

    samples = np.random.normal(loc=centers, scale=sigma)
    clipped = np.clip(samples, -0.999, 0.999)
    if n_samples == 1:
        return float(clipped[0])
    #safety clip
    return clipped	

#-----> spin distribution                                                                                                  
def extreme_spin_distribution_Ushape(samples=1, mu=0.95, sigma=0.02):
    chi1 = np.empty(samples)
    chi2 = np.empty(samples)
    signs = np.random.choice([-1.0, 1.0], size=samples)
    centers = signs * mu

    for i, s in enumerate(signs):
        chi1[i] = np.random.normal(loc=s * mu, scale=sigma)
        chi2[i] = np.random.normal(loc=s * mu, scale=sigma)
    
    chi1 = np.clip(chi1, -0.999, 0.999)
    chi2 = np.clip(chi2, -0.999, 0.999)
    
    if samples == 1:
        return float(chi1[0]), float(chi2[0])
    #safety clip                                                                                   
    return chi1, chi2

def extreme_spin_distribution_single_values_bbh(samples=1):
    chi1 = np.empty(samples)
    chi2 = np.empty(samples)
    signs = np.random.choice([-1.0, 1.0], size=samples)
    #centers = signs * mu

    for i, s in enumerate(signs):
        chi1[i] = signs#np.random.normal(loc=s * mu, scale=sigma)
        chi2[i] = signs #np.random.normal(loc=s * mu, scale=sigma)

    #chi1 = np.clip(chi1, -0.999, 0.999)
    #chi2 = np.clip(chi2, -0.999, 0.999)

    if samples == 1:
        return float(chi1[0]), float(chi2[0])
    #safety clip                                                                                                                                             
    return chi1, chi2

def extreme_spin_distribution_single_values_bns(samples=1):
    chi1 = np.empty(samples)
    chi2 = np.empty(samples)
    signs = np.random.choice([-0.5, 0.5], size=samples)
    #centers = signs * mu                                                                                                                                                                                    
    for i, s in enumerate(signs):
        chi1[i] = signs
        chi2[i] = signs
    
    if samples == 1:
        return float(chi1[0]), float(chi2[0])
    #safety clip                                                                                                                                                                                              
    return chi1, chi2

samples = bbh_distribution_law(size=100000)
print("non-finite:", np.sum(~np.isfinite(samples)))
