#!/home/lorenzo-mobilia/.conda/envs/myenv3/bin/python

import matplotlib.pyplot as plt
import numpy as np

def plot(ts_inj, ts_noise, ts_glitch, td_glitch, analysis, threshold, activator, type_inj, job_id):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12)) 
    if(analysis == 'injection'):
        axs[0].plot(ts_noise.sample_times, ts_noise, label = 'Noise + Inj')
        axs[0].plot(ts_inj.sample_times, ts_inj, label = 'Inj')
        if activator >= threshold:
            axs[0].plot(ts_glitch.sample_times, ts_glitch, label = 'sinGauss')
        axs[0].axvline(x=59, color = 'r')
        axs[0].axvline(x=63, color = 'r')
        axs[0].set_title(f'Injection {job_id}')
        axs[0].set_xlabel('time[s]')
        axs[0].set_ylabel('strain')
        axs[0].legend(loc='upper right', fontsize='medium', title='Legend with Details', frameon=True)
        # zoom on glitch
        if activator >= threshold:
            # Non zero values
            non_zero_indices = td_glitch != 0
            sineGaussian_nonzero_td = td_glitch[non_zero_indices]
            mask = abs(sineGaussian_nonzero_td) > 5e-29
            sineGaussian_nonzero_td = sineGaussian_nonzero_td[mask]
            # Select corresponding values in time series
            corresponding_times = ts_glitch.sample_times[non_zero_indices]
            corresponding_times = corresponding_times[mask]
            axs[1].plot(corresponding_times, sineGaussian_nonzero_td)
            axs[1].set_title('Injected Glitch (zoom)')
            axs[1].set_xlabel('time[s]')
            axs[1].set_ylabel('strain')
            plt.tight_layout()
        plt.savefig('/home/lorenzo-mobilia/public_html/mixed-distances-x2-run3/injections_15k_bbh/'+ f'/Injection_{job_id}.png', bbox_inches='tight', dpi = 150)
        plt.close()
    elif(analysis == 'noise'):
        axs[0].plot(ts_noise.sample_times, ts_noise, label = 'Noise')
        if activator >= threshold:
            axs[0].plot(ts_glitch.sample_times, ts_glitch, label = 'sinGauss')
        axs[0].set_xlabel('time[s]')
        axs[0].set_ylabel('strain')
        axs[0].legend(loc='upper right', fontsize='medium', title='Legend with Details', frameon=True)
        # zoom on glitch
        if activator >= threshold:
            # Non zero values
            non_zero_indices = sineGaussian_td != 0
            sineGaussian_nonzero_td = sineGaussian_td[non_zero_indices]
            mask = abs(sineGaussian_nonzero_td) > 5e-29
            sineGaussian_nonzero_td = sineGaussian_nonzero_td[mask]
            # Select corresponding values in time series
            corresponding_times = ts_glitch.sample_times[non_zero_indices]
            corresponding_times = corresponding_times[mask]
            axs[1].plot(corresponding_times, sineGaussian_nonzero_td)
            axs[1].set_title('Injected Glitch (zoom)')
            axs[1].set_xlabel('time[s]')
            axs[1].set_ylabel('strain')
            plt.tight_layout()
        plt.savefig('/home/lorenzo-mobilia/public_html/mixed-distances-x2-run3/noise_15k_bbh/'+ f'/Noise_{job_id}.png', bbox_inches='tight')  
