"""Place a one-dimensional CBC template bank along chirp mass."""

import logging
import argparse
import time
from tqdm import trange
import numpy as np
import matplotlib.pyplot as pp
import h5py
import pycbc
import pycbc.filter
from pycbc.waveform import get_fd_waveform, get_waveform_filter_length_in_time
from pycbc.psd import (
    insert_psd_option_group, verify_psd_options, from_cli as psd_from_cli
)
from pycbc.conversions import (
    mass1_from_mchirp_eta, mass2_from_mchirp_eta, mchirp_from_mass1_mass2
)


def parse_cli():
    parser = argparse.ArgumentParser()
    insert_psd_option_group(parser)
    parser.add_argument('--bank-file', required=True)
    parser.add_argument('--min-frequency', type=float, default=30)
    parser.add_argument('--sample-rate', type=int, default=2048)
    parser.add_argument('--waveform-model', default='IMRPhenomD')
    parser.add_argument('--output-ff', required=True)
    parser.add_argument('--output-plot', required=True)
    args = parser.parse_args()
    verify_psd_options(args, parser)
    return args


def mchirp_from_mtot(mtot):
    return mchirp_from_mass1_mass2(mtot / 2, mtot / 2)


def next_power_of_two(x):
    y = 1
    while True:
        if y >= x:
            return y
        y *= 2


class Stopwatch:
    def __init__(self):
        self.time = 0
        self.enter_time = None

    def __enter__(self):
        #self.enter_time = time.time()
        pass

    def __exit__(self, *_):
        #self.time += time.time() - self.enter_time
        #self.enter_time = None
        pass


class OneDimMatchedFilterContext:
    def __init__(self, cli_args, **config):
        self.cli_args = cli_args
        self.min_freq = config['min_freq']
        self.sample_rate = config['sample_rate']
        self.approximant = config['approximant']

        with h5py.File(config['bank_file'], 'r') as bfh:
            self.bank = mchirp_from_mtot(bfh['mass1'][:] + bfh['mass2'][:])

        self.waveform_stopwatch = Stopwatch()
        self.match_stopwatch = Stopwatch()

    def template_duration(self, mchirp):
        mass1 = mass1_from_mchirp_eta(mchirp, 0.25)
        mass2 = mass2_from_mchirp_eta(mchirp, 0.25)
        return get_waveform_filter_length_in_time(
            mass1=mass1,
            mass2=mass2,
            approximant=self.approximant,
            f_lower=self.min_freq
        )

    def make_template(self, mchirp, freq_resolution=None):
        mass1 = mass1_from_mchirp_eta(mchirp, 0.25)
        mass2 = mass2_from_mchirp_eta(mchirp, 0.25)
        if freq_resolution is None:
            duration = get_waveform_filter_length_in_time(
                mass1=mass1,
                mass2=mass2,
                approximant=self.approximant,
                f_lower=self.min_freq,
            )
            if duration < 2:
                duration = 2
            duration = next_power_of_two(duration)
            freq_resolution = 1 / duration
        with self.waveform_stopwatch:
            template, _ = get_fd_waveform(
                mass1=mass1,
                mass2=mass2,
                approximant=self.approximant,
                f_lower=self.min_freq,
                f_final=self.sample_rate / 2,
                delta_f=freq_resolution
            )
        return template

    def banksim(self):
        num_injs_per_template = 3
        num_test_templates = len(self.bank) - 1
        num_injs = num_test_templates * num_injs_per_template
        fit_factors = np.empty(num_injs)
        inj_mchirps = np.empty(num_injs)
        j = 0
        for i in trange(num_test_templates):
            with self.waveform_stopwatch:
                template_left = self.make_template(
                    self.bank[i]
                )
                template_right = self.make_template(
                    self.bank[i + 1],
                    freq_resolution=template_left.delta_f
                )
            psd = psd_from_cli(
                self.cli_args,
                length=len(template_left),
                delta_f=template_left.delta_f,
                low_frequency_cutoff=self.min_freq
            )
            min_freq_idx = np.searchsorted(
                psd.sample_frequencies, self.min_freq
            )
            psd[:min_freq_idx] = np.inf
            for _ in range(num_injs_per_template):
                inj_mchirps[j] = np.random.uniform(
                    self.bank[i], self.bank[i + 1]
                )
                with self.waveform_stopwatch:
                    injection = self.make_template(
                        inj_mchirps[j], freq_resolution=template_left.delta_f
                    )
                with self.match_stopwatch:
                    match_left, _ = pycbc.filter.match(
                        injection,
                        template_left,
                        psd=psd,
                        low_frequency_cutoff=self.min_freq
                    )
                    match_right, _ = pycbc.filter.match(
                        injection,
                        template_right,
                        psd=psd,
                        low_frequency_cutoff=self.min_freq
                    )
                fit_factors[j] = max(match_left, match_right)
                j += 1
        return fit_factors, inj_mchirps


pycbc.init_logging(verbose=True)

cli_args = parse_cli()

mf = OneDimMatchedFilterContext(
    cli_args,
    min_freq=cli_args.min_frequency,
    sample_rate=cli_args.sample_rate,
    approximant=cli_args.waveform_model,
    bank_file=cli_args.bank_file
)

logging.info('Calculating fitting factors')

fit_factors, inj_mchirps = mf.banksim()

logging.info(
    'Time spent calculating waveforms: %f s', mf.waveform_stopwatch.time
)
logging.info(
    'Time spent calculating matches: %f s', mf.match_stopwatch.time
)

logging.info('Writing results to file')

with h5py.File(cli_args.output_ff, 'w') as fff:
    fff['fitting_factor'] = fit_factors
    fff['injected_mchirp'] = inj_mchirps

logging.info('Making plots')

pp.figure(figsize=(10, 10))

pp.subplot(2, 1, 1)
pp.plot(inj_mchirps, fit_factors, '.', lw=0)
#for x in mf.bank[:10]:
#    pp.axvline(x, color='y', lw=0.5)
pp.xscale('log')
pp.xlabel('Chirp mass')
pp.ylabel('Fitting factor')

pp.subplot(2, 1, 2)
pp.hist(fit_factors, 1000, cumulative=True, density=True, histtype='step')
pp.yscale('log')
pp.grid()
pp.xlabel('Fitting factor')
pp.ylabel('Cumulative fraction')

pp.tight_layout()
pp.savefig(cli_args.output_plot, dpi=150)

logging.info('Done')
