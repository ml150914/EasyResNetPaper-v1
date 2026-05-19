"""Place a one-dimensional CBC template bank along chirp mass."""

import logging
import argparse
import time
import numpy as np
import matplotlib.pyplot as pp
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
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
    parser.add_argument('--min-frequency', type=float, default=30)
    parser.add_argument('--min-total-mass', type=float, default=2)
    parser.add_argument('--max-total-mass', type=float, default=50)
    parser.add_argument('--sample-rate', type=int, default=2048)
    parser.add_argument('--min-match', type=float, default=0.97)
    parser.add_argument('--waveform-model', default='IMRPhenomD')
    parser.add_argument('--output-bank', required=True)
    parser.add_argument('--output-plot', required=True)
    parser.add_argument('--spacing-tolerance', type=float, default=0.1)
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
        self.enter_time = time.time()

    def __exit__(self, *_):
        self.time += time.time() - self.enter_time
        self.enter_time = None


class OneDimMatchedFilterContext:
    def __init__(self, cli_args, **config):
        self.cli_args = cli_args

        self.min_freq = config['min_freq']
        self.sample_rate = config['sample_rate']
        self.min_mchirp = config['min_mchirp']
        self.max_mchirp = config['max_mchirp']
        self.min_match = config['min_match']
        self.approximant = config['approximant']

        self.bank = [self.min_mchirp]

        self.waveform_stopwatch = Stopwatch()
        self.match_stopwatch = Stopwatch()
        self.learn_stopwatch = Stopwatch()
        self.place_stopwatch = Stopwatch()

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

    def template_spacing(self, mchirp, approx_estimate=None):
        """Return the difference in chirp mass between two templates that
        achieves the given minimal match.
        """
        logging.info('Measuring spacing at %f MSun', mchirp)
        template1 = self.make_template(mchirp)
        psd = psd_from_cli(
            self.cli_args,
            length=len(template1),
            delta_f=template1.delta_f,
            low_frequency_cutoff=self.min_freq
        )
        min_freq_idx = np.searchsorted(psd.sample_frequencies, self.min_freq)
        psd[:min_freq_idx] = np.inf

        def objective(mchirp2):
            template2 = self.make_template(mchirp2, template1.delta_f)
            with self.match_stopwatch:
                m, _ = pycbc.filter.match(
                    template1,
                    template2,
                    psd=psd,
                    low_frequency_cutoff=self.min_freq
                )
            return m - self.min_match

        # FIXME It is easy to get waveform failures at the high end of the mass
        # if the min frequency is too high, because this calculation can push
        # `max_mchirp` above the range we are trying to cover. It may be useful
        # to have the ability to look from `mchirp` *up* or from `mchirp`
        # *down*, depending on where `mchirp` is in the target range.
        if approx_estimate:
            max_mchirp = mchirp + 2 * approx_estimate
        else:
            max_mchirp = 2 * mchirp
        mchirp_at_min_match = brentq(objective, mchirp, max_mchirp)
        # Note that this code cheat a little bit. We just determined the chirp
        # mass that makes the match *between the two neighboring templates*
        # equal to the target minimal match. But in order to achieve the target
        # minimal match, the second template should go at an even higher chirp
        # mass! Where, exactly? Well, we would need a second optimization for
        # that. But let's just assume that the minimum fitting factor is found
        # exactly in the middle of the two templates, and take the correct
        # spacing to be simply twice what we just obtained.
        return (mchirp_at_min_match - mchirp) * 2

    def learn_spacing(self, tolerance=0.1):
        """Perform an adaptive sampling of the required spacing between
        templates, over the entire range of chirp mass, to achieve the
        required minimal match. New samples are iteratively added
        between consecutive samples that differ in spacing more than
        `tolerance` (interpreted as relative absolute difference).
        """
        spacing_samples = [
            [self.min_mchirp, self.template_spacing(self.min_mchirp)],
            [self.max_mchirp, self.template_spacing(self.max_mchirp)]
        ]
        while True:
            new_points = []
            for s1, s2 in zip(spacing_samples[:-1], spacing_samples[1:]):
                rel_delta = abs((s2[1] - s1[1]) / s1[1])
                if rel_delta > tolerance:
                    new_point = (s1[0] + s2[0]) / 2
                    new_points.append(
                        [new_point, self.template_spacing(new_point, s2[1])]
                    )
            if not new_points:
                break
            spacing_samples += new_points
            spacing_samples.sort()
        self.spacing_samples = np.array(spacing_samples)

    def place_templates(self):
        # Invert spacing to get density: n(Mc)
        self.density = 1 / self.spacing_samples[:,1]
        # Integrate density to get number of templates up to a given mchirp: N(Mc)
        self.cum_num = cumulative_trapezoid(
            self.density, self.spacing_samples[:,0], initial=0
        )
        # Invert N(Mc) to get template placement: Mc_i = N^(-1)(i) with i = {0 … N}
        cum_num_interp = interp1d(
            self.spacing_samples[:,0],
            self.cum_num,
            kind='linear',
            copy=False,
            assume_sorted=True
        )
        num_templates = int(self.cum_num[-1]) + 1
        self.bank = np.empty(num_templates)
        for i in range(num_templates - 1):
            self.bank[i] = brentq(
                lambda mchirp: cum_num_interp(mchirp) - i,
                self.min_mchirp,
                self.max_mchirp
            )
        self.bank[-1] = self.max_mchirp


pycbc.init_logging(verbose=True)

cli_args = parse_cli()

mf = OneDimMatchedFilterContext(
    cli_args,
    min_freq=cli_args.min_frequency,
    sample_rate=cli_args.sample_rate,
    min_mchirp=mchirp_from_mtot(cli_args.min_total_mass),
    max_mchirp=mchirp_from_mtot(cli_args.max_total_mass),
    min_match=cli_args.min_match,
    approximant=cli_args.waveform_model
)

logging.info('Learning the required spacing between templates')

with mf.learn_stopwatch:
    mf.learn_spacing(cli_args.spacing_tolerance)

logging.info('Spacing sampled over %d points', mf.spacing_samples.shape[0])

logging.info('Placing templates')

with mf.place_stopwatch:
    mf.place_templates()

logging.info('%d templates', len(mf.bank))

logging.info('Time taken to learn spacing: %f s', mf.learn_stopwatch.time)
logging.info('Time taken to place templates: %f s', mf.place_stopwatch.time)
logging.info('Time spent calculating waveforms: %f s', mf.waveform_stopwatch.time)
logging.info('Time spent calculating matches: %f s', mf.match_stopwatch.time)
logging.info(
    'Fraction of time calculating waveforms: %f',
    mf.waveform_stopwatch.time / (mf.learn_stopwatch.time + mf.place_stopwatch.time)
)
logging.info(
    'Fraction of time calculating matches: %f',
    mf.match_stopwatch.time / (mf.learn_stopwatch.time + mf.place_stopwatch.time)
)

logging.info('Writing bank to file')
with h5py.File(cli_args.output_bank, 'w') as bank_file:
    bank_file['mass1'] = mass1_from_mchirp_eta(mf.bank, 0.25)
    bank_file['mass2'] = mass2_from_mchirp_eta(mf.bank, 0.25)
    bank_file['spin1z'] = np.zeros_like(mf.bank)
    bank_file['spin2z'] = np.zeros_like(mf.bank)
    bank_file.attrs['time_to_learn_spacing'] = mf.learn_stopwatch.time
    bank_file.attrs['time_to_place_templates'] = mf.place_stopwatch.time
    bank_file.attrs['time_in_waveform_model'] = mf.waveform_stopwatch.time
    bank_file.attrs['time_in_match'] = mf.match_stopwatch.time

fit_exp = np.log(mf.spacing_samples[0,1] / mf.spacing_samples[1,1]) / np.log(mf.spacing_samples[0,0] / mf.spacing_samples[1,0])
fit_scale = mf.spacing_samples[0,1] * mf.spacing_samples[0,0] ** (-fit_exp)

logging.info('Power law model: scale %f, exponent %f', fit_scale, fit_exp)

logging.info('Making plots')

pp.figure(figsize=(10, 5))

pp.subplot(1, 2, 1)
pp.loglog(
    mf.spacing_samples[:,0],
    mf.spacing_samples[:,1],
    '.',
    markeredgewidth=0,
    label='Measured'
)
pp.loglog(
    mf.spacing_samples[:,0],
    fit_scale * mf.spacing_samples[:,0] ** fit_exp,
    '--',
    label='Power law model'
)
pp.xlabel('Chirp mass [$M_\\odot$]')
pp.ylabel('Template spacing [$M_\\odot$]')
pp.legend()
pp.grid()

pp.subplot(1, 2, 2)
pp.plot(
    mf.spacing_samples[:,0],
    mf.cum_num[-1] - mf.cum_num + 1,
    '-',
    label='Planned'
)
pp.step(
    mf.bank, np.arange(len(mf.bank))[::-1] + 1, where='pre', label='Placed'
)
pp.xscale('log')
pp.yscale('log')
pp.legend()
pp.grid()
pp.xlabel('Chirp mass [$M_\\odot$]')
pp.ylabel('Cumulative number of templates')

pp.tight_layout()
pp.savefig(cli_args.output_plot, dpi=150)

logging.info('Done')
