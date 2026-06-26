#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Imports
########################################

import re
import sys, os

# Update this to point to the directory where you copied the SciAnalysis base code
# SciAnalysis_PATH='/home/kyager/current/code/SciAnalysis/main/'
# SciAnalysis_PATH='/nsls2/xf11bm/software/SciAnalysis/'
SciAnalysis_PATH = '/nsls2/data/cms/legacy/xf11bm/software/SciAnalysis/'
SciAnalysis_PATH in sys.path or sys.path.append(SciAnalysis_PATH)

Abs_Path = "/nsls2/data/cms/proposals/2026-1/pass-319051/experiments/0_test/"

import glob
import pandas as pd
import tempfile
from PIL import Image
from SciAnalysis import tools
from SciAnalysis.XSAnalysis.Data import *
from SciAnalysis.XSAnalysis import Protocols

import time

FITTING_PLAN = 'circular_average_q2I_fit_FWHM' #CZ
EXTRACTION = 'fit_peaks_fwhm1'

# Define some custom analysis routines
########################################
from SciAnalysis.Result import *  # The Results() class allows one to extract data from XML files.


# ==============================================
# ==============waxs section====================
# ==============================================

class linecut_qz_fit(Protocols.linecut_qz):  # TODO: Use class fit_peaks

    def __init__(self, name='linecut_qz_fit', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {
            'show_region': False,
            'plot_range': [None, None, 0, None],
            'show_guides': True,
            'markersize': 0,
            'linewidth': 1.5,
        }
        self.run_args.update(kwargs)

    def output_exists(self, name, output_dir):

        outfile = self.get_outfile(name, output_dir, ext='-fit.png')
        return os.path.isfile(outfile)

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        # Usage example:
        # linecut_qz_fit(show_region=False, show=False, qr=0.009, dq=0.0025, q_mode='qr', fit_range=fit_range, q0_expected=q0_expected, plot_range=[0, 0.08, 0, None]) ,

        results = {}

        line = data.linecut_qz(**run_args)

        if 'show_region' in run_args:
            if run_args['show_region'] == 'save':
                outfile = self.get_outfile(data.name, output_dir, ext='_region.png')
                data.plot(save=outfile)
            elif run_args['show_region']:
                data.plot(show=True)

        # line.smooth(2.0, bins=10)
        if 'trim_range' in run_args:
            line.trim(run_args['trim_range'][0], run_args['trim_range'][1])

        # if 'plots' in run_args['save_results']:
        if False:
            self.label_filename(data, line, **run_args)
            outfile = self.get_outfile(data.name, output_dir)
            line.plot(save=outfile, **run_args)

        if 'txt' in run_args['save_results']:
            outfile = self.get_outfile(data.name, output_dir, ext='.dat')
            line.save_data(outfile)

        if 'incident_angle' not in run_args:
            run_args['incident_angle'] = data.calibration.incident_angle

            import re
            filename_re = re.compile(r'^.+_th(-?\d+\.\d+)_.+$')
            m = filename_re.match(data.name)
            if m:
                run_args['incident_angle'] = float(m.groups()[0])

        if 'verbosity' in run_args and run_args['verbosity'] >= 4:
            print('    Using incident_angle = {:.3f} degrees'.format(run_args['incident_angle']))

        # if 'critical_angle_film' not in run_args:
        # run_args['critical_angle_film'] = 0
        # if 'critical_angle_substrate' not in run_args:
        # run_args['critical_angle_substrate'] = 0

        # Fit data
        lm_result, fit_line, fit_line_extended = self._fit_peaks(line, **run_args)

        # Save fit results
        fit_name = 'fit_peaks'
        prefactor_total = 0
        for param_name, param in lm_result.params.items():
            results['{}_{}'.format(fit_name, param_name)] = {'value': param.value, 'error': param.stderr, }
            if 'prefactor' in param_name:
                prefactor_total += np.abs(param.value)

        results['{}_prefactor_total'.format(fit_name)] = prefactor_total
        results['{}_chi_squared'.format(fit_name)] = lm_result.chisqr / lm_result.nfree

        # Calculate some additional things
        q0 = results['{}_x_center1'.format(fit_name)]['value']
        d = 0.1 * 2. * np.pi / q0
        results['{}_d0'.format(fit_name)] = d
        xi = 0.1 * (2. * np.pi / np.sqrt(2. * np.pi)) / results['{}_sigma1'.format(fit_name)]['value']
        results['{}_grain_size'.format(fit_name)] = xi

        def angle_to_q(two_theta_s_rad):
            k = data.calibration.get_k()
            qz = 2 * k * np.sin(two_theta_s_rad / 2.0)
            return qz

        def q_to_angle(q):
            k = data.calibration.get_k()
            two_theta_s_rad = 2.0 * np.arcsin(q / (2.0 * k))
            return two_theta_s_rad

        if 'critical_angle_film' in run_args:
            # Account for refraction distortion

            theta_incident_rad = np.radians(run_args['incident_angle'])
            theta_c_f_rad = np.radians(run_args['critical_angle_film'])
            # theta_c_s_rad = np.radians(run_args['critical_angle_substrate'])

            alpha_i_rad = np.arccos(np.cos(theta_incident_rad) / np.cos(theta_c_f_rad))

            # Scattering from incident (refracted) beam
            two_theta_s_rad = q_to_angle(q0)
            theta_f_rad = two_theta_s_rad - theta_incident_rad

            alpha_f_rad = np.arccos(np.cos(theta_f_rad) / np.cos(theta_c_f_rad))

            two_alpha_s_rad = alpha_i_rad + alpha_f_rad
            qT = angle_to_q(two_alpha_s_rad)
            results['{}_qT'.format(fit_name)] = qT
            results['{}_dT'.format(fit_name)] = 0.1 * 2. * np.pi / qT

            # Scattering from reflected beam
            two_alpha_s_rad = abs(alpha_f_rad - alpha_i_rad)
            qR = angle_to_q(two_alpha_s_rad)
            results['{}_qR'.format(fit_name)] = qR
            results['{}_dR'.format(fit_name)] = 0.1 * 2. * np.pi / qR

        # Plot and save data
        class DataLines_current(DataLines):

            def _plot_extra(self, **plot_args):

                xi, xf, yi, yf = self.ax.axis()
                y_fit_max = np.max(self.lines[1].y)
                yf = y_fit_max * 2.0
                v_spacing = (yf - yi) * 0.06

                q0 = self.results['fit_peaks_x_center1']['value']
                color = 'purple'

                yp = yf
                s = r'$p = \, {:.4f}$'.format(self.results['fit_peaks_prefactor1']['value'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                s = r'$\chi^2 = \, {:.4f}$'.format(self.results['fit_peaks_chi_squared'])
                self.ax.text(xi, yi, s, size=15, color=color, verticalalignment='bottom', horizontalalignment='left')

                yp -= v_spacing
                s = r'$q_0 = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(q0)
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$d_0 \approx \, {:.1f} \, \mathrm{{nm}}$'.format(self.results['fit_peaks_d0'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$\sigma = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(self.results['fit_peaks_sigma1']['value'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$\xi \approx \, {:.1f} \, \mathrm{{nm}}$'.format(self.results['fit_peaks_grain_size'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                self.ax.axvline(q0, color=color, linewidth=0.5)
                self.ax.text(q0, yf, '$q_0$', size=20, color=color, horizontalalignment='center',
                             verticalalignment='bottom')

                if 'critical_angle_film' in self.run_args:
                    s = r'$q_T = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$ \n $d_T = \, {:.1f} \, \mathrm{{nm}}$'.format(
                        self.results['fit_peaks_qT'], self.results['fit_peaks_dT'])
                    self.ax.text(q0, y_fit_max, s, size=15, color='b', horizontalalignment='left',
                                 verticalalignment='bottom')
                    self.ax.plot([self.results['fit_peaks_qT'], q0], [y_fit_max, y_fit_max], '-', color='b')

                    s = r'$q_R = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$ \n $d_R = \, {:.1f} \, \mathrm{{nm}}$'.format(
                        self.results['fit_peaks_qR'], self.results['fit_peaks_dR'])
                    self.ax.text(q0, 0, s, size=15, color='r', horizontalalignment='left', verticalalignment='bottom')
                    self.ax.plot([self.results['fit_peaks_qR'], q0], [yi, yi], '-', color='r')

                if self.run_args['show_guides']:
                    # Show various guides of scattering features
                    theta_incident_rad = np.radians(self.run_args['incident_angle'])

                    # Direct
                    qz = 0
                    self.ax.axvline(qz, color='0.25')
                    self.ax.text(qz, yf, r'$\mathrm{D}$', size=20, color='0.25', horizontalalignment='center',
                                 verticalalignment='bottom')

                    # Horizon
                    qz = angle_to_q(theta_incident_rad)
                    l = self.ax.axvline(qz, color='0.5')
                    l.set_dashes([10, 6])
                    self.ax.text(qz, yf, r'$\mathrm{H}$', size=20, color='0.5', horizontalalignment='center',
                                 verticalalignment='bottom')

                    # Specular beam
                    qz = angle_to_q(2 * theta_incident_rad)
                    self.ax.axvline(qz, color='r')
                    self.ax.text(qz, yf, r'$\mathrm{R}$', size=20, color='r', horizontalalignment='center',
                                 verticalalignment='bottom')

                    if 'critical_angle_film' in self.run_args:
                        theta_c_f_rad = np.radians(self.run_args['critical_angle_film'])

                        # Transmitted (direct beam refracted by film)
                        if theta_incident_rad <= theta_c_f_rad:
                            qz = angle_to_q(theta_incident_rad)  # Horizon
                        else:
                            alpha_i_rad = np.arccos(np.cos(theta_incident_rad) / np.cos(theta_c_f_rad))
                            two_theta_s_rad = theta_incident_rad - alpha_i_rad
                            qz = angle_to_q(two_theta_s_rad)
                        l = self.ax.axvline(qz, color='b')
                        l.set_dashes([4, 4])

                        # Yoneda
                        qz = angle_to_q(theta_incident_rad + theta_c_f_rad)
                        self.ax.axvline(qz, color='gold')
                        self.ax.text(qz, yf, r'$\mathrm{Y}_f$', size=20, color='gold', horizontalalignment='center',
                                     verticalalignment='bottom')

                    if 'critical_angle_substrate' in self.run_args:
                        theta_c_s_rad = np.radians(self.run_args['critical_angle_substrate'])

                        # Transmitted (direct beam refracted by substrate)
                        if theta_incident_rad <= theta_c_s_rad:
                            qz = angle_to_q(theta_incident_rad)  # Horizon
                        else:
                            alpha_i_rad = np.arccos(np.cos(theta_incident_rad) / np.cos(theta_c_s_rad))
                            two_theta_s_rad = theta_incident_rad - alpha_i_rad
                            qz = angle_to_q(two_theta_s_rad)
                        self.ax.axvline(qz, color='b')
                        self.ax.text(qz, yf, r'$\mathrm{T}$', size=20, color='b', horizontalalignment='center',
                                     verticalalignment='bottom')

                        # Yoneda
                        qz = angle_to_q(theta_incident_rad + theta_c_s_rad)
                        self.ax.axvline(qz, color='gold')
                        self.ax.text(qz, yf, r'$\mathrm{Y}_s$', size=20, color='gold', horizontalalignment='center',
                                     verticalalignment='bottom')

                self.ax.axis([xi, xf, yi, yf])

        lines = DataLines_current([line, fit_line, fit_line_extended])
        lines.copy_labels(line)
        lines.results = results
        lines.run_args = run_args

        if 'plots' in run_args['save_results']:
            self.label_filename(data, lines, **run_args)
            outfile = self.get_outfile(data.name + '-fit', output_dir, ext='.png')
            # lines.plot(save=outfile, error_band=False, ecolor='0.75', capsize=2, elinewidth=1, **run_args)
            lines.plot(save=outfile, **run_args)

        if 'hdf5' in run_args['save_results']:
            self.save_DataLine_HDF5(line, data.name, output_dir, results=results)

            # print(results)

        return results

    def _fit_peaks(self, line, num_curves=1, **run_args):
        # TODO: Use class fit_peaks

        # Usage: lm_result, fit_line, fit_line_extended = self.fit_peaks(line, **run_args)

        line_full = line
        if 'fit_range' in run_args:
            line = line.sub_range(run_args['fit_range'][0], run_args['fit_range'][1])

        import lmfit

        def model(v, x):
            '''Gaussians with constant background.'''
            m = v['m'] * x + v['b']
            for i in range(num_curves):
                m += v['prefactor{:d}'.format(i + 1)] * np.exp(
                    -np.square(x - v['x_center{:d}'.format(i + 1)]) / (2 * (v['sigma{:d}'.format(i + 1)] ** 2)))
            return m

        def func2minimize(params, x, data):

            v = params.valuesdict()
            m = model(v, x)

            return m - data

        params = lmfit.Parameters()

        m = (line.y[-1] - line.y[0]) / (line.x[-1] - line.x[0])
        b = line.y[0] - m * line.x[0]

        # params.add('m', value=0)
        # params.add('b', value=np.min(line.y), min=0, max=np.max(line.y))
        # params.add('m', value=m, min=abs(m)*-10, max=abs(m)*+10)
        params.add('m', value=m, min=np.min([abs(m) * -5, -1e-10]), max=1e-12)  # Slope must be negative
        params.add('b', value=b, min=0)

        xspan = np.max(line.x) - np.min(line.x)
        xpeak, ypeak = line.target_y(np.max(line.y))

        if 'q0' in run_args and run_args['q0'] is not None:
            xpeak, ypeak = line.target_x(run_args['q0'])

        xpeak = run_args['q0']
        prefactor = ypeak - (m * xpeak + b)

        xmin, xmax = xpeak - xspan * 0.4, xpeak + xspan * 0.4
        print(xpeak, xmin, xmax)

        for i in range(num_curves):
            params.add('prefactor{:d}'.format(i + 1), value=prefactor, min=0, max=max(np.max(line.y) * 4, 1e-4))
            params.add('x_center{:d}'.format(i + 1), value=xpeak, min=xmin, max=xmax)
            # params.add('x_center{:d}'.format(i+1), value=-0.009, min=np.min(line.x), max=np.max(line.x))
            params.add('sigma{:d}'.format(i + 1), value=run_args['sigma'], min=0.0002, max=xspan * 0.5)

        lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y))
        # https://lmfit.github.io/lmfit-py/fitting.html
        # lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y), method='nelder')

        if run_args['verbosity'] >= 5:
            print('Fit results (lmfit):')
            lmfit.report_fit(lm_result.params)

        fit_x = line.x
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line = DataLine(x=fit_x, y=fit_y,
                            plot_args={'linestyle': '-', 'color': 'purple', 'marker': None, 'linewidth': 4.0})

        span = abs(np.max(line.x) - np.min(line.x))
        fit_x = np.linspace(np.min(line.x) - 0.5 * span, np.max(line.x) + 0.5 * span, num=500)
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line_extended = DataLine(x=fit_x, y=fit_y,
                                     plot_args={'linestyle': '-', 'color': 'purple', 'alpha': 0.5, 'marker': None,
                                                'linewidth': 2.0})

        return lm_result, fit_line, fit_line_extended

        # End class linecut_qz_fit(linecut_qz)
        ########################################


class roi_ratio(Protocols.Protocol):

    def __init__(self, name='roi_ratio', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.txt'
        self.run_args = {
            'extra': '',
        }
        self.run_args.update(kwargs)

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        results.update(data.roi_q(**run_args))

        # Regions just outside the roi
        qx, dqx, qz, dqz = run_args['qx'], run_args['dqx'], run_args['qz'], run_args['dqz']
        left = data.roi_q(qx=qx - 2 * dqx, dqx=dqx, qz=qz, dqz=dqz)
        right = data.roi_q(qx=qx + 2 * dqx, dqx=dqx, qz=qz, dqz=dqz)
        background = left['stats_total'] + right['stats_total']
        results['stats_total_ratio'] = results['stats_total'] / background

        if 'show_region' in run_args:
            if run_args['show_region'] == 'save':
                outfile = self.get_outfile(data.name, output_dir, ext='_region.png')
                data.plot(save=outfile)
            elif run_args['show_region']:
                data.plot(show=True)

        if run_args['verbosity'] >= 3:
            print('ROI stats:')
            print(results)

        outfile = self.get_outfile(data.name, output_dir)
        with open(outfile, 'w') as fout:
            for k, v in results.items():
                fout.write('{} : {}\n'.format(k, v))

        return results

    # ==============================================


# ============SAXS section=======================
# ==============================================

class linecut_qr_fit_custom(Protocols.linecut_qr):
    '''Takes a linecut along qr, and fits the data to a simple model
    (Gaussian peak with background).'''

    def __init__(self, name='linecut_qr_fit_custom', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {'show_region': False,
                         'plot_range': [None, None, 0, None],
                         'auto_plot_range_fit': True,
                         }
        self.run_args.update(kwargs)

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        line = data.linecut_qr(**run_args)

        if 'show_region' in run_args:
            if run_args['show_region'] == 'save':
                outfile = self.get_outfile(data.name, output_dir, ext='_region.png')
                data.plot(save=outfile)
            elif run_args['show_region']:
                data.plot(show=True)

        # line.smooth(2.0, bins=10)

        outfile = self.get_outfile(data.name, output_dir)
        # line.plot(save=outfile, **run_args)
        outfile = self.get_outfile(data.name, output_dir, ext='.dat')
        line.save_data(outfile)

        # Fit data
        # if 'fit_range' in run_args:
        # line = line.sub_range(run_args['fit_range'][0], run_args['fit_range'][1])
        # line.trim(run_args['fit_range'][0], run_args['fit_range'][1])
        if 'trim_range' in run_args:
            line.trim(run_args['trim_range'][0], run_args['trim_range'][1])

        # lm_result, fit_line, fit_line_extended = self._fit_peaks(line, **run_args)
        lm_result, fit_line, fit_line_extended = self._fit_Guinier(line, **run_args)

        # Save fit results
        fit_name = 'fit_Guinier'
        prefactor_total = 0
        for param_name, param in lm_result.params.items():
            results['{}_{}'.format(fit_name, param_name)] = {'value': param.value, 'error': param.stderr, }
            if 'prefactor' in param_name:
                prefactor_total += np.abs(param.value)

        results['{}_prefactor_total'.format(fit_name)] = prefactor_total
        results['{}_chi_squared'.format(fit_name)] = lm_result.chisqr / lm_result.nfree

        # Plot and save data
        class DataLines_current(DataLines):

            def _plot_extra(self, **plot_args):
                xi, xf, yi, yf = self.ax.axis()
                v_spacing = (yf - yi) * 0.10

                yp = yf
                s = r'$I_0 = \, {:.4f} $'.format(self.results['fit_Guinier_I0']['value'])
                self.ax.text(xf, yp, s, size=20, color='b', verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$R_g = \, {:.4f} \, \mathrm{{\AA}}$'.format(self.results['fit_Guinier_Rg']['value'])
                self.ax.text(xf, yp, s, size=20, color='b', verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$R_g = \, {:.1f} \, \mathrm{{nm}}$'.format(self.results['fit_Guinier_Rg']['value'] / 10)
                self.ax.text(xf, yp, s, size=20, color='b', verticalalignment='top', horizontalalignment='right')

        lines = DataLines_current([line, fit_line, fit_line_extended])
        lines.copy_labels(line)
        lines.results = results

        outfile = self.get_outfile(data.name + '-fit', output_dir, ext='.png')

        # Tweak the plotting range for the fit-plot
        run_args_cur = run_args.copy()
        if run_args['auto_plot_range_fit']:
            run_args_cur['plot_range'] = [run_args['plot_range'][0], run_args['plot_range'][1],
                                          run_args['plot_range'][2], run_args['plot_range'][3]]
            if 'fit_range' in run_args_cur:
                span = abs(run_args['fit_range'][1] - run_args_cur['fit_range'][0])
                run_args_cur['plot_range'][0] = run_args['fit_range'][0] - span * 0.25
                run_args_cur['plot_range'][1] = run_args_cur['fit_range'][1] + span * 0.25

            run_args_cur['plot_range'][2] = 0
            # run_args_cur['plot_range'][3] = max(fit_line.y)*1.3
            run_args_cur['plot_range'][3] = max(max(fit_line.y) * 1.3, max(line.y))

        try:
            # lines.plot(save=outfile, error_band=False, ecolor='0.75', capsize=2, elinewidth=1, **run_args)
            lines.plot(save=outfile, **run_args_cur)
        except ValueError:
            pass

        outfile = self.get_outfile(data.name, output_dir, ext='.dat')
        line.save_data(outfile)

        # print(results)
        return results

    def _fit_Guinier(self, line, **run_args):

        line_full = line
        if 'fit_range' in run_args:
            line = line.sub_range(run_args['fit_range'][0], run_args['fit_range'][1])

        import lmfit

        def model(v, x):
            # Guinier equation:
            # I(q) = I0 * exp(-Rg^2 * q^2 / 3 )
            m = v['I0'] * np.exp(-np.square(v['Rg']) * np.square(x) / 3)
            m += v['b']

            return m

        def func2minimize(params, x, data):

            v = params.valuesdict()
            m = model(v, x)

            return m - data

        params = lmfit.Parameters()

        ys = line.y
        params.add('I0', value=np.max(ys), min=np.max(ys) * 1e-1, max=np.max(ys) * 1e2, vary=True)
        params.add('Rg', value=55, min=0.1, max=1000, vary=True)
        params.add('b', value=np.min(ys), min=0, max=np.min(ys) * 1.1, vary=False)

        # Fit
        lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y))

        if False:
            lm_result = lmfit.minimize(func2minimize, lm_result.params, args=(line.x, line.y))

        if run_args['verbosity'] >= 5:
            print('Fit results (lmfit):')
            lmfit.report_fit(lm_result.params)

        fit_x = line.x
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line = DataLine(x=fit_x, y=fit_y,
                            plot_args={'linestyle': '-', 'color': 'b', 'marker': None, 'linewidth': 4.0})

        # fit_x = np.linspace(np.min(line_full.x), np.max(line_full.x), num=200)
        fit_x = np.linspace(np.average([np.min(line_full.x), np.min(line.x)]), np.average([0, np.max(line.x)]), num=200)
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line_extended = DataLine(x=fit_x, y=fit_y,
                                     plot_args={'linestyle': '-', 'color': 'b', 'alpha': 0.5, 'marker': None,
                                                'linewidth': 2.0})

        return lm_result, fit_line, fit_line_extended

    def _fit_peaks(self, line, num_curves=1, **run_args):

        # Usage: lm_result, fit_line, fit_line_extended = self._fit_peaks(line, **run_args)

        line_full = line
        if 'fit_range' in run_args:
            line = line.sub_range(run_args['fit_range'][0], run_args['fit_range'][1])

        import lmfit

        def model(v, x):

            # Linear background
            m = v['m'] * x + v['b']

            # Power-law background
            m += v['qp'] * np.power(np.abs(x), v['qalpha'])

            # Gaussian peaks
            for i in range(num_curves):
                m += v['prefactor{:d}'.format(i + 1)] * np.exp(
                    -np.square(x - v['x_center{:d}'.format(i + 1)]) / (2 * (v['sigma{:d}'.format(i + 1)] ** 2)))
            return m

        def func2minimize(params, x, data):

            v = params.valuesdict()
            m = model(v, x)

            return m - data

        params = lmfit.Parameters()

        m = (line.y[-1] - line.y[0]) / (line.x[-1] - line.x[0])
        b = line.y[0] - m * line.x[0]

        xs = np.abs(line.x)
        ys = line.y
        qalpha = (np.log(ys[0]) - np.log(ys[-1])) / (np.log(xs[0]) - np.log(xs[-1]))
        qp = np.exp(np.log(ys[0]) - qalpha * np.log(xs[0]))

        if True:
            # Linear background
            params.add('m', value=m, min=abs(m) * -4, max=abs(m) * +4 + 1e-12, vary=False)
            params.add('b', value=b, min=0, max=np.max(line.y) * 100 + 1e-12, vary=False)

            params.add('qp', value=0, vary=False)
            params.add('qalpha', value=1.0, vary=False)

        else:
            # Power-law background
            params.add('m', value=0, vary=False)
            params.add('b', value=0, vary=False)

            params.add('qp', value=qp, vary=False)
            params.add('qalpha', value=qalpha, vary=False)

        xspan = np.max(line.x) - np.min(line.x)
        xpeak, ypeak = line.target_y(np.max(line.y))

        # Best guess for peak position
        if True:
            # Account for power-law scaling (Kratky-like)
            xs = np.asarray(line.x)
            ys = np.asarray(line.y)

            ys = ys * np.power(np.abs(xs), np.abs(qalpha))  # Kratky-like

            # Sort
            indices = np.argsort(ys)
            x_sorted = xs[indices]
            y_sorted = ys[indices]

            target = np.max(ys);

            # Search through y for the target
            idx = np.where(y_sorted >= target)[0][0]
            xpeak = x_sorted[idx]
            ypeak = y_sorted[idx]

            xpeak, ypeak = line.target_x(xpeak)

        prefactor = ypeak - (m * xpeak + b)
        sigma = 0.05 * xspan

        for i in range(num_curves):
            params.add('prefactor{:d}'.format(i + 1), value=prefactor, min=0, max=np.max(line.y) * 1.5, vary=False)
            params.add('x_center{:d}'.format(i + 1), value=xpeak, min=np.min(line.x), max=np.max(line.x), vary=False)
            params.add('sigma{:d}'.format(i + 1), value=sigma, min=0, max=xspan * 0.75, vary=False)

        # Fit only the peak width
        params['sigma1'].vary = True
        lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y))

        if True:
            # Tweak peak position
            lm_result.params['sigma1'].vary = False
            lm_result.params['x_center1'].vary = True
            lm_result = lmfit.minimize(func2minimize, lm_result.params, args=(line.x, line.y))

        if True:
            # Relax entire fit
            lm_result.params['m'].vary = True
            lm_result.params['b'].vary = True
            # lm_result.params['qp'].vary = True
            # lm_result.params['qalpha'].vary = True

            lm_result.params['prefactor1'].vary = True
            lm_result.params['sigma1'].vary = True
            lm_result.params['x_center1'].vary = True
            lm_result = lmfit.minimize(func2minimize, lm_result.params, args=(line.x, line.y))

        if run_args['verbosity'] >= 5:
            print('Fit results (lmfit):')
            lmfit.report_fit(lm_result.params)

        fit_x = line.x
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line = DataLine(x=fit_x, y=fit_y,
                            plot_args={'linestyle': '-', 'color': 'b', 'marker': None, 'linewidth': 4.0})

        # fit_x = np.linspace(np.min(line_full.x), np.max(line_full.x), num=200)
        fit_x = np.linspace(np.average([np.min(line_full.x), np.min(line.x)]), np.average([0, np.max(line.x)]), num=200)
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line_extended = DataLine(x=fit_x, y=fit_y,
                                     plot_args={'linestyle': '-', 'color': 'b', 'alpha': 0.5, 'marker': None,
                                                'linewidth': 2.0})

        return lm_result, fit_line, fit_line_extended

        # End class linecut_qr_fit_custom(linecut_qr)
        ########################################


class linecut_qz_fit(Protocols.linecut_qz):  # TODO: Use class fit_peaks

    def __init__(self, name='linecut_qz_fit', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {
            'show_region': False,
            'plot_range': [None, None, 0, None],
            'show_guides': True,
            'markersize': 0,
            'linewidth': 1.5,
        }
        self.run_args.update(kwargs)

    def output_exists(self, name, output_dir):

        outfile = self.get_outfile(name, output_dir, ext='-fit.png')
        return os.path.isfile(outfile)

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        # Usage example:
        # linecut_qz_fit(show_region=False, show=False, qr=0.009, dq=0.0025, q_mode='qr', fit_range=fit_range, q0_expected=q0_expected, plot_range=[0, 0.08, 0, None]) ,

        results = {}

        line = data.linecut_qz(**run_args)

        if 'show_region' in run_args:
            if run_args['show_region'] == 'save':
                outfile = self.get_outfile(data.name, output_dir, ext='_region.png')
                data.plot(save=outfile)
            elif run_args['show_region']:
                data.plot(show=True)

        # line.smooth(2.0, bins=10)
        if 'trim_range' in run_args:
            line.trim(run_args['trim_range'][0], run_args['trim_range'][1])

        # if 'plots' in run_args['save_results']:
        if False:
            self.label_filename(data, line, **run_args)
            outfile = self.get_outfile(data.name, output_dir)
            line.plot(save=outfile, **run_args)

        if 'txt' in run_args['save_results']:
            outfile = self.get_outfile(data.name, output_dir, ext='.dat')
            line.save_data(outfile)

        if 'incident_angle' not in run_args:
            run_args['incident_angle'] = data.calibration.incident_angle

            import re
            filename_re = re.compile(r'^.+_th(-?\d+\.\d+)_.+$')
            m = filename_re.match(data.name)
            if m:
                run_args['incident_angle'] = float(m.groups()[0])

        if 'verbosity' in run_args and run_args['verbosity'] >= 4:
            print('    Using incident_angle = {:.3f} degrees'.format(run_args['incident_angle']))

        # if 'critical_angle_film' not in run_args:
        # run_args['critical_angle_film'] = 0
        # if 'critical_angle_substrate' not in run_args:
        # run_args['critical_angle_substrate'] = 0

        # Fit data
        lm_result, fit_line, fit_line_extended = self._fit_peaks(line, **run_args)

        # Save fit results
        fit_name = 'fit_peaks'
        prefactor_total = 0
        for param_name, param in lm_result.params.items():
            results['{}_{}'.format(fit_name, param_name)] = {'value': param.value, 'error': param.stderr, }
            if 'prefactor' in param_name:
                prefactor_total += np.abs(param.value)

        results['{}_prefactor_total'.format(fit_name)] = prefactor_total
        results['{}_chi_squared'.format(fit_name)] = lm_result.chisqr / lm_result.nfree

        # Calculate some additional things
        q0 = results['{}_x_center1'.format(fit_name)]['value']
        d = 0.1 * 2. * np.pi / q0
        results['{}_d0'.format(fit_name)] = d
        xi = 0.1 * (2. * np.pi / np.sqrt(2. * np.pi)) / results['{}_sigma1'.format(fit_name)]['value']
        results['{}_grain_size'.format(fit_name)] = xi

        def angle_to_q(two_theta_s_rad):
            k = data.calibration.get_k()
            qz = 2 * k * np.sin(two_theta_s_rad / 2.0)
            return qz

        def q_to_angle(q):
            k = data.calibration.get_k()
            two_theta_s_rad = 2.0 * np.arcsin(q / (2.0 * k))
            return two_theta_s_rad

        if 'critical_angle_film' in run_args:
            # Account for refraction distortion

            theta_incident_rad = np.radians(run_args['incident_angle'])
            theta_c_f_rad = np.radians(run_args['critical_angle_film'])
            # theta_c_s_rad = np.radians(run_args['critical_angle_substrate'])

            alpha_i_rad = np.arccos(np.cos(theta_incident_rad) / np.cos(theta_c_f_rad))

            # Scattering from incident (refracted) beam
            two_theta_s_rad = q_to_angle(q0)
            theta_f_rad = two_theta_s_rad - theta_incident_rad

            alpha_f_rad = np.arccos(np.cos(theta_f_rad) / np.cos(theta_c_f_rad))

            two_alpha_s_rad = alpha_i_rad + alpha_f_rad
            qT = angle_to_q(two_alpha_s_rad)
            results['{}_qT'.format(fit_name)] = qT
            results['{}_dT'.format(fit_name)] = 0.1 * 2. * np.pi / qT

            # Scattering from reflected beam
            two_alpha_s_rad = abs(alpha_f_rad - alpha_i_rad)
            qR = angle_to_q(two_alpha_s_rad)
            results['{}_qR'.format(fit_name)] = qR
            results['{}_dR'.format(fit_name)] = 0.1 * 2. * np.pi / qR

        # Plot and save data
        class DataLines_current(DataLines):

            def _plot_extra(self, **plot_args):

                xi, xf, yi, yf = self.ax.axis()
                y_fit_max = np.max(self.lines[1].y)
                yf = y_fit_max * 2.0
                v_spacing = (yf - yi) * 0.06

                q0 = self.results['fit_peaks_x_center1']['value']
                color = 'purple'

                yp = yf
                s = r'$p = \, {:.4f}$'.format(self.results['fit_peaks_prefactor1']['value'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                s = r'$\chi^2 = \, {:.4f}$'.format(self.results['fit_peaks_chi_squared'])
                self.ax.text(xi, yi, s, size=15, color=color, verticalalignment='bottom', horizontalalignment='left')

                yp -= v_spacing
                s = r'$q_0 = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(q0)
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$d_0 \approx \, {:.1f} \, \mathrm{{nm}}$'.format(self.results['fit_peaks_d0'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$\sigma = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(self.results['fit_peaks_sigma1']['value'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                yp -= v_spacing
                s = r'$\xi \approx \, {:.1f} \, \mathrm{{nm}}$'.format(self.results['fit_peaks_grain_size'])
                self.ax.text(xf, yp, s, size=15, color=color, verticalalignment='top', horizontalalignment='right')

                self.ax.axvline(q0, color=color, linewidth=0.5)
                self.ax.text(q0, yf, '$q_0$', size=20, color=color, horizontalalignment='center',
                             verticalalignment='bottom')

                if 'critical_angle_film' in self.run_args:
                    s = r'$q_T = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$ \n $d_T = \, {:.1f} \, \mathrm{{nm}}$'.format(
                        self.results['fit_peaks_qT'], self.results['fit_peaks_dT'])
                    self.ax.text(q0, y_fit_max, s, size=15, color='b', horizontalalignment='left',
                                 verticalalignment='bottom')
                    self.ax.plot([self.results['fit_peaks_qT'], q0], [y_fit_max, y_fit_max], '-', color='b')

                    s = r'$q_R = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$ \n $d_R = \, {:.1f} \, \mathrm{{nm}}$'.format(
                        self.results['fit_peaks_qR'], self.results['fit_peaks_dR'])
                    self.ax.text(q0, 0, s, size=15, color='r', horizontalalignment='left', verticalalignment='bottom')
                    self.ax.plot([self.results['fit_peaks_qR'], q0], [yi, yi], '-', color='r')

                if self.run_args['show_guides']:
                    # Show various guides of scattering features
                    theta_incident_rad = np.radians(self.run_args['incident_angle'])

                    # Direct
                    qz = 0
                    self.ax.axvline(qz, color='0.25')
                    self.ax.text(qz, yf, r'$\mathrm{D}$', size=20, color='0.25', horizontalalignment='center',
                                 verticalalignment='bottom')

                    # Horizon
                    qz = angle_to_q(theta_incident_rad)
                    l = self.ax.axvline(qz, color='0.5')
                    l.set_dashes([10, 6])
                    self.ax.text(qz, yf, r'$\mathrm{H}$', size=20, color='0.5', horizontalalignment='center',
                                 verticalalignment='bottom')

                    # Specular beam
                    qz = angle_to_q(2 * theta_incident_rad)
                    self.ax.axvline(qz, color='r')
                    self.ax.text(qz, yf, r'$\mathrm{R}$', size=20, color='r', horizontalalignment='center',
                                 verticalalignment='bottom')

                    if 'critical_angle_film' in self.run_args:
                        theta_c_f_rad = np.radians(self.run_args['critical_angle_film'])

                        # Transmitted (direct beam refracted by film)
                        if theta_incident_rad <= theta_c_f_rad:
                            qz = angle_to_q(theta_incident_rad)  # Horizon
                        else:
                            alpha_i_rad = np.arccos(np.cos(theta_incident_rad) / np.cos(theta_c_f_rad))
                            two_theta_s_rad = theta_incident_rad - alpha_i_rad
                            qz = angle_to_q(two_theta_s_rad)
                        l = self.ax.axvline(qz, color='b')
                        l.set_dashes([4, 4])

                        # Yoneda
                        qz = angle_to_q(theta_incident_rad + theta_c_f_rad)
                        self.ax.axvline(qz, color='gold')
                        self.ax.text(qz, yf, r'$\mathrm{Y}_f$', size=20, color='gold', horizontalalignment='center',
                                     verticalalignment='bottom')

                    if 'critical_angle_substrate' in self.run_args:
                        theta_c_s_rad = np.radians(self.run_args['critical_angle_substrate'])

                        # Transmitted (direct beam refracted by substrate)
                        if theta_incident_rad <= theta_c_s_rad:
                            qz = angle_to_q(theta_incident_rad)  # Horizon
                        else:
                            alpha_i_rad = np.arccos(np.cos(theta_incident_rad) / np.cos(theta_c_s_rad))
                            two_theta_s_rad = theta_incident_rad - alpha_i_rad
                            qz = angle_to_q(two_theta_s_rad)
                        self.ax.axvline(qz, color='b')
                        self.ax.text(qz, yf, r'$\mathrm{T}$', size=20, color='b', horizontalalignment='center',
                                     verticalalignment='bottom')

                        # Yoneda
                        qz = angle_to_q(theta_incident_rad + theta_c_s_rad)
                        self.ax.axvline(qz, color='gold')
                        self.ax.text(qz, yf, r'$\mathrm{Y}_s$', size=20, color='gold', horizontalalignment='center',
                                     verticalalignment='bottom')

                self.ax.axis([xi, xf, yi, yf])

        lines = DataLines_current([line, fit_line, fit_line_extended])
        lines.copy_labels(line)
        lines.results = results
        lines.run_args = run_args

        if 'plots' in run_args['save_results']:
            self.label_filename(data, lines, **run_args)
            outfile = self.get_outfile(data.name + '-fit', output_dir, ext='.png')
            # lines.plot(save=outfile, error_band=False, ecolor='0.75', capsize=2, elinewidth=1, **run_args)
            lines.plot(save=outfile, **run_args)

        if 'hdf5' in run_args['save_results']:
            self.save_DataLine_HDF5(line, data.name, output_dir, results=results)

            # print(results)

        return results

    def _fit_peaks(self, line, num_curves=1, **run_args):
        # TODO: Use class fit_peaks

        # Usage: lm_result, fit_line, fit_line_extended = self.fit_peaks(line, **run_args)

        line_full = line
        if 'fit_range' in run_args:
            line = line.sub_range(run_args['fit_range'][0], run_args['fit_range'][1])

        import lmfit

        def model(v, x):
            '''Gaussians with constant background.'''
            m = v['m'] * x + v['b']
            for i in range(num_curves):
                m += v['prefactor{:d}'.format(i + 1)] * np.exp(
                    -np.square(x - v['x_center{:d}'.format(i + 1)]) / (2 * (v['sigma{:d}'.format(i + 1)] ** 2)))
            return m

        def func2minimize(params, x, data):

            v = params.valuesdict()
            m = model(v, x)

            return m - data

        params = lmfit.Parameters()

        m = (line.y[-1] - line.y[0]) / (line.x[-1] - line.x[0])
        b = line.y[0] - m * line.x[0]

        # params.add('m', value=0)
        # params.add('b', value=np.min(line.y), min=0, max=np.max(line.y))
        # params.add('m', value=m, min=abs(m)*-10, max=abs(m)*+10)
        params.add('m', value=m, min=np.min([abs(m) * -5, -1e-10]), max=1e-12)  # Slope must be negative
        params.add('b', value=b, min=0)

        xspan = np.max(line.x) - np.min(line.x)
        xpeak, ypeak = line.target_y(np.max(line.y))

        if 'q0' in run_args and run_args['q0'] is not None:
            xpeak, ypeak = line.target_x(run_args['q0'])

        xpeak = run_args['q0']
        prefactor = ypeak - (m * xpeak + b)

        xmin, xmax = xpeak - xspan * 0.4, xpeak + xspan * 0.4
        print(xpeak, xmin, xmax)

        for i in range(num_curves):
            params.add('prefactor{:d}'.format(i + 1), value=prefactor, min=0, max=max(np.max(line.y) * 4, 1e-4))
            params.add('x_center{:d}'.format(i + 1), value=xpeak, min=xmin, max=xmax)
            # params.add('x_center{:d}'.format(i+1), value=-0.009, min=np.min(line.x), max=np.max(line.x))
            params.add('sigma{:d}'.format(i + 1), value=run_args['sigma'], min=0.0002, max=xspan * 0.5)

        lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y))
        # https://lmfit.github.io/lmfit-py/fitting.html
        # lm_result = lmfit.minimize(func2minimize, params, args=(line.x, line.y), method='nelder')

        if run_args['verbosity'] >= 5:
            print('Fit results (lmfit):')
            lmfit.report_fit(lm_result.params)

        fit_x = line.x
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line = DataLine(x=fit_x, y=fit_y,
                            plot_args={'linestyle': '-', 'color': 'purple', 'marker': None, 'linewidth': 4.0})

        span = abs(np.max(line.x) - np.min(line.x))
        fit_x = np.linspace(np.min(line.x) - 0.5 * span, np.max(line.x) + 0.5 * span, num=500)
        fit_y = model(lm_result.params.valuesdict(), fit_x)
        fit_line_extended = DataLine(x=fit_x, y=fit_y,
                                     plot_args={'linestyle': '-', 'color': 'purple', 'alpha': 0.5, 'marker': None,
                                                'linewidth': 2.0})

        return lm_result, fit_line, fit_line_extended

        # End class linecut_qz_fit(linecut_qz)
        ########################################


class roi_ratio(Protocols.Protocol):

    def __init__(self, name='roi_ratio', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.txt'
        self.run_args = {
            'extra': '',
        }
        self.run_args.update(kwargs)

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        results.update(data.roi_q(**run_args))

        # Regions just outside the roi
        qx, dqx, qz, dqz = run_args['qx'], run_args['dqx'], run_args['qz'], run_args['dqz']
        left = data.roi_q(qx=qx - 2 * dqx, dqx=dqx, qz=qz, dqz=dqz)
        right = data.roi_q(qx=qx + 2 * dqx, dqx=dqx, qz=qz, dqz=dqz)
        background = left['stats_total'] + right['stats_total']
        results['stats_total_ratio'] = results['stats_total'] / background

        if 'show_region' in run_args:
            if run_args['show_region'] == 'save':
                outfile = self.get_outfile(data.name, output_dir, ext='_region.png')
                data.plot(save=outfile)
            elif run_args['show_region']:
                data.plot(show=True)

        if run_args['verbosity'] >= 3:
            print('ROI stats:')
            print(results)

        outfile = self.get_outfile(data.name, output_dir)
        with open(outfile, 'w') as fout:
            for k, v in results.items():
                fout.write('{} : {}\n'.format(k, v))

        return results

    # Cheng-Chu's two peak splitting fit code


class circular_average_q2I_custom(Protocols.circular_average_q2I):

    def __init__(self, name='circular_average_q2I_custom', **kwargs):

        self.name = self.__class__.__name__ if name is None else name

        self.default_ext = '.png'
        self.run_args = {
            'bins_relative': 1.0,
            'markersize': 0,
            'linewidth': 1.5,
            'qn_power': 2.0,
            'num_curves': 1,  # For (optional) fitting
        }
        self.run_args.update(kwargs)

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        line = data.circular_average_q_bin(error=True, bins_relative=run_args['bins_relative'])

        if 'trim_range' in run_args:
            line.trim(run_args['trim_range'][0], run_args['trim_range'][1])

        if run_args['qn_power'] == 2.0:
            line.y *= np.square(line.x)
            line.y_label = 'q^2*I(q)'
            line.y_rlabel = r'$q^2 I(q) \, (\AA^{-2} \mathrm{counts/pixel})$'

        else:
            line.y *= np.power(line.x, run_args['qn_power'])
            line.y_label = 'q^n*I(q)'
            line.y_rlabel = r'$q^n I(q) \, (\AA^{-n} \mathrm{counts/pixel})$'

        results['qn_power'] = run_args['qn_power']

        if 'plots' in run_args['save_results']:
            self.label_filename(data, line, **run_args)
            outfile = self.get_outfile(data.name, output_dir, ext='_q2I{}'.format(self.default_ext))
            line.plot(save=outfile, **run_args)

        if 'txt' in run_args['save_results']:
            outfile = self.get_outfile(data.name, output_dir, ext='_q2I.dat')
            line.save_data(outfile)

        if 'hdf5' in run_args['save_results']:
            self.save_DataLine_HDF5(line, data.name, output_dir, results=results)

        return results

    def output_exists(self, name, output_dir):

        if 'file_extension' in self.run_args:
            ext = '_q2I{}'.format(self.run_args['file_extension'])
        else:
            ext = '_q2I{}'.format(self.default_ext)

        outfile = self.get_outfile(name, output_dir, ext=ext)
        return os.path.isfile(outfile)


class fit_peaks_custom(Protocols.circular_average_q2I):

    def _fit(self, line, results, **run_args):

        # Fit
        fit_result, fit_line, fit_line_extended, fit_line_curves = self._fit_peaks(line, **run_args)

        fit_name = 'fit_peaks'

        # Store background parameters (these don't need sorting)
        for i, param_name in enumerate(['m', 'b', 'qp', 'qalpha']):
            results['{}_{}'.format(fit_name, param_name)] = {
                'value': fit_result['params'][i],
                'error': fit_result['perr'][i],
            }

        # Store sorted peak parameters using the peak_sort_map
        # This ensures q1 < q2 < q3... and that p1, q1, sigma1 all refer to the same peak
        prefactor_total = 0
        for sorted_idx in range(run_args['num_curves']):
            # Get the original fitted index for this sorted peak
            _, original_idx, _ = fit_result['peak_sort_map'][sorted_idx]

            # Store sorted parameters with conventional naming (1=leftmost, 2=next, etc.)
            for param_type in ['x_center', 'prefactor', 'sigma']:
                # Map parameter type to index offset in params array
                if param_type == 'prefactor':
                    param_idx = 4 + (original_idx - 1) * 3 + 0
                elif param_type == 'x_center':
                    param_idx = 4 + (original_idx - 1) * 3 + 1
                else:  # sigma
                    param_idx = 4 + (original_idx - 1) * 3 + 2

                sorted_param_name = '{}{:d}'.format(param_type, sorted_idx + 1)

                results['{}_{}'.format(fit_name, sorted_param_name)] = {
                    'value': fit_result['params'][param_idx],
                    'error': fit_result['perr'][param_idx],
                }

                print(f'Error: {fit_name}_{sorted_param_name} = {fit_result["perr"][param_idx]}')

                if param_type == 'prefactor':
                    prefactor_total += np.abs(fit_result['params'][param_idx])

        results['{}_prefactor_total'.format(fit_name)] = prefactor_total
        results['{}_chi_squared'.format(fit_name)] = fit_result['chisqr'] / fit_result['nfree']

        # Calculate some additional things (now using sorted peaks)
        for i in range(run_args['num_curves']):
            q = results['{}_x_center{}'.format(fit_name, i + 1)]['value']
            d = 0.1 * 2. * np.pi / q
            err = results['{}_x_center{}'.format(fit_name, i + 1)]['error']
            if err is None or np.isnan(err):
                err = 0
            d_err = err * (d / q)
            results['{}_d0{}'.format(fit_name, i + 1)] = {'value': d, 'error': d_err}

            sigma = results['{}_sigma{}'.format(fit_name, i + 1)]['value']
            if 'instrumental_resolution' in run_args:
                sigma = np.sqrt(np.square(sigma) - np.square(run_args['instrumental_resolution']))
            xi = 0.1 * (2. * np.pi / np.sqrt(2. * np.pi)) / sigma
            err = results['{}_sigma{}'.format(fit_name, i + 1)]['error']
            if err is None or np.isnan(err):
                err = 0
            xi_err = err * (xi / sigma)
            results['{}_grain_size{}'.format(fit_name, i + 1)] = {'value': xi, 'error': xi_err}

        results['{}_d0'.format(fit_name)] = results['{}_d01'.format(fit_name)]
        results['{}_grain_size'.format(fit_name)] = results['{}_grain_size1'.format(fit_name)]

        # Plot and save data
        class DataLines_current(DataLines):

            def _plot_extra(self, **plot_args):

                xi, xf, yi, yf = self.ax.axis()

                if 'fit_range' in self._run_args:
                    xstart, xend = self._run_args['fit_range']
                    line = self.lines[0].sub_range(xstart, xend)
                else:
                    line = self.lines[0]

                yf = np.max(line.y) * 1.5
                self.ax.axis([xi, xf, yi, yf])

                color = 'b'
                font_size = self._run_args['font_size'] if 'font_size' in self._run_args else 18
                v_spacing = (yf - yi) * 0.065 * (font_size / 20)

                s = r'$\chi^2 = \, {:.4g}$'.format(self.results['fit_peaks_chi_squared'])
                self.ax.text(xi, yi, s, size=font_size, color=color, verticalalignment='bottom',
                             horizontalalignment='left')

                for i in range(self._run_args['num_curves']):

                    self.ax.axvline(self.results['fit_peaks_x_center{}'.format(i + 1)]['value'], linewidth=1,
                                    color=color, alpha=0.5)

                    if i <= 1:
                        yp = yf
                    else:
                        yp -= v_spacing * 1.5
                    if i == 0:
                        ha, xp = 'right', xf
                    else:
                        ha, xp = 'left', xi

                    s = r'$p_{{ {:d} }} = \, {:.3g}$'.format(i + 1, self.results['fit_peaks_prefactor{}'.format(i + 1)][
                        'value'])
                    self.ax.text(xp, yp, s, size=font_size, color=color, verticalalignment='top',
                                 horizontalalignment=ha)

                    yp -= v_spacing
                    s = r'$q_{{ {:d} }} = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(
                        i + 1, self.results['fit_peaks_x_center{}'.format(i + 1)]['value'])
                    self.ax.text(xp, yp, s, size=font_size, color=color, verticalalignment='top',
                                 horizontalalignment=ha)

                    yp -= v_spacing
                    s = r'$d_{{ {:d} }} \approx \, {:.1f} \, \mathrm{{nm}}$'.format(
                        i + 1, self.results['fit_peaks_d0{}'.format(i + 1)]['value'])
                    self.ax.text(xp, yp, s, size=font_size, color=color, verticalalignment='top',
                                 horizontalalignment=ha)

                    yp -= v_spacing
                    s = r'$\sigma_{{ {:d} }} = \, {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(
                        i + 1, self.results['fit_peaks_sigma{}'.format(i + 1)]['value'])
                    self.ax.text(xp, yp, s, size=font_size, color=color, verticalalignment='top',
                                 horizontalalignment=ha)

                    yp -= v_spacing
                    s = r'$\xi_{{ {:d} }} \approx \, {:.1f} \, \mathrm{{nm}}$'.format(
                        i + 1, self.results['fit_peaks_grain_size{}'.format(i + 1)]['value'])
                    self.ax.text(xp, yp, s, size=font_size, color=color, verticalalignment='top',
                                 horizontalalignment=ha)

        lines = DataLines_current([line, fit_line, fit_line_extended])
        if 'num_curves' in run_args and run_args['num_curves'] > 1 and 'show_curves' in run_args and run_args[
            'show_curves']:
            for curve in fit_line_curves:
                lines.add_line(curve)
        lines.results = results
        lines._run_args = run_args
        lines.copy_labels(line)

        return lines

    def _fit_peaks(self, line, q0=None, num_curves=1, **run_args):
        from scipy.optimize import least_squares
        import numpy as np

        line_full = line
        if 'fit_range' in run_args:
            line = line.sub_range(run_args['fit_range'][0], run_args['fit_range'][1])

        def model(params, x):
            """
            Model function
            params array structure: [m, b, qp, qalpha, p1, q1, s1, p2, q2, s2, ...]
            """
            # Linear background
            background = params[0] * x + params[1]
            # Power-law background
            background += params[2] * np.power(np.abs(x), params[3])

            # Gaussian peaks
            result = background.copy()
            for i in range(num_curves):
                idx = 4 + i * 3  # Start index for this peak's parameters
                prefactor = params[idx]
                x_center = params[idx + 1]
                sigma = params[idx + 2]
                result += prefactor * np.exp(-np.square(x - x_center) / (2 * sigma ** 2)) #CZ sigma is the Gaussian standard deviation in q units (A^-1)

            return result

        def residuals(params, x, y):
            """Residual function for least_squares"""
            return model(params, x) - y

        # Initialize background parameters (same as before)
        m = (line.y[-1] - line.y[0]) / (line.x[-1] - line.x[0])
        b = line.y[0] - m * line.x[0]

        xs = np.abs(line.x)
        ys = line.y

        # Safer initialization for power-law parameters
        try:
            qalpha = (np.log(ys[0]) - np.log(ys[-1])) / (np.log(xs[0]) - np.log(xs[-1]))
            qp = np.exp(np.log(ys[0]) - qalpha * np.log(xs[0]))
        except:
            qalpha = 1.0
            qp = 0.0

        xspan = np.max(line.x) - np.min(line.x)
        xpeak, ypeak = line.target_y_max()

        # Best guess for peak position
        xs = np.asarray(line.x)
        ys = np.asarray(line.y)

        if isinstance(q0, (list, tuple, np.ndarray)):
            q0s = q0
            q0 = q0s[0]
        else:
            q0s = None

        if q0 is not None:
            indices = np.argsort(xs)
            x_sorted = xs[indices]
            y_sorted = ys[indices]
            idx = np.where(x_sorted >= q0)[0][0]
            xpeak = x_sorted[idx]
            ypeak = y_sorted[idx]
        else:
            indices = np.argsort(ys)
            x_sorted = xs[indices]
            y_sorted = ys[indices]
            target = np.max(ys)
            idx = np.where(y_sorted >= target)[0][0]
            xpeak = x_sorted[idx]
            ypeak = y_sorted[idx]

        xpeak, ypeak = line.target_x(xpeak)

        prefactor = ypeak - (m * xpeak + b)
        if 'sigma' in run_args:
            sigma = run_args['sigma']
        else:
            sigma = 0.1 * xspan

        # Build initial parameter array and bounds
        # Order: [m, b, qp, qalpha, p1, q1, s1, p2, q2, s2, ...]
        x0 = [m, b, 0, 1.0]  # Background parameters (keeping power-law off)
        lower_bounds = [abs(m) * -10, -np.inf, 0, 0]
        upper_bounds = [abs(m) * +10 + 1e-12, np.inf, np.inf, np.inf]

        for i in range(num_curves):
            if i == 0:
                xpos = xpeak
            else:
                if q0s is not None and len(q0s) > i:
                    xpos = q0s[i]
                else:
                    xpos = np.min(line.x) + (xspan / num_curves) * i

            x0.extend([prefactor, xpos, sigma])
            lower_bounds.extend([0, np.min(line.x), 0.00001])
            upper_bounds.extend([max(np.max(line.y) * 1.5, 0) + 1e-12, np.max(line.x), xspan * 0.5])

        x0 = np.array(x0)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        # CRITICAL FIX: Ensure x0 is within bounds
        def enforce_bounds(x0, lb, ub):
            """Ensure initial values are strictly within bounds"""
            x0_safe = x0.copy()
            for i in range(len(x0)):
                if not np.isfinite(lb[i]):
                    lb[i] = -1e10  # Replace -inf with large negative
                if not np.isfinite(ub[i]):
                    ub[i] = 1e10  # Replace +inf with large positive

                # Ensure x0_safe is within bounds
                if x0_safe[i] <= lb[i]:
                    x0_safe[i] = lb[i] + abs(lb[i]) * 1e-6 + 1e-10
                if x0_safe[i] >= ub[i]:
                    x0_safe[i] = ub[i] - abs(ub[i]) * 1e-6 - 1e-10

                # Final check: ensure lb < x0 < ub
                x0_safe[i] = np.clip(x0_safe[i], lb[i] + 1e-10, ub[i] - 1e-10)

            return x0_safe, lb, ub

        x0, lower_bounds, upper_bounds = enforce_bounds(x0, lower_bounds, upper_bounds)

        # Debug output (optional - remove after testing)
        if run_args.get('verbosity', 0) >= 4:
            print("\nInitial parameters and bounds check:")
            param_names = ['m', 'b', 'qp', 'qalpha']
            for i in range(num_curves):
                param_names.extend([f'p{i + 1}', f'q{i + 1}', f's{i + 1}'])

            for i, name in enumerate(param_names):
                print(f"  {name}: lb={lower_bounds[i]:.6e}, x0={x0[i]:.6e}, ub={upper_bounds[i]:.6e}")
                if not (lower_bounds[i] < x0[i] < upper_bounds[i]):
                    print(f"    ^^^ WARNING: Infeasible!")

        # Create bounds with a small epsilon to avoid lb == ub
        def create_bounds_mask(vary_mask, current_params):
            """Create bounds based on which parameters should vary"""
            lb = lower_bounds.copy()
            ub = upper_bounds.copy()
            epsilon = 1e-10  # Small value to ensure lb < ub

            for i, vary in enumerate(vary_mask):
                if not vary:
                    # "Fix" parameter by setting very tight bounds around current value
                    val = current_params[i]
                    # Ensure bounds don't violate original constraints
                    lb[i] = max(val - epsilon, lower_bounds[i])
                    ub[i] = min(val + epsilon, upper_bounds[i])

                    # Final safety check
                    if lb[i] >= ub[i]:
                        ub[i] = lb[i] + epsilon

            return lb, ub

        # Stage 1: Fit only the first peak's sigma
        vary_mask = [False, False, False, False]  # Background fixed
        for i in range(num_curves):
            if i == 0:
                vary_mask.extend([False, False, True])  # Only sigma1 varies
            else:
                vary_mask.extend([False, False, False])  # Other peaks fixed

        lb, ub = create_bounds_mask(vary_mask, x0)

        try:
            result = least_squares(residuals, x0, args=(line.x, line.y),
                                   bounds=(lb, ub),
                                   method='trf',
                                   ftol=1e-10, xtol=1e-10,
                                   max_nfev=5000)
            x0 = result.x
        except ValueError as e:
            print(f"Stage 1 fit failed: {e}")
            print("Trying with looser bounds...")
            # Fallback: skip stage 1, go directly to full fit
            pass

        if True:
            # Stage 2: Tweak peak position
            vary_mask = [False, False, False, False]  # Background fixed
            for i in range(num_curves):
                if i == 0:
                    vary_mask.extend([False, True, False])  # Only x_center1 varies
                else:
                    vary_mask.extend([False, False, False])

            lb, ub = create_bounds_mask(vary_mask, x0)

            try:
                result = least_squares(residuals, x0, args=(line.x, line.y),
                                       bounds=(lb, ub),
                                       method='trf',
                                       ftol=1e-10, xtol=1e-10,
                                       max_nfev=5000)
                x0 = result.x
            except ValueError as e:
                print(f"Stage 2 fit failed: {e}")
                print("Skipping to stage 3...")
                pass

        if True:
            # Stage 3: Relax entire fit
            vary_mask = [True, True, False, False]  # Background varies, power-law fixed
            for i in range(num_curves):
                vary_mask.extend([True, True, True])  # All peak parameters vary

            lb, ub = create_bounds_mask(vary_mask, x0)

            result = least_squares(residuals, x0, args=(line.x, line.y),
                                   bounds=(lb, ub),
                                   method='trf',
                                   ftol=1e-10, xtol=1e-10,
                                   max_nfev=5000)

        # Extract final parameters
        params_final = result.x

        # Compute covariance matrix using SVD (more robust than direct inverse)
        try:
            J = result.jac
            # Use SVD for numerical stability
            U, s, Vt = np.linalg.svd(J, full_matrices=False)

            # Filter out small singular values
            threshold = np.finfo(float).eps * max(J.shape) * s[0]
            s_inv = np.where(s > threshold, 1.0 / s, 0)

            # Compute covariance: (J^T J)^-1 = V * diag(1/s^2) * V^T
            pcov = (Vt.T * s_inv ** 2) @ Vt

            # Adjust for residual variance
            # DOF = n_data - n_params
            n_data = len(line.x)
            n_params = len(params_final)
            dof = max(n_data - n_params, 1)

            s_sq = np.sum(result.fun ** 2) / dof  # Residual variance
            pcov = pcov * s_sq

            # Extract standard errors
            perr = np.sqrt(np.diag(pcov))

            # Check for invalid uncertainties
            if np.any(np.isnan(perr)) or np.any(np.isinf(perr)):
                print("WARNING: Some uncertainties are NaN/Inf after SVD covariance calculation")
                print(f"Condition number of Jacobian: {np.linalg.cond(J):.2e}")

                # Try to identify which parameters have issues
                for i, err in enumerate(perr):
                    if np.isnan(err) or np.isinf(err):
                        print(f"  Parameter {i} (value={params_final[i]:.6f}): stderr={err}")

        except Exception as e:
            print(f"Covariance calculation failed: {e}")
            perr = np.full(len(params_final), np.nan)

        # **CREATE SORTED PEAK MAPPING**
        peak_sort_map = []

        if num_curves >= 2:
            peak_positions = []
            for i in range(num_curves):
                idx = 4 + i * 3 + 1  # x_center index
                peak_positions.append({
                    'original_idx': i + 1,
                    'x_center': params_final[idx],
                })

            peak_positions_sorted = sorted(peak_positions, key=lambda x: x['x_center'])

            for sorted_idx, peak in enumerate(peak_positions_sorted):
                peak_sort_map.append((sorted_idx + 1, peak['original_idx'], peak['x_center']))

            if run_args.get('verbosity', 0) >= 3:
                print(f"Peak ordering by x_center:")
                for sorted_idx, orig_idx, x_center in peak_sort_map:
                    print(f"  q{sorted_idx} -> fitted peak {orig_idx} at x_center={x_center:.4f}")
        else:
            peak_sort_map.append((1, 1, params_final[5]))  # x_center1 is at index 5

        if run_args.get('verbosity', 0) >= 5:
            print('Fit results (scipy.least_squares):')

            print(f"  Success: {result.success}")
            print(f"  Message: {result.message}")
            print(f"  Cost: {result.cost:.6e}")
            print(f"  Parameters:")
            param_names = ['m', 'b', 'qp', 'qalpha']
            for i in range(num_curves):
                param_names.extend([f'prefactor{i + 1}', f'x_center{i + 1}', f'sigma{i + 1}'])
            for name, val, err in zip(param_names, params_final, perr):
                print(f"    {name}: {val:.6e} +/- {err:.6e}")

        # Generate fit lines
        fit_x = line.x
        fit_y = model(params_final, fit_x)
        fit_line = DataLine(x=fit_x, y=fit_y,
                            plot_args={'linestyle': '-', 'color': 'b', 'marker': None, 'linewidth': 4.0})

        x_span = abs(np.max(line.x) - np.min(line.x))
        fit_x_extended = np.linspace(np.min(line.x) - x_span, np.max(line.x) + x_span, num=2000)
        fit_y_extended = model(params_final, fit_x_extended)
        fit_line_extended = DataLine(x=fit_x_extended, y=fit_y_extended,
                                     plot_args={'linestyle': '-', 'color': 'b', 'alpha': 0.5, 'marker': None,
                                                'linewidth': 2.0})

        # Generate component curves - using SORTED order
        fit_line_curves = []
        for sorted_idx, orig_idx, _ in peak_sort_map:
            # Create a copy of parameters with only one peak active
            params_single = params_final.copy()

            for j in range(num_curves):
                idx = 4 + j * 3  # prefactor index for peak j+1
                if j + 1 == orig_idx:
                    # Keep this peak's prefactor
                    pass
                else:
                    # Zero out other peaks
                    params_single[idx] = 0

            fit_y_curve = model(params_single, fit_x_extended)
            fit_line_curve = DataLine(x=fit_x_extended, y=fit_y_curve,
                                      plot_args={'linestyle': '-', 'color': 'purple', 'alpha': 0.5,
                                                 'marker': None, 'linewidth': 1.0})
            fit_line_curves.append(fit_line_curve)

        # Calculate chi-squared
        residuals_final = result.fun
        chisqr = np.sum(residuals_final ** 2)
        ndata = len(line.x)
        nparams = len(params_final)
        nfree = max(ndata - nparams, 1)

        # Package results in a dictionary similar to lmfit format
        fit_result = {
            'params': params_final,
            'perr': perr,
            'peak_sort_map': peak_sort_map,
            'success': result.success,
            'message': result.message,
            'chisqr': chisqr,
            'nfree': nfree,
            'ndata': ndata,
            'nparams': nparams,
            'scipy_result': result  # Store full scipy result for debugging
        }

        return fit_result, fit_line, fit_line_extended, fit_line_curves


class circular_average_q2I_fit_custom(circular_average_q2I_custom, fit_peaks_custom):

    @tools.run_default
    def run(self, data, output_dir, **run_args):

        results = {}

        line = data.circular_average_q_bin(error=True, bins_relative=run_args['bins_relative'])

        if run_args['qn_power'] == 2.0:
            line.y *= np.square(line.x)
            line.y_label = 'q^2*I(q)'
            line.y_rlabel = r'$q^2 I(q) \, (\AA^{-2} \mathrm{counts/pixel})$'

        else:
            line.y *= np.power(line.x, run_args['qn_power'])
            line.y_label = 'q^n*I(q)'
            line.y_rlabel = r'$q^n I(q) \, (\AA^{-n} \mathrm{counts/pixel})$'

        results['qn_power'] = run_args['qn_power']

        if 'txt' in run_args['save_results']:
            outfile = self.get_outfile(data.name, output_dir, ext='_q2I.dat')
            line.save_data(outfile)

        if 'trim_range' in run_args:
            line.trim(run_args['trim_range'][0], run_args['trim_range'][1])

        lines = self._fit(line, results, **run_args)

        if 'save_fit' in run_args and run_args['save_fit']:
            # lines.lines contains: line, fit_line, fit_line_extended, fit_curve1, fit_curve2, ...
            if 'show_curves' in run_args and run_args['show_curves']:
                for i, line in enumerate(lines.lines[2:]):
                    outfile = self.get_outfile(data.name, output_dir, ext='_q2I_fit{}.dat'.format(i))
                    line.save_data(outfile)
            else:
                outfile = self.get_outfile(data.name, output_dir, ext='_q2I_fit.dat')
                lines.lines[1].save_data(outfile)

        if 'plots' in run_args['save_results']:
            self.label_filename(data, lines, **run_args)
            outfile = self.get_outfile(data.name, output_dir, ext='_q2I{}'.format(self.default_ext))
            lines.plot(save=outfile, **run_args)

        if 'hdf5' in run_args['save_results']:
            self.save_DataLine_HDF5(line, data.name, output_dir, results=results)

        return results


class circular_average_q2I_fit_sorted(Protocols.circular_average_q2I_fit):
    """Subclass of circular_average_q2I_fit that sorts fitted peak positions in ascending q order.
    Useful when num_curves>=2 and the optimizer may return peaks in arbitrary order."""

    def _fit(self, line, results, **run_args):
        # Run the standard lmfit-based fitting
        lines = super()._fit(line, results, **run_args)

        num_curves = run_args.get('num_curves', 1)
        if num_curves >= 2:
            fit_name = 'fit_peaks'

            # Collect each peak's parameter set
            peaks = []
            for i in range(num_curves):
                peaks.append({
                    'x_center': results.get('{}_x_center{}'.format(fit_name, i + 1)),
                    'prefactor': results.get('{}_prefactor{}'.format(fit_name, i + 1)),
                    'sigma': results.get('{}_sigma{}'.format(fit_name, i + 1)),
                    'd0': results.get('{}_d0{}'.format(fit_name, i + 1)),
                    'grain_size': results.get('{}_grain_size{}'.format(fit_name, i + 1)),
                })

            # Sort ascending by q (x_center value)
            peaks_sorted = sorted(peaks, key=lambda p: p['x_center']['value'])

            # Write sorted peaks back into results
            for i, peak in enumerate(peaks_sorted):
                results['{}_x_center{}'.format(fit_name, i + 1)] = peak['x_center']
                results['{}_prefactor{}'.format(fit_name, i + 1)] = peak['prefactor']
                results['{}_sigma{}'.format(fit_name, i + 1)] = peak['sigma']
                results['{}_d0{}'.format(fit_name, i + 1)] = peak['d0']
                results['{}_grain_size{}'.format(fit_name, i + 1)] = peak['grain_size']

            # Refresh single-peak aliases to point to the lowest-q peak
            results['{}_d0'.format(fit_name)] = results['{}_d01'.format(fit_name)]
            results['{}_grain_size'.format(fit_name)] = results['{}_grain_size1'.format(fit_name)]

        return lines


class circular_average_q2I_fit_FWHM(circular_average_q2I_fit_sorted):
    """Same Gaussian peak fit as circular_average_q2I_fit_sorted, but also
    reports the FWHM (in q, A^-1) for each peak: FWHM = 2*sqrt(2*ln2)*sigma."""

    def _fit(self, line, results, **run_args):
        # Run the normal (sorted) fit; this populates sigma1, sigma2, ... in `results`
        lines = super()._fit(line, results, **run_args)

        fit_name = 'fit_peaks'
        FWHM_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~ 2.35482

        num_curves = run_args.get('num_curves', 1)
        for i in range(num_curves):
            sigma_res = results.get('{}_sigma{}'.format(fit_name, i + 1))
            if sigma_res is None:
                continue

            sigma = sigma_res['value']
            sigma_err = sigma_res.get('error')

            fwhm = FWHM_factor * sigma
            if sigma_err is None or np.isnan(sigma_err):
                fwhm_err = 0
            else:
                fwhm_err = FWHM_factor * sigma_err  # linear scaling

            results['{}_fwhm{}'.format(fit_name, i + 1)] = {'value': fwhm, 'error': fwhm_err}

        # Single-peak alias (points at the lowest-q peak)
        if '{}_fwhm1'.format(fit_name) in results:
            results['{}_fwhm'.format(fit_name)] = results['{}_fwhm1'.format(fit_name)]

        return lines


from scipy.signal import savgol_filter, find_peaks, peak_widths


class linecut_qr_kratky(Protocols.linecut_qr):
    '''Takes a linecut along qr and plots Kratky plot (q^2*I) to identify peaks.'''

    def __init__(self, name='linecut_qr_kratky', **kwargs):
        self.name = self.__class__.__name__ if name is None else name
        self.default_ext = '.png'
        self.run_args = {
            'show_region': False,
            'plot_range': [None, None, 0, None],
            'qalpha': 2.0,
            'peak_threshold': 0.1,
            'peak_search_range': [None, None],
            'smooth': True,
            'window_length': 21,
            'polyorder': 3,
            'error_method': 'peak_width',  # ADD THIS LINE
            'constant_error': 0.001,  # ADD THIS LINE
            'no_peak_error_multiplier': 10,  # ADD THIS LINE
        }
        self.run_args.update(kwargs)

    @tools.run_default
    def run(self, data, output_dir, **run_args):
        results = {}

        # Get linecut
        line = data.linecut_qr(**run_args)

        if 'trim_range' in run_args:
            line.trim(run_args['trim_range'][0], run_args['trim_range'][1])

        # Apply Kratky transformation
        qalpha = run_args.get('qalpha', 2.0)
        x = np.asarray(line.x)
        y = np.asarray(line.y)

        # Apply smoothing if requested
        smooth = run_args.get('smooth', True)
        if smooth:
            window_length = run_args.get('window_length', 21)
            polyorder = run_args.get('polyorder', 3)

            # Ensure window_length is odd and less than data length
            if window_length % 2 == 0:
                window_length += 1
            window_length = min(window_length, len(y) - 1)
            if window_length % 2 == 0:
                window_length -= 1

            y_smooth = savgol_filter(y, window_length, polyorder)
        else:
            y_smooth = y

        # Transformed intensity (using smoothed data)
        y_kratky = y_smooth * np.power(np.abs(x), qalpha)
        y_kratky_raw = y * np.power(np.abs(x), qalpha)  # Keep raw for plotting

        # Define peak search mask
        peak_search_range = run_args.get('peak_search_range', [None, None])
        q_min = peak_search_range[0] if peak_search_range[0] is not None else np.min(x)
        q_max = peak_search_range[1] if peak_search_range[1] is not None else np.max(x)

        mask = (x >= q_min) & (x <= q_max)

        # Find peaks using scipy find_peaks within masked region
        y_kratky_masked = y_kratky[mask]
        x_masked = x[mask]

        # ADD THESE TWO LINES HERE:
        error_method = run_args.get('error_method', 'peak_width')
        constant_error = run_args.get('constant_error', 0.001)
        no_peak_error_multiplier = run_args.get('no_peak_error_multiplier', 10)

        if len(y_kratky_masked) > 0:
            # Find all peaks
            peaks_indices, properties = find_peaks(y_kratky_masked)

            if len(peaks_indices) > 0:
                # PEAK FOUND - Take the highest peak
                highest_peak_idx = peaks_indices[np.argmax(y_kratky_masked[peaks_indices])]
                q_peak = x_masked[highest_peak_idx]
                I_peak = y_kratky_masked[highest_peak_idx]

                # Calculate uncertainty based on method
                if error_method == 'peak_width':
                    try:
                        widths, width_heights, left_ips, right_ips = peak_widths(
                            y_kratky_masked, [highest_peak_idx], rel_height=0.5
                        )
                        dq = np.mean(np.diff(x_masked))
                        peak_width = widths[0] * dq
                        q_peak_error = peak_width / 2  # FWHM/2
                    except:
                        q_peak_error = constant_error

                elif error_method == 'snr':
                    I_baseline = np.median(y_kratky_masked)
                    I_noise = np.std(y_kratky_masked[y_kratky_masked < I_baseline])
                    I_signal = I_peak - I_baseline
                    SNR = I_signal / I_noise if I_noise > 0 else 100
                    dq = np.mean(np.diff(x_masked))
                    q_peak_error = dq / np.sqrt(max(SNR, 1))
                    results['SNR'] = SNR

                elif error_method == 'bootstrap':
                    q_peak_boot, q_peak_error = self._bootstrap_uncertainty(
                        x_masked, y_kratky_masked, n_bootstrap=50
                    )
                    if q_peak_error is None:
                        q_peak_error = constant_error

                else:  # 'constant'
                    q_peak_error = constant_error

                has_peak = True

            else:
                # NO PEAKS found by find_peaks - Treat as NO PEAK
                has_peak = False
                q_peak = q_max  # Use upper bound as "best guess"
                I_peak = np.max(y_kratky_masked)  # Still record max intensity
                q_peak_error = None  # Will be assigned large error below

        else:
            # No data in mask region
            has_peak = False
            q_peak = q_max  # Use upper bound
            I_peak = 0
            q_peak_error = None  # Will be assigned large error below

        # Handle results based on peak detection
        if has_peak:
            results['q_peak'] = {
                'value': q_peak,
                'error': q_peak_error
            }
            results['I_peak_kratky'] = I_peak
            results['peak_detected'] = True
            peak_label = r'$q_{{peak}} = {:.4f} \pm {:.4f} \, \mathrm{{\AA}}^{{-1}}$'.format(
                q_peak, q_peak_error
            )
        else:
            # NO PEAK FOUND - MUCH LARGER uncertainty for gpCAM
            # Error is much larger than typical peak errors
            full_range = (q_max - q_min)
            baseline_error = constant_error * 3  # What weak peak would have
            large_error = max(
                baseline_error * no_peak_error_multiplier,
                full_range * 1.5  # 1.5x the full range
            )

            results['q_peak'] = {
                'value': q_max,  # Report upper bound
                'error': large_error
            }
            results['q_upper_bound'] = q_max
            results['I_peak_kratky'] = I_peak if I_peak is not None else 0
            results['peak_detected'] = False
            peak_label = r'No peak. $q < {:.4f} \, (\Delta q = {:.3f})$'.format(
                q_max, large_error
            )

        # Create Kratky plot - raw data
        kratky_line_raw = DataLine(x=x, y=y_kratky_raw,
                                   plot_args={'linestyle': '-', 'color': 'gray',
                                              'marker': 'o', 'linewidth': 1.0, 'alpha': 0.5,
                                              'label': 'Raw data'})

        # Smoothed data (dashed line)
        if smooth:
            kratky_line_smooth = DataLine(x=x, y=y_kratky,
                                          plot_args={'linestyle': '--', 'color': 'k',
                                                     'marker': None, 'linewidth': 2.0,
                                                     'label': 'Smoothed'})
            plot_lines = [kratky_line_raw, kratky_line_smooth]
        else:
            plot_lines = [kratky_line_raw]

        # Mark the peak (only if detected)
        if has_peak:
            peak_line = DataLine(x=[q_peak], y=[I_peak],
                                 plot_args={'marker': '*', 'color': 'r',
                                            'markersize': 20, 'linestyle': 'None',
                                            'label': 'Peak'})
            plot_lines.append(peak_line)

            # ADD ERROR BARS
            error_line = DataLine(
                x=[q_peak - q_peak_error, q_peak + q_peak_error],
                y=[I_peak, I_peak],
                plot_args={'linestyle': '-', 'color': 'r',
                           'linewidth': 3, 'alpha': 0.5}
            )
            plot_lines.append(error_line)

        lines = DataLines(plot_lines)

        # Add peak annotation
        class DataLines_kratky(DataLines):
            def _plot_extra(self, **plot_args):
                xi, xf, yi, yf = self.ax.axis()

                # Show peak search range if specified
                if peak_search_range[0] is not None or peak_search_range[1] is not None:
                    # Shade the search region
                    self.ax.axvspan(q_min, q_max, alpha=0.1, color='blue', zorder=0)

                self.ax.text(0.95, 0.95, peak_label, size=16, color='b',
                             transform=self.ax.transAxes,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                # Set label
                self.ax.set_xlabel(r'$q \, (\mathrm{{\AA}}^{{-1}})$', size=32)
                self.ax.set_ylabel(r'$q^{{{:.0f}}}I(q)$'.format(qalpha), size=32)

                # Add legend
                self.ax.legend(loc='upper left', fontsize=14)

        lines_plot = DataLines_kratky(plot_lines)

        outfile = self.get_outfile(data.name + '-kratky', output_dir, ext='.png')
        lines_plot.plot(save=outfile, **run_args)

        # Save data (both raw and smoothed)
        outfile = self.get_outfile(data.name + '-kratky', output_dir, ext='.dat')
        kratky_line_raw.save_data(outfile)

        if smooth:
            outfile_smooth = self.get_outfile(data.name + '-kratky-smooth', output_dir, ext='.dat')
            kratky_line_smooth.save_data(outfile_smooth)

        if has_peak:
            print(
                f"Peak: q = {results['q_peak']['value']:.4f} ± {results['q_peak']['error']:.4f} Å⁻¹ (range: [{q_min:.4f}, {q_max:.4f}])")
        else:
            print(
                f"No peak. Upper bound: q < {results['q_peak']['value']:.4f} Å⁻¹, error = {results['q_peak']['error']:.4f}")

        return results

    def _bootstrap_uncertainty(self, x, y, n_bootstrap=50):
        """Estimate peak uncertainty via bootstrapping"""
        peak_positions = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y), size=len(y), replace=True)
            y_boot = y[indices]

            peaks_boot, _ = find_peaks(y_boot)
            if len(peaks_boot) > 0:
                peak_idx = peaks_boot[np.argmax(y_boot[peaks_boot])]
                peak_positions.append(x[peak_idx])

        if len(peak_positions) > 10:
            return np.mean(peak_positions), np.std(peak_positions)
        else:
            return None, None


# ==========================


def autonomous_result(xml_file, clean=True, verbosity=3):
    # DEPRECATED: Use SQL database instead:
    # new_result = ResultsDB().extract_single(infile)

    time.sleep(0.5)  # Kludge to avoid corrupting XML file?

    extractions = [
        ['metadata_extract',
         [
             'x_position',
             'y_position',
             # 'sequence_ID',
         ]
         ],
        # ['circular_average_q2I_fit',
        #     [
        #     'fit_peaks_prefactor1',
        #     'fit_peaks_x_center1',
        #     'fit_peaks_sigma1',
        #     'fit_peaks_chi_squared',
        #     'fit_peaks_prefactor1_error',
        #     'fit_peaks_x_center1_error',
        #     'fit_peaks_sigma1_error',
        #     ]
        # ],
        # ['linecut_angle_fit',
        #     [
        #     'fit_eta_eta',
        #     'orientation_factor',
        #     'orientation_angle',
        #     'fit_eta_span_prefactor',
        #     ]
        # ],
        ['circular_average_q2I_fit_sorted',
         [
             'fit_peaks_prefactor1',
             'fit_peaks_x_center1',
             'fit_peaks_sigma1',
             'fit_peaks_chi_squared',
             'fit_peaks_prefactor1_error',
             'fit_peaks_x_center1_error',
             'fit_peaks_sigma1_error',

             'fit_peaks_prefactor2',
             'fit_peaks_x_center2',
             'fit_peaks_sigma2',
             'fit_peaks_chi_squared2',
             'fit_peaks_prefactor2_error',
             'fit_peaks_x_center2_error',
             'fit_peaks_sigma2_error',
         ]
         ],
    ]

    results_dict = Results().extract_dict([xml_file], extractions, verbosity=verbosity)[0]

    if clean:
        for key, result in results_dict.items():
            if key != 'filename':
                results_dict[key] = np.nan_to_num(float(result))

    return results_dict


# Experimental parameters
########################################


## TOCHANGE
# calibration = Calibration(wavelength_A=0.7294) # 17 keV
# #calibration = Calibration(wavelength_A=0.9184) # 13.5 keV
# calibration.set_image_size(981, height=1043) # Pilatus800k
# calibration.set_pixel_size(pixel_size_um=172.0)

# # calibration.set_beam_position(619.5, 1043-254)
# # calibration.set_distance(0.225)
# calibration.set_beam_position(255, 1043-272)
# calibration.set_distance(0.26)

## CARLY To change
#################### WAXS CALIBRATION
import yaml

with open('caliMS.yaml', 'r') as f:  # Carly
    cfg = yaml.safe_load(f)

from SciAnalysis.XSAnalysis.DataRQconv import *

calibration = Calibration(wavelength_A=cfg["wavelength_A"])
calibration.set_image_size(cfg["image_size"][0], height=cfg["image_size"][1])
calibration.set_pixel_size(pixel_size_um=cfg["pixel_size_um"])
calibration.set_beam_position(cfg["beam_position"][0], cfg["beam_position"][1])
calibration.set_distance(cfg["distance"])

mask_dir = SciAnalysis_PATH + '/SciAnalysis/XSAnalysis/masks/'
mask = Mask(mask_dir + 'Dectris/Pilatus800k2_gaps-mask.png')
# mask.load('./mask3.png')
mask.load('./combined_mask_MAXS_PTA2.png')  # Carly

# Files to analyze
########################################
source_dir = '../maxs/raw/'  # Carly
# output_dir = '../waxs/analysis/'
waxs_output_dir = '../maxs/analysis/'  # Carly

pattern = '*'

infiles = glob.glob(os.path.join(source_dir, pattern + '.tiff'))
infiles.sort()

# Analysis to perform
########################################

load_args = {'calibration': calibration,
             'mask': mask,
             # 'background' : source_dir+'empty*saxs.tiff',
             # 'transmission_int': '../../data/Transmission_output.csv', # Can also specify an float value.
             }
run_args = {'verbosity': 3,
            # 'save_results' : ['xml', 'plots', 'txt', 'hdf5', 'sql'],
            'rcParams': {'axes.labelsize': 20,
                         'xtick.labelsize': 13,
                         'ytick.labelsize': 13,
                         'xtick.major.pad': 11,
                         'ytick.major.pad': 11,
                         },
            }

process = Protocols.ProcessorXS(load_args=load_args, run_args=run_args)
# process.connect_databroker('cms') # Access databroker metadata


patterns = [
    ['Ti', r'.+_Ti(\d+\.?\d+)_.+'],
    ['Tm', r'.+_Tm(\d+\.?\d+)_.+'],
    ['Tf', r'.+_Tf(\d+\.?\d+)_.+'],
    ['Tc', r'.+_Tc(\d+\.?\d+)_.+'],
    ['run', r'.+_run(\d+)_.+'],
    ['clock', r'.+_Tc\d+\.?\d+_(\d+\.\d)s.+'],
    ['theta', r'.+_th(\d+\.\d+)_.+'],
    ['x_position', r'.+_x(-?\d+\.\d+)_.+'],
    ['y_position', r'.+_yy(-?\d+\.\d+)_.+'],
    # ['anneal_time', r'.+_anneal(\d+)_.+'] ,
    # ['cost', r'.+_Cost(\d+\.?\d+)_.+'] ,
    ['annealing_temperature', r'.+_T(\d+\.\d\d\d)C_.+'],
    # ['annealing_time', r'.+_(\d+\.\d)s_T.+'] ,
    # ['annealing_temperature', r'.+_localT(\d+\.\d)_.+'] ,
    # ['annealing_time', r'.+_clock(\d+\.\d\d)_.+'] ,
    # ['o_position', r'.+_opos(\d+\.\d+)_.+'] ,
    # ['l_position', r'.+_lpos(\d+\.\d+)_.+'] ,
    ['exposure_time', r'.+_(\d+\.\d+)s_\d+_saxs.+'],
    ['sequence_ID', r'.+_(\d+).+'],
]

# # qp1, dqp1 = 2.65, 0.1
# #for sample: test_run

# qp1, dqp1 = 2.55, 0.2
# #for sample: b48-01_NbAlCu
# # qp1, dqp1 = 2.62, 0.5
# #for sample: b53-01_CuNbAlCu
# qp1, dqp1 = 2.99, 0.1

# #for sample: b53-01_CuNbAlCu, test another peak
# #qp1, dqp1 = 3.46, 0.2

# #for sample: b50-01_NbAlSc
# #qp1, dqp1 = 2.711, 0.1 #AB phase: Nb0.8Al0.2 04-002-9614
# #qp1, dqp1 = 2.623, 0.1 #BC phases: ScAl 04-001-2001
# qp1, dqp1 = 2.286, 0.1 #beta Sc
# #qp1, dqp1 = 2.46, 0.1 #alpha Sc, determined after fitting the pristine data

# qp1, dqp1 = 2.85, 0.15 #Mo peak

# qp1, dqp1 = 3.05, 0.25 #VMn peak
# qp1, dqp1 = 3.55, 0.15 #NiCr/Ni/NiCu peak

# #Mar 2025
# qp1, dqp1 = 2.65, 0.15 #Ti peak

# qp1, dqp1 = 3.00, 0.15 #Cu peak


# #Jun. 2025.
# qp1, dqp1 = 3.05, 0.25 #Cu peak

# Dec. 2025
# qp1, dqp1 = 3.5, 0.15 # Ni (2 0 0) peak

# Mar. 2026
qp1, dqp1 = 3.52, 0.06  # Ni3Al (0 0 2) peak

protocols = [
    # Protocols.HDF5(save_results=['hdf5'])
    # Protocols.calibration_check(show=False, AgBH=True, q0=0.010, num_rings=4, ztrim=[0.05, 0.05], ) ,
    # Protocols.circular_average(ylog=True, plot_range=[0, 0.12, None, None], label_filename=True) ,
    Protocols.thumbnails(crop=None, resize=1.0, blur=None, cmap=cmap_vge, ztrim=[0.01, 0.001]),

    # Protocols.circular_average_q2I_fit(show=False, q0=0.0140, qn_power=2.5, sigma=0.0008, plot_range=[0, 0.06, 0, None], fit_range=[0.008, 0.022]) ,
    # Protocols.circular_average_q2I_fit(qn_power=3.5, trim_range=[0.005, 0.03], fit_range=[0.007, 0.019], q0=0.0120, sigma=0.0008) ,
    # Protocols.circular_average_q2I_fit(qn_power=3.0, trim_range=[0.005, 0.035], fit_range=[0.008, 0.03], q0=0.0180, sigma=0.001) ,

    # linecut_qz_fit(qr=0.0185, dq=0.004, show_region=False, label_filename=True, trim_range=[0, 0.06], fit_range=[0.036, 0.055], plot_range=[0, 0.06, 0, None], q0=0.043, sigma=0.0022, critical_angle_substrate=0.132, critical_angle_film=0.094, ),
    # Protocols.roi(name='roi', qx=0.0227, dqx=0.00225, qz=0.0310, dqz=0.0058, show_region=False) ,
    # roi_ratio(name='roi_ratio', qx=0.0224, dqx=0.00219, qz=0.0310, dqz=0.0058, show_region='save') ,

    # NB for Karen: You can change which protocol runs here
    # Protocols.circular_average_sum('sum_p1', ylog=False, plot_range=[1.5, 4, 0, None], dezing=True, sum_range=[qp1-dqp1/2, qp1+dqp1/2]) ,
    # Protocols.circular_average_q2I_fit(plot_range=[3,3, 3.7, 0, None], q0=qp1, fit_range=[qp1-dqp1, qp1+dqp1]),
    ##circular_average_q2I_fit_custom('circular_average_q2I_fit', plot_range=[1, 4.5, 0, None], q0 = qp1, fit_range=[qp1-dqp1, qp1+dqp1], num_curves=1),
    #circular_average_q2I_fit_sorted('circular_average_q2I_fit_sorted', plot_range=[3.3, 3.7, 0, None], q0=qp1,
    #                                fit_range=[qp1 - dqp1, qp1 + dqp1], num_curves=1),
    circular_average_q2I_fit_FWHM('circular_average_q2I_fit_FWHM', plot_range=[2.8, 3.3, 0, None], q0=3.07, fit_range=[3.07-0.08, 3.07+0.08], num_curves=1),
    # Protocols.databroker_extract(constraints={'measure_type':'measure'}, timestamp=True, sectino='start'),
    Protocols.metadata_extract(patterns=patterns),
]

# Experimental parameters
########################################


## TOCHANGE
# calibration_saxs = Calibration(wavelength_A=0.7294) # 17 keV
# #calibration = Calibration(wavelength_A=0.9184) # 13.5 keV
# calibration_saxs.set_image_size(1475, height=1679) # Pilatus2M
# calibration_saxs.set_pixel_size(pixel_size_um=172.0)

# # calibration_saxs.set_beam_position(751, 1679-547) # SAXSx -60, SAXSy -73 #1125488
# # calibration_saxs.set_distance(6)
# calibration_saxs.set_beam_position(741, 1679-550) # SAXSx -60, SAXSy -73 #1125488
# calibration_saxs.set_distance(5.03)
#########################################################
##########CARLY TO CHANGE
import yaml

with open('caliXS.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

calibration_saxs = Calibration(wavelength_A=cfg["wavelength_A"])
calibration_saxs.set_image_size(cfg["image_size"][0], height=cfg["image_size"][1])
calibration_saxs.set_pixel_size(pixel_size_um=cfg["pixel_size_um"])
calibration_saxs.set_beam_position(cfg["beam_position"][0], cfg["beam_position"][1])
calibration_saxs.set_distance(cfg["distance"])

mask_dir_saxs = SciAnalysis_PATH + '/SciAnalysis/XSAnalysis/masks/'
mask_saxs = Mask(mask_dir_saxs + 'Dectris/Pilatus2M_gaps-mask.png')
# mask_saxs.load('./Pilatus2M_current-mask.png')
mask_saxs.load('./combined_mask_SAXS2.png')

source_dir = '../saxs/raw/'
# output_dir = '../saxs/analysis/'
saxs_output_dir = '../saxs/analysis/'

pattern = '*'

infiles = glob.glob(os.path.join(source_dir, pattern + '.tiff'))
infiles.sort()
# Analysis to perform
########################################

load_args_saxs = {'calibration': calibration_saxs,
                  'mask': mask_saxs,
                  # 'background' : source_dir+'empty*saxs.tiff',
                  # 'transmission_int': '../../data/Transmission_output.csv', # Can also specify an float value.
                  }
run_args_saxs = {'verbosity': 3,
                 # 'save_results' : ['xml', 'plots', 'txt', 'hdf5'],
                 'rcParams': {'axes.labelsize': 20,
                              'xtick.labelsize': 13,
                              'ytick.labelsize': 13,
                              'xtick.major.pad': 11,
                              'ytick.major.pad': 11,
                              },
                 }

process_saxs = Protocols.ProcessorXS(load_args=load_args_saxs, run_args=run_args_saxs)

protocols_saxs = [
    Protocols.qr_image(blur=None, colorbar=True, cmap=cmap_vge, ztrim=[0.01, 0.001]),
    # linecut_qr_fit_custom(name='linecut_qr_fit_custom', qz=0.058, dq=0.005, trim_range=[0.003, 0.02]) ,
    Protocols.linecut_qz(name='linecut_qz', ylog=True, qr=0.02, dq=0.005, show_region=False,
                         plot_range=[0.04, 0.1, None, None]),
    # Default: peak_width method
    linecut_qr_kratky(
        qz=0.075,
        dq=0.005,
        trim_range=[0.003, 0.05],  # Only use data in this q-range
        peak_search_range=[0.003, 0.015],
        peak_threshold=0.05,
        qalpha=2.0,
        smooth=True  # Disable smoothing
    ),
    # Protocols.databroker_extract(constraints={'measure_type':'measure'}, timestamp=True, sectino='start'),
    Protocols.metadata_extract(patterns=patterns),
]


# Helpers
########################################
# DEPRECATED in favor of tools.val_stats
def print_d(d, i=4):
    '''Simple helper to print a dictionary.'''
    for k, v in d.items():
        if isinstance(v, dict):
            print('{}{} : <dict>'.format(' ' * i, k))
            print_d(v, i=i + 4)
        elif isinstance(v, (np.ndarray)):
            print('{}{} : Ar{}: {}'.format(' ' * i, k, v.shape, v))
        elif isinstance(v, (list, tuple)):
            print('{}{} : L{}: {}'.format(' ' * i, k, len(v), v))
        else:
            print('{}{} : {}'.format(' ' * i, k, v))


def print_results(results):
    '''Simple helper to print out a list of dictionaries.'''
    for i, result in enumerate(results):
        print(i)
        print_d(result)


def print_n(d):
    '''Simple helper to print nested arrays/dicts'''
    if isinstance(d, (list, tuple, np.ndarray)):
        print_results(d)
    elif isinstance(d, dict):
        print_d(d)
    else:
        print(d)


def val_stats(values, name='z'):
    span = np.max(values) - np.min(values)
    print("  {} = {:.2g} ± {:.2g} (span {:.2g}, from {:.3g} to {:.3g})".format(name, np.average(values), np.std(values),
                                                                               span, np.min(values), np.max(values)))


# Inspect .npy files
########################################
# This code can be pasted into a new ipython shell
# to help inspect the .npy files that are passed
# in the AE loop.
'''

import numpy as np

def print_d(d, i=4):
    for k, v in d.items():
        if isinstance(v,dict):
            print('{}{} : <dict>'.format(' '*i,k))
            print_d(v, i=i+4)
        else:
            print('{}{} : {}'.format(' '*i,k,v))

def print_results(results):
    for i, result in enumerate(results):
        print(i)
        print_d(result)


results = np.load('analyze-received.npy', allow_pickle=True); print_results(results)
results = np.load('analyze-sent.npy', allow_pickle=True); print_results(results)

results = np.load('../../measure-received.npy', allow_pickle=True); print_results(results)
results = np.load('../../measure-sent.npy', allow_pickle=True); print_results(results)

results = np.load('../../gpcamv4and5/scripts/decision-received.npy', allow_pickle=True); print_results(results)
results = np.load('../../gpcamv4and5/scripts/decision-sent.npy', allow_pickle=True); print_results(results)

print_results(results)

'''


def get_analysis_result_old(infile, protocol='circular_average_q2I_fit', verbosity=3):
    # Get the result of this analysis from the SQL database
    # results_dict = ResultsDB().extract_single(infile, verbosity=verbosity)

    results = Results().extract(infile, 'circular_average_q2I_fit', 'fit_peaks_x_center1')
    print(results)

    # if verbosity>=6:
    # print(results_dict[protocol])

    # TOCHANGE
    if False:
        value = results_dict['linecut_qz_fit']['fit_peaks_prefactor1']
        try:
            error = results_dict['linecut_qz_fit']['fit_peaks_prefactor1_error']
        except KeyError:
            error = max(value * 0.5, 0.1)

    if False:
        # ROI
        # value = results_dict['roi']['stats_total'] # We accidentally used this for some of the beamtime (inconsistent with prelim data given to gpCAM)
        # value = (results_dict[protocol]['stats_total'])/404357.0 # Should be this (to be consistent with aePrep)
        error = 0.1 * value

    # NB for Karen: You can change which protocol runs here
    if False:
        value = results_dict['sum_p1']['values_sum']
        error = 0.05 * value

    # NB for Karen: You can change which protocol runs here
    # if True:
    #     # value = results_dict['circular_average_q2I_fit']['fit_peaks_prefactor1']
    #     #for NiCr/Ni/NiCu
    #     value = results_dict['circular_average_q2I_fit']['fit_peaks_x_center2']
    #     try:
    #         # error = results_dict['circular_average_q2I_fit']['fit_peaks_prefactor1_error']
    #         #for NiCr/Ni/NiCu
    #         error = results_dict['circular_average_q2I_fit']['fit_peaks_x_center2_error']
    #     except KeyError:
    #         error = value*0.05

    if True:
        value = results['circular_average_q2I_fit']['fit_peaks_x_center1']['value']
        error = results['circular_average_q2I_fit']['fit_peaks_x_center1']['error']

    variance = np.square(error)

    return value, variance, results

    # pass


def get_analysis_result(infile, protocol=FITTING_PLAN, verbosity=3):
    """
    Get the result of this analysis from the XML file
    Returns: value, variance, results_dict
    """

    # Extract results from XML file
    extractions = [
        ['metadata_extract',
         [
             'x_position',
             'y_position',
             'sequence_ID',
         ]
         ],
        [f'{FITTING_PLAN}',
         [
             'fit_peaks_prefactor1',  # fitting peak height
             'fit_peaks_x_center1',  # q position
             'fit_peaks_sigma1',  # gaus fitting factor (width)
             'fit_peaks_fwhm1',
             'fit_peaks_fwhm1_error',
             'fit_peaks_chi_squared',
             'fit_peaks_prefactor1_error',
             'fit_peaks_x_center1_error',
             'fit_peaks_sigma1_error',
             'fit_peaks_x_center2',
             'fit_peaks_x_center2_error',
             'fit_peaks_prefactor2',
             'fit_peaks_prefactor2_error',
         ]
         ],
    ]

    # Get XML filename (replace .tiff with .xml)
    # xml_file = infile.replace('.tiff', '.xml').replace('../waxs/raw/', '/analysis_swaxs/results/')
    xml_file = infile.replace('.tiff', '.xml').replace('./maxs/raw/', '../maxs/analysis/results/')
    ## Mock Experiment: Read analysis value from .csv for fixed composition
    if True:  # was: if False — enabled for the time/position mock
        # Beamline filename example: ..._x12.183_th0.200_1260.9s_2048356_000000_waxs.xml
        # Parse TIME and X-POSITION from the filename (position represents temperature)
        import re as _re
        xmatch = _re.search(r'_x(-?\d+\.\d+)_', infile)
        tmatch = _re.search(r'_(\d+\.\d+)s_', infile)
        x_pos = float(xmatch.group(1)) if xmatch else 0.0
        t_val = float(tmatch.group(1)) if tmatch else 0.0

        # Load scattered mock data: columns time, x_position, fwhm, fwhm_err
        mockfile = '/path/to/CrCuNiCr_insitu_NEW.csv'  # <-- set the real beamline path
        df = pd.read_csv(mockfile)

        # drop failed-fit rows so they don't distort the interpolation
        df = df[(df['fwhm'] > 0.05) & (df['fwhm'] < 0.13)]

        # Build 2D interpolators over (x_position, time)
        from scipy.interpolate import LinearNDInterpolator
        pts = df[['x_position', 'time']].values
        interp_fwhm = LinearNDInterpolator(pts, df['fwhm'].values)
        interp_err = LinearNDInterpolator(pts, df['fwhm_err'].values)

        # interpolate FWHM and its error at the requested (position, time)
        value = float(interp_fwhm(x_pos, t_val))
        error = float(interp_err(x_pos, t_val))

        # random fallback only when asked outside the measured region (NaN)
        if np.isnan(value):
            value = np.random.uniform(0.09, 0.13)
        if np.isnan(error) or error <= 0:
            error = value * 0.05
    # # Mock Experiment: Read analysis value from .csv for combinatorial
    # if False:
    #     # Example file name HW_C01_40_Tc_x0.000_th0.300_10.00s_2248944_000000_waxs.xml
    #     parts = infile.split('_')
    #     composition = int(parts[-8])
    #     x_pos = float(parts[-6].replace('x', ''))
    #
    #     mockfile = '/nsls2/data3/cms/shared/config/bluesky/profile_collection/users/2026-1/KChen-Wiegart/2026C1/mockData/HW_CrCuNiCr_fit_peaks.csv'
    #     df = pd.read_csv(mockfile)
    #
    #     df.columns = [40, 45, 50, 55, 60]
    #     from scipy.interpolate import interp1d
    #     x = np.arange(df.shape[1])  # sample index
    #     y = np.arange(df.shape[0])  # scan index
    #     y = y * 0.4
    #
    #     # Fine y grid for the lookup table
    #     y_fine = np.linspace(y.min(), y.max(), 1000)
    #
    #     # Build interpolated lookup table: rows = fine y positions, cols = samples
    #     lut = pd.DataFrame(index=y_fine, columns=df.columns, dtype=float)
    #
    #     for col in df.columns:
    #         f = interp1d(y, df[col].values, kind='cubic')
    #         lut[col] = f(y_fine)
    #
    #         interp_per_sample = {
    #             col: interp1d(y, df[col].values, kind='cubic')
    #             for col in df.columns
    #         }
    #
    #     # composition = 'HW-C02-CrCuNiCr_th0.30'
    #     # x_pos = 30
    #     value = interp_per_sample[composition](x_pos)
    #     error = value * 0.05

    try:
        # Extract data from XML
        results_dict = Results().extract_dict([xml_file], extractions, verbosity=verbosity)[0]

        if verbosity >= 6:
            print(f"Extracted results from {xml_file}:")
            print(results_dict)

        # FIX: Access the nested structure correctly
        # The XML shows results are nested under protocol names
        if protocol in results_dict:
            protocol_results = results_dict[protocol]
        else:
            protocol_results = results_dict

        # Option 1: Use x_center1 (peak position)
        if f'{EXTRACTION}' in protocol_results:
            value = np.float64(protocol_results[EXTRACTION])
            try:
                error = np.float64(protocol_results[f'{EXTRACTION}_error'])
                if error is None or np.isnan(error) or error <= 0:
                    error = value * 0.05  # 5% default error
            except (KeyError, TypeError, ValueError):
                print('Default error')
                error = value * 0.05

        # Alternative: Use direct access to flat dictionary
        elif f'{FITTING_PLAN}__{EXTRACTION}' in results_dict:
            value = np.float64(results_dict[f'{FITTING_PLAN}__{EXTRACTION}'])
            try:
                error = np.float64(results_dict[f'{FITTING_PLAN}__{EXTRACTION}_error'])
                if error is None or np.isnan(error) or error <= 0:
                    error = value * 0.05
            except (KeyError, TypeError, ValueError):
                print('Default error')
                error = value * 0.05

        else:
            # Fallback if no expected parameters found
            if verbosity >= 1:
                print(f"Available keys in results_dict: {list(results_dict.keys())}")
                if protocol in results_dict:
                    print(f"Available keys in {protocol}: {list(results_dict[protocol].keys())}")
            value = 3.5  # Use your expected peak position as default
            error = 0.1

        # # Ensure positive values
        # value = max(value, 1e-8)
        # error = max(error, 1e-8)

        variance = np.float64(np.square(error))

        if verbosity >= 3:
            print(f"Analysis result: value = {value:.4f}, error = {error:.4f}, variance = {variance:.6f}")

        return value, variance, results_dict

    except Exception as e:
        if verbosity >= 1:
            print(f"Error reading XML file {xml_file}: {e}")

        # Return default values if XML reading fails
        return 3.5, 0.01, {}  # Use expected peak position as default


def get_analysis_result_saxs_old(infile, protocol='sum_p1', verbosity=3):
    # Get the result of this analysis from the SQL database
    results_dict = ResultsDB().extract_single(infile, verbosity=verbosity)

    if verbosity >= 6:
        print(results_dict[protocol])

    # TOCHANGE
    # NB for Karen: You can change which protocol runs here
    if True:
        value = results_dict['linecut_qr_fit_custom']['fit_Guinier_Rg']
        try:
            error = results_dict['linecut_qr_fit_custom']['fit_Guinier_Rg_error']
        except KeyError:
            error = value * 0.05

    variance = np.square(error)

    return value, variance, results_dict


def get_analysis_result_saxs(infile, protocol='linecut_qr_kratky', verbosity=3):
    # xml_file = infile.replace('.tiff', '.xml').replace('/saxs/raw/', '/analysis_swaxs/results/')
    xml_file = infile.replace('.tiff', '.xml').replace('./saxs/raw/', '../saxs/analysis/results/')

    try:
        # Use ResultsXML extractor (same as Result.py does internally)
        extractor = ResultsXML()
        result_names, results = extractor.extract_results_from_xml(xml_file, protocol, verbosity=verbosity)

        if verbosity >= 5:
            print(f"Extracted result_names: {result_names}")
            print(f"Extracted results: {results}")

        # Build dict for easier access (follows Result.py pattern)
        results_dict = dict(zip(result_names, results))

        if verbosity >= 3:
            print(f"Results dict: {results_dict}")

        # Look for q_peak in the extracted results
        if 'q_peak' in results_dict:
            value = float(results_dict['q_peak'])
            error_key = 'q_peak_error'
            error = float(results_dict[error_key]) if error_key in results_dict else value * 0.05

        else:
            # Fallback if q_peak not found
            if verbosity >= 1:
                print(f"Warning: q_peak not found in {protocol}")
                print(f"Available results: {result_names}")
            value = 0.015
            error = 0.005

        # Ensure positive values
        value = max(value, 1e-6)
        error = max(error, 1e-6)
        variance = np.square(error)

        if verbosity >= 3:
            print(f"SAXS result: q_peak = {value:.6f} ± {error:.6f}, variance = {variance:.8f}")

        return value, variance, results_dict

    except Exception as e:
        if verbosity >= 1:
            print(f"Error extracting SAXS results from {xml_file}: {e}")
        return 0.015, 0.000025, {}


def determine_infile(filename, source_dir=Abs_Path + '/waxs/raw/', suffix='_000000_waxs.tiff', filename_re=None,
                     verbosity=3):
    #  Usage:
    # filename_re = re.compile('.+_x(-?\d+\.\d+)_y(-?\d+\.\d+)_.+_(\d+)_saxs.+') # TOCHANGE
    # determine_infile(result['filename'], source_dir=source_dir, filename_re=filename_re, verbosity=verbosity, suffix='_saxs.tiff') # TOCHANGE

    infile = os.path.join(source_dir, filename + suffix)
    # print(f'Initial infile: {infile}')

    while not os.path.exists(infile):
        print(f'File not found: {infile}')
        time.sleep(10)

    # infile = '{}{}'.format(source_dir, filename)
    print(f'File found ! infile: {infile}')

    # Code to handle bug where saved filename doesn't exactly match what is specified in metadata:
    if not os.path.exists(infile):
        if verbosity >= 1:
            print('Specified infile is missing. We will attempt to locate the right file based on sequence_ID.')
        if verbosity >= 5:
            print('  infile: {}'.format(infile))

        if filename_re is None:
            if verbosity >= 1:
                print("    No RE provided. Aborting.")
            return None

        else:
            m = filename_re.match(infile)
            if m:
                # print(m.groups())
                sID = int(m.groups()[-1])
                if verbosity >= 2:
                    print('    sequence ID: {:d}'.format(sID))
                mfiles = glob.glob('{}*_{:d}_000000_waxs.tiff'.format(source_dir, sID))  # TOCHANGE

                print(f'{source_dir} , {sID} , {mfiles}')

                if len(mfiles) < 1:
                    if verbosity >= 1:
                        print('    No file matches sequence ID {}.'.format(sID))
                elif len(mfiles) == 1:
                    infile = mfiles[0]
                    if verbosity >= 1:
                        print('    Using filename: {}'.format(infile))
                else:
                    if verbosity >= 1:
                        print('    {} files match sequence ID {}'.format(len(mfiles), sID))
                        print('    Aborting.')
                    return None
            else:
                if verbosity >= 1:
                    print("    RE did not match. Aborting.")
                return None

    return infile


def determine_infile_saxs(filename, source_dir=Abs_Path + '/saxs/raw/', suffix='_000000_saxs.tiff', filename_re=None,
                          verbosity=3):
    #  Usage:
    # filename_re = re.compile('.+_x(-?\d+\.\d+)_y(-?\d+\.\d+)_.+_(\d+)_saxs.+') # TOCHANGE
    # determine_infile(result['filename'], source_dir=source_dir, filename_re=filename_re, verbosity=verbosity, suffix='_saxs.tiff') # TOCHANGE

    # infile = '{}{}{}'.format(source_dir, filename, suffix)

    infile = os.path.join(source_dir, filename + suffix)
    # print(f'Initial infile: {infile}')

    while not os.path.exists(infile):
        print(f'File not found: {infile}')
        time.sleep(10)

    print(f'File found ! infile: {infile}')

    # Code to handle bug where saved filename doesn't exactly match what is specified in metadata:
    if not os.path.exists(infile):
        if verbosity >= 1:
            print('Specified infile is missing. We will attempt to locate the right file based on sequence_ID.')
        if verbosity >= 5:
            print('  infile: {}'.format(infile))

        if filename_re is None:
            if verbosity >= 1:
                print("    No RE provided. Aborting.")
            return None

        else:
            m = filename_re.match(infile)
            if m:
                sID = int(m.groups()[-1])
                if verbosity >= 2:
                    print('    sequence ID: {:d}'.format(sID))
                mfiles = glob.glob('{}*_{:d}_000000_waxs.tiff'.format(source_dir, sID))  # TOCHANGE
                if len(mfiles) < 1:
                    if verbosity >= 1:
                        print('    No file matches sequence ID {}.'.format(sID))
                elif len(mfiles) == 1:
                    infile = mfiles[0]
                    if verbosity >= 1:
                        print('    Using filename: {}'.format(infile))
                else:
                    if verbosity >= 1:
                        print('    {} files match sequence ID {}'.format(len(mfiles), sID))
                        print('    Aborting.')
                    return None
            else:
                if verbosity >= 1:
                    print("    RE did not match. Aborting.")
                return None

    return infile


# Run autonomous loop
def run_autonomous_loop_tiled(protocols, clear=False, force_load=False, republish=False, verbosity=3, simulate=False):
    # This is a tiled version of the AE loop that read data from the tiled client instead of waiting for files to appear in the directory.

    from tiled.client import from_profile
    tiled_client = cat = from_profile("nsls2", username=None)["cms/migration"]

    from CustomQueue import Queue_analyze as queue
    q = queue()

    if clear:
        q.clear()
    if republish:
        q.republish()

    if verbosity >= 3:
        print('\n\n\n')
        print('=============================')
        print('==  Autonomous Experiment  ==')
        print('=============================')
        print('\n')

    while True:  # Loop forever

        results = []
        results.append(q.get(force_load=force_load))  # Get analysis command
        print(results)
        print('------------------')
        # results = q.get(force_load=force_load) # Get analysis command
        force_load = False  # Only force a reload on the 1st iteration

        results = results[0]  # Carly ; results[0] is list and results[0][0] shows dict in list

        num_to_analyze = int(sum(1.0 for result in results if result['analyzed'] is False))

        if verbosity >= 3:
            print('Analysis requested for {} results (total {} results)'.format(num_to_analyze, len(results)))
        if verbosity >= 10:
            tools.print_results(results)

        ianalyze = 0
        for i, result in enumerate(results):

            if 'analyzed' in result and result['analyzed'] is False and 'filename' in result:
                # if 'analyzed' in result and result['analyzed'] is False: # TOCHANGE: For simulation
                ianalyze += 1

                infile_uid = result['uid']
                time.sleep(45)
                data = tiled_client[infile_uid]
                print(f"Received analysis command for UID: {data.start['uid']}, filename: {data.start['filename']}")

                saxs_data = None
                maxs_data = None  # Carly

                try:
                    saxs_data = data.primary["pilatus2m-1_image"]
                    print(f"saxs_data_size is {saxs_data.shape}")
                except KeyError:
                    print(f"no saxs data found for result {i}")

                try:
                    maxs_data = data.primary["pilatus800k-2_image"]  # Carly
                    print(f"maxs_data_size is {maxs_data.shape}")
                except KeyError:
                    print(f"no waxs data found for result {i}")

                while len(saxs_data.shape) == 4 or len(maxs_data.shape) == 4:
                    time.sleep(5)

                    saxs_data = data.primary["pilatus2m-1_image"]
                    print(f"saxs_data_size is {saxs_data.shape}")

                    maxs_data = data.primary["pilatus800k-2_image"]  # Carly
                    print(f"maxs_data_size is {maxs_data.shape}")

                # Save saxs_data and maxs_data to temporary tiff files for SciAnalysis processor
                infile = None
                infile_saxs = None

                if maxs_data is not None:
                    try:
                        import numpy as np
                        # Convert to uint16 for TIFF format if needed
                        # waxs_array = np.asarray(maxs_data[0]) if len(maxs_data.shape) > 2 else np.asarray(maxs_data)
                        # waxs_array = waxs_array.astype(np.uint16)
                        waxs_array = np.squeeze(maxs_data)
                        print(f'this is squeeze maxs_data {waxs_array.shape}')

                        infile = './maxs/raw/' + data.start['filename'] + '_000000_maxs.tiff'
                        img_waxs = Image.fromarray(waxs_array)
                        img_waxs.save(infile)
                        print(f"Saved WAXS data to file: {infile}")

                        # Create temporary file for WAXS data
                        # with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp_waxs:
                        #     infile = tmp_waxs.name
                        #     img_waxs = Image.fromarray(waxs_array)
                        #     img_waxs.save(infile)
                        #     print(f"Saved WAXS data to temporary file: {infile}")
                    except Exception as e:
                        print(f"Error saving WAXS data to temporary file: {e}")
                        infile = None

                if saxs_data is not None:
                    try:
                        import numpy as np
                        # Convert to uint16 for TIFF format if needed
                        # saxs_array = np.asarray(saxs_data[0]) if len(saxs_data.shape) > 2 else np.asarray(saxs_data)
                        # saxs_array = saxs_array.astype(np.uint16)

                        saxs_array = np.squeeze(saxs_data)

                        infile_saxs = './saxs/raw/' + data.start['filename'] + '_000000_saxs.tiff'
                        img_saxs = Image.fromarray(saxs_array)
                        img_saxs.save(infile_saxs)
                        print(f"Saved SAXS data to file: {infile_saxs}")

                        # Create temporary file for SAXS data
                        # with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp_saxs:
                        #     infile_saxs = tmp_saxs.name
                        #     img_saxs = Image.fromarray(saxs_array)
                        #     img_saxs.save(infile_saxs)
                        #     print(f"Saved SAXS data to temporary file: {infile_saxs}")
                    except Exception as e:
                        print(f"Error saving SAXS data to temporary file: {e}")
                        infile_saxs = None

                        # if verbosity>=3:
                        # print('        Analysis for result {}/{}, file: {}'.format(ianalyze, num_to_analyze, infile))

                        # if simulate:
                        #     print('Generating simulated (fake) result point.')
                        #     time.sleep(2)
                        #     value, variance, d = get_analysis_result(infile, verbosity=verbosity)
                        #     value = max(value, 1e-9)
                        #     variance = max(variance, 1e-9)

                        # value = np.random.random()*10
                        # variance = value*0.04

                # Initialize analysis variables
                value = None
                variance = None
                value_saxs = None
                variance_saxs = None

                if infile is not None and infile_saxs is not None:
                    print('Doing MAXS...')
                    process.run([infile], protocols, output_dir=waxs_output_dir, force=True)
                    time.sleep(1)  # Give some time for file to be fully written
                    value, variance, d = get_analysis_result(infile, verbosity=verbosity)
                    value = max(value, 1e-9)
                    variance = max(variance, 1e-9)
                    # value = min(value, 1e-4)
                    # variance = min(variance, 1e-4)

                    print('Doing SAXS...')
                    process_saxs.run([infile_saxs], protocols_saxs, output_dir=saxs_output_dir, force=True)
                    time.sleep(1)  # Give some time for file to be fully written
                    value_saxs, variance_saxs, d = get_analysis_result_saxs(infile_saxs, verbosity=verbosity)
                    value_saxs = max(value_saxs, 1e-9)
                    variance_saxs = max(variance_saxs, 1e-9)
                    # value_saxs = min(value_saxs, 1e-4)
                    # variance_saxs = min(variance_saxs, 1e-4)

                # Package for Tsuchinoko/gpCAM
                if verbosity >= 5:
                    print('Packaging result: {:.4g} ± {:.4g}')

                # Only mark as analyzed if we successfully obtained values
                if value is not None and variance is not None and value_saxs is not None and variance_saxs is not None:
                    result['value'] = [value, value_saxs]
                    result['variance'] = [variance, variance_saxs]
                    result['analyzed'] = True
                else:
                    print(
                        f"Warning: Analysis failed to produce valid results. value={value}, variance={variance}, value_saxs={value_saxs}, variance_saxs={variance_saxs}")
                    result['analyzed'] = False

        if verbosity >= 3:
            print('Analyzed {} results'.format(ianalyze))
        if verbosity >= 1 and ianalyze < 1:
            print('WARNING: No results were analyzed.')
        if verbosity >= 5:
            print_results(results)

        q.publish(results)

    # # Get list of all files to analyze
    # source_dir = Abs_Path+'/waxs/raw/'
    # pattern = '*_000000_waxs.tiff' # TOCHANGE
    # infiles = glob.glob(os.path.join(source_dir, pattern))
    # infiles.sort()

    # if verbosity>=3:
    #     print(f'Found {len(infiles)} files to analyze.')

    # for infile in infiles:
    #     print(f'Processing file: {infile}')
    #     process.run([infile], protocols, output_dir=output_dir, force=True)
    #     time.sleep(1) # Give some time for file to be fully written


def test_analysis(protocols, clear=False, force_load=False, republish=False, verbosity=3, simulate=False):
    result = []

    # This is a tiled version of the AE loop that read data from the tiled client instead of waiting for files to appear in the directory.
    from tiled.client import from_profile
    tiled_client = cat = from_profile("nsls2", username=None)["cms/migration"]

    from CustomQueue import Queue_analyze as queue
    q = queue()

    if clear:
        q.clear()
    if republish:
        q.republish()

    infile_uid = '37f267d4-d46e-48e0-8505-e1ebc004ff47'
    data = tiled_client[infile_uid]
    print(f"Received analysis command for UID: {data.start['uid']}, filename: {data.start['filename']}")

    saxs_data = None
    maxs_data = None
    try:
        saxs_data = data.primary["pilatus2m-1_image"]
    except KeyError:
        print(f"no saxs data found for result {i}")

    try:
        maxs_data = data.primary["pilatus800k-1_image"]
    except KeyError:
        print(f"no waxs data found for result {i}")

    # Save saxs_data and maxs_data to temporary tiff files for SciAnalysis processor
    infile = None
    infile_saxs = None

    if maxs_data is not None:
        try:
            import numpy as np
            # Convert to uint16 for TIFF format if needed
            # waxs_array = np.asarray(maxs_data[0]) if len(maxs_data.shape) > 2 else np.asarray(maxs_data)
            # waxs_array = waxs_array.astype(np.uint16)
            waxs_array = np.squeeze(maxs_data)

            infile = './waxs/raw/' + data.start['filename'] + '_000000_waxs.tiff'
            img_waxs = Image.fromarray(waxs_array)
            img_waxs.save(infile)
            print(f"Saved WAXS data to file: {infile}")

            # Create temporary file for WAXS data
            # with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp_waxs:
            #     infile = tmp_waxs.name
            #     img_waxs = Image.fromarray(waxs_array)
            #     img_waxs.save(infile)
            #     print(f"Saved WAXS data to temporary file: {infile}")
        except Exception as e:
            print(f"Error saving WAXS data to temporary file: {e}")
            infile = None

    if saxs_data is not None:
        try:
            import numpy as np
            # Convert to uint16 for TIFF format if needed
            # saxs_array = np.asarray(saxs_data[0]) if len(saxs_data.shape) > 2 else np.asarray(saxs_data)
            # saxs_array = saxs_array.astype(np.uint16)

            saxs_array = np.squeeze(saxs_data)

            infile_saxs = './saxs/raw/' + data.start['filename'] + '_000000_saxs.tiff'
            img_saxs = Image.fromarray(saxs_array)
            img_saxs.save(infile_saxs)
            print(f"Saved WAXS data to file: {infile_saxs}")

            # Create temporary file for SAXS data
            # with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp_saxs:
            #     infile_saxs = tmp_saxs.name
            #     img_saxs = Image.fromarray(saxs_array)
            #     img_saxs.save(infile_saxs)
            #     print(f"Saved SAXS data to temporary file: {infile_saxs}")
        except Exception as e:
            print(f"Error saving SAXS data to temporary file: {e}")
            infile_saxs = None

            # if verbosity>=3:
            # print('        Analysis for result {}/{}, file: {}'.format(ianalyze, num_to_analyze, infile))

            # if simulate:
            #     print('Generating simulated (fake) result point.')
            #     time.sleep(2)
            #     value, variance, d = get_analysis_result(infile, verbosity=verbosity)
            #     value = max(value, 1e-9)
            #     variance = max(variance, 1e-9)

            # value = np.random.random()*10
            # variance = value*0.04

    # Initialize analysis variables
    value = None
    variance = None
    value_saxs = None
    variance_saxs = None

    if infile is not None and infile_saxs is not None:
        print('Doing WAXS...')
        process.run([infile], protocols, output_dir=waxs_output_dir, force=True)
        time.sleep(1)  # Give some time for file to be fully written
        value, variance, d = get_analysis_result(infile, verbosity=verbosity)
        value = max(value, 1e-9)
        variance = max(variance, 1e-9)
        # value = min(value, 1e-4)
        # variance = min(variance, 1e-4)

        print('Doing SAXS...')
        process_saxs.run([infile_saxs], protocols_saxs, output_dir=saxs_output_dir, force=True)
        time.sleep(1)  # Give some time for file to be fully written
        value_saxs, variance_saxs, d = get_analysis_result_saxs(infile_saxs, verbosity=verbosity)
        value_saxs = max(value_saxs, 1e-9)
        variance_saxs = max(variance_saxs, 1e-9)
        # value_saxs = min(value_saxs, 1e-4)
        # variance_saxs = min(variance_saxs, 1e-4)

    # Package for Tsuchinoko/gpCAM
    if verbosity >= 5:
        print('Packaging result: {:.4g} ± {:.4g}')

    # Only mark as analyzed if we successfully obtained values
    if value is not None and variance is not None and value_saxs is not None and variance_saxs is not None:
        result['value'] = [value, value_saxs]
        result['variance'] = [variance, variance_saxs]
        result['analyzed'] = True
    else:
        print(
            f"Warning: Analysis failed to produce valid results. value={value}, variance={variance}, value_saxs={value_saxs}, variance_saxs={variance_saxs}")
        result['analyzed'] = False

    ianalyze = 1

    if verbosity >= 3:
        print('Analyzed {} results'.format(ianalyze))
    if verbosity >= 1 and ianalyze < 1:
        print('WARNING: No results were analyzed.')
    if verbosity >= 5:
        print_results(result)

    q.publish(result)


########################################
def run_autonomous_loop(protocols, clear=False, force_load=False, republish=False, verbosity=3, simulate=False):
    # IMPORTANT NOTE: Search for "# TOCHANGE" in the code below for
    # beamline-specific and experiment-specific assumptions that need
    # to be adjusted.

    # Connect to queue to receive the next analysis command
    # code_PATH='/nsls2/data/cms/legacy/xf11bm/data/2024_3/KChen-Wiegart6/'
    # code_PATH='../'
    # code_PATH in sys.path or sys.path.append(code_PATH)
    # print(f'Current Code PATH: {sys.path}')

    from CustomQueue import Queue_analyze as queue
    # from CustomQueue import Queue_analyzeFix as queue
    # from CustomS3 import Queue_analyze as queue
    q = queue()

    if clear:
        q.clear()
    if republish:
        q.republish()

    if verbosity >= 3:
        print('\n\n\n')
        print('=============================')
        print('==  Autonomous Experiment  ==')
        print('=============================')
        print('\n')

    # filename_re = re.compile('.+_x(-?\d+\.\d+)_y(-?\d+\.\d+)_.+_(\d+)_saxs.+')
    # infile: ../waxs/raw/HW_C04_55_Tc_x28.889_th0.300_10.00s_2246964_000000_waxs.tiff
    filename_re = re.compile(r'.*_x(-?\d+\.\d+)_th(?:-?\d+\.\d+)_([0-9]+(?:\.\d+)?)s_(\d+)_000000', re.IGNORECASE)

    # filename_re = re.compile('.+_(-?\d+\.\d+)s_x(-?\d+\.\d+)_.+_(\d+)_waxs.+') # TOCHANGE

    while True:  # Loop forever

        results = q.get(force_load=force_load)  # Get analysis command
        force_load = False  # Only force a reload on the 1st iteration

        num_to_analyze = int(sum(1.0 for result in results if result['analyzed'] is False))

        if verbosity >= 3:
            print('Analysis requested for {} results (total {} results)'.format(num_to_analyze, len(results)))
        if verbosity >= 10:
            tools.print_results(results)

        ianalyze = 0
        for i, result in enumerate(results):

            if 'analyzed' in result and result['analyzed'] is False and 'filename' in result:
                # if 'analyzed' in result and result['analyzed'] is False: # TOCHANGE: For simulation
                ianalyze += 1

                infile = determine_infile(result['filename'], source_dir=Abs_Path + '/waxs/raw/',
                                          filename_re=filename_re, verbosity=verbosity,
                                          suffix='_000000_waxs.tiff')  # TOCHANGE
                print(f'result_infile: {infile}')
                infile_saxs = determine_infile_saxs(result['filename'], source_dir=Abs_Path + '/saxs/raw/',
                                                    filename_re=filename_re, verbosity=verbosity,
                                                    suffix='_000000_saxs.tiff')  # TOCHANGE
                # infile = 'simulation_infile' # TOCHANGE: For simulation

                if verbosity >= 3:
                    print('        Analysis for result {}/{}, file: {}'.format(ianalyze, num_to_analyze, infile))

                if simulate:
                    print('Generating simulated (fake) result point.')
                    time.sleep(2)
                    value, variance, d = get_analysis_result(infile, verbosity=verbosity)
                    value = max(value, 1e-9)
                    variance = max(variance, 1e-9)

                    # value = np.random.random()*10
                    # variance = value*0.04

                else:
                    print('Doing MAXS...')
                    process.run([infile], protocols, output_dir=waxs_output_dir, force=True)
                    time.sleep(1)  # Give some time for file to be fully written
                    value, variance, d = get_analysis_result(infile, verbosity=verbosity)
                    value = max(value, 1e-9)
                    variance = max(variance, 1e-9)
                    # value = min(value, 1e-4)
                    # variance = min(variance, 1e-4)

                    print('Doing SAXS...')
                    process_saxs.run([infile_saxs], protocols_saxs, output_dir=saxs_output_dir, force=True)
                    time.sleep(1)  # Give some time for file to be fully written
                    value_saxs, variance_saxs, d = get_analysis_result_saxs(infile_saxs, verbosity=verbosity)
                    value_saxs = max(value_saxs, 1e-9)
                    variance_saxs = max(variance_saxs, 1e-9)
                    # value_saxs = min(value_saxs, 1e-4)
                    # variance_saxs = min(variance_saxs, 1e-4)

                # Package for Tsuchinoko/gpCAM
                if verbosity >= 5:
                    print('Packaging result: {:.4g} ± {:.4g}')

                result['value'] = [value, value_saxs]
                result['variance'] = [variance, variance_saxs]
                result['analyzed'] = True

        if verbosity >= 3:
            print('Analyzed {} results'.format(ianalyze))
        if verbosity >= 1 and ianalyze < 1:
            print('WARNING: No results were analyzed.')
        if verbosity >= 5:
            print_results(results)

        q.publish(results)


# infiles = glob.glob(os.path.join(source_dir, '*590873_*saxs.tiff'))
# infiles = glob.glob(os.path.join(source_dir, '*591166_*saxs.tiff'))
# infiles = glob.glob(os.path.join('../maxs/raw/', '*1858937_maxs.tiff'))

######################################

# Run WAXS Only

# process.run(infiles, protocols, output_dir=output_dir, force=True)

######################################

# Run SAXS Only
# source_dir = '../saxs/raw/'
# output_dir = './'

# pattern = '*'

# infiles = glob.glob(os.path.join(source_dir, pattern+'.tiff'))
# infiles.sort()
# process_saxs.run(infiles, protocols_saxs, output_dir=output_dir, force=True)

######################################

# Run Autonomous Loop

if __name__ == '__main__':
    # run_autonomous_loop(protocols, clear=False, force_load=False, republish=False, verbosity=5, simulate=True)
    # test_analysis(protocols, clear=False, force_load=False, republish=False, verbosity=5, simulate=False)
    run_autonomous_loop_tiled(protocols, clear=False, force_load=False, republish=False, verbosity=5, simulate=False)
# %%
