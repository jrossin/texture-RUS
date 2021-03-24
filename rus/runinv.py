import errno
import numpy as np
import os
import time
import sys

from datetime import datetime
from mpi4py import MPI
from scipy.optimize import minimize
import scipy.io

import rus.forward
import rus.tools as rus

from rus.forward import formodel
from rus.rus_propagator import rus_propagator

from smcpy.mcmc.parallel_mcmc import ParallelMCMC
from smcpy.utils.plotter import plot_pairwise
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel
from smcpy import SMCSampler, ImproperUniform
from smcpy.smc.propagator import Propagator


class runinv:

    def __init__(self, name, chain_number, input, params, filename):

        self.name = name
        self.dicttot = input.copy()
        self.chaincount = chain_number
        self.params = params
        self.filename = filename

        if 'c11' in params:
            if 'rs' in params:
                self.dicttot.update({'num_moduli': len(params)-1})
            else:
                self.dicttot.update({'num_moduli': len(params)})
        elif 'c400' in params or 'v1111' in params:
            if len(params) in (3,4):
                self.dicttot.update({'num_moduli': 9})
            elif len(params) in (9,10):
                self.dicttot.update({'num_moduli': 21})
            else:
                raise ValueError('Incorrect number of texture coefficients '
                                 'given. 3 constants need to be estimated for '
                                 'ortho sample symmetry, or 9 for '
                                 'cubic sample symmetry '
                                 '(respectively)')
        else:
            raise ValueError('Elastic constants (1-9 independent constants) or '
                             'texture coefficients (3 or 9) must be '
                             'specified as params')

    def runinverse(self, params, proposals):

        comm = MPI.COMM_WORLD.Clone()
        rank = comm.Get_rank()

        start_init = time.time()

        # set input file of frequencies
        exp_freqs, nfreq = rus.read_input(self.filename)
        self.dicttot.update({'nfreq': nfreq})
        data = np.array(exp_freqs)
        # set up model
        std_dev = None
        param_order = list(params.keys())
        param_priors = [params[key] for key in param_order]

        if std_dev is None:
            param_order +=  ['std_dev']
            param_priors += [ImproperUniform(0, None)]
        model = formodel(self.name, self.dicttot, params, param_order)
        start_model_eval = time.time()
        if rank == 0:
            print("Time to initialize model(s):")
            print(start_model_eval-start_init)
        # run smc
        num_particles = self.dicttot['particles']
        num_smc_steps = self.dicttot['timesteps']
        num_mcmc_steps = self.dicttot['MCMC_steps']
        ess_threshold = 0.75
        proposal_samples = dict()
        proposal_pdfs = np.zeros((num_particles, len(param_order)))
        for i, key in enumerate(param_order):
            proposal_samples[key] = proposals[key].rvs(num_particles)
            proposal_pdfs[:, i] = proposals[key].pdf(proposal_samples[key])
        proposal_pdfs = np.product(proposal_pdfs, axis=1)
        proposal = (proposal_samples, proposal_pdfs)
        mcmc = ParallelMCMC(model.evaluate, data, param_priors, comm, std_dev)
        mcmc_kernel = VectorMCMCKernel(mcmc, param_order=param_order)

        smc = SMCSampler(mcmc_kernel)

        phi_exp = 4
        x = np.linspace(0, 1, num_smc_steps)
        phi_sequence = (np.exp(x * phi_exp) - 1) / (np.exp(phi_exp) - 1)
        sys.stdout.flush()
        step_list, mll_list = smc.sample(num_particles=num_particles,
                                         num_mcmc_samples=num_mcmc_steps,
                                         phi_sequence=phi_sequence,
                                         ess_threshold=ess_threshold,
                                         proposal=proposal,
                                         progress_bar=False)

        prefix = 'results/inversion_output_' + self.name
        outputfolder = '_output/'
        basefname = prefix + outputfolder + 'chain_' + str(self.chaincount)
        filename = basefname  +'_results.txt'

        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        output_file = open(filename, 'a+')
        output_file.truncate(0)

        print('sampling complete for core')
        print(time.time()-start_init)
        sys.stdout.flush()
        print(f'step_list_len={len(step_list)}')
        if rank == 0:
            import pickle
            dumpfile = os.path.join(os.path.dirname(filename), 'dump.p')
            with open(dumpfile, 'wb') as pf:
                pickle.dump(step_list, pf)

            s = step_list[-1]

            mean = s.compute_mean()
            var = s.compute_variance()
            std = s.compute_std_dev()
            covar = s.compute_covariance()

            poly_order = model.polynomial_order

            print('Parameters used in Model:')
            print('Dimensions in mm: ' + str(model.d1) +','+ str(model.d2) +','+ str(model.d3) +', Density:' + str(model.density))
            printpriors = {k:(v.mean(),v.std()) for k, v in proposals.items()}
            output_file.writelines([f'\nInput Prior ImproperUniform()\nParameter Proposals are norm(Mean,Std_dev) for V or invgamma(mean,std) for sigma(std_dev):\n{printpriors}'])
            output_file.writelines([f'\n\n Dimension 1 in mm: {model.d1}'])
            output_file.writelines([f'\n Dimension 2 in mm: {model.d2}'])
            output_file.writelines([f'\n Dimension 3 in mm: {model.d3}'])
            output_file.writelines([f'\n Density: {model.density}'])
            output_file.writelines([f'\n Polynomial Order: {poly_order}'])
            output_file.writelines([f'\n SMCPy Parameters'])
            output_file.writelines([f'\n Particle Number: {num_particles}'])
            output_file.writelines([f'\n Number of time_steps: {num_smc_steps}'])
            output_file.writelines([f'\n Number of MCMC steps: {num_mcmc_steps}'])
            print('Calculated Variable Means:')
            print(mean)
            output_file.writelines([f'\n Calculated Variable Mean {mean}\n'])
            print('Calculated Variable  Variance:')
            print(var)
            output_file.writelines([f'\n Calculated Variable Variance {var}\n'])
            print('Calculated Variable Std-Dev')
            print(std)
            output_file.writelines([f'\n Calculated Variable Std Dev {std}\n'])
            print('Calculated Variable Covar:')
            print(covar)
            output_file.writelines([f'\n Calculated Variable Covar {covar}\n'])

            if 'rs' in mean:
                meanrs = 100.0 *  mean.get('rs')
                stdrs = 100.0 * std.get('rs')
                output_file.writelines([f'\n Residual stress factor across all frequencies (in %): \n N({meanrs}, {stdrs})'])

            outputpngname = basefname + '_pairwise_plot'
            plot_pairwise(s.params,weights=s.weights, save=True,
                          show=False, param_names=param_order,
                          prefix=outputpngname)

            if 'c11' in params:
                output_file.writelines([f'\n 6x6 Cij Mean Output \n {rus.calc_forward_cm(mean, model.ns)}'])
                output_file.writelines([f'\n 6x6 Cij Std Dev \n {rus.calc_forward_cm(std, model.ns)}'])
            elif 'v1111' in params:
                print('Single Crystal Elastic Constants used in Model:')
                print(model.sc)
                output_file.writelines([f'\n Single Crystal Elastic Constants used in Model: {model.sc}\n'])
                output_file.writelines([f'\n Mean 6x6 Vtens \n {rus.voigt(rus.gen_4th_varr(mean))}'])
                output_file.writelines([f'\n Std Dev 6x6 Vtens \n {rus.voigt(rus.gen_4th_varr(std))}'])
                rusprop = rus_propagator('propagated outputs', self.dicttot,params,param_order)
                if rusprop.ns == 9:
                    output_names = rus.out9param()
                elif rusprop.ns == 21:
                    output_names = rus.out21param()
                else:
                    raise ValueError('Number texture coefficients is not '
                        'registering correctly as solving for 9 or 21 elastic '
                        'constants..somehow - propagation of constants not '
                        'occuring correctly in rus_propagator')
                if len(output_names) == 30:
                    cijindex = slice(0,9)
                    hsuppslice = slice(9,18)
                    hslowslice = slice(18,27)
                    clmnslice = slice(27,30)
                elif len(output_names) == 72:
                    cijindex = slice(0,21)
                    hsuppslice = slice(21,42)
                    hslowslice = slice(42,63)
                    clmnslice = slice(63,72)

                resfreqname = []
                for i in range(nfreq):
                    resfreqname.append('fr_'+ str(i+1))
                resfreqname = np.array(resfreqname)
                output_names = np.append(output_names,resfreqname)
                #propagate outputs
                smc_prop = Propagator()
                smc_step_cij = smc_prop.propagate(rusprop.evaluate,s, output_names=output_names)
                smcstepcijmean = smc_step_cij.compute_mean()
                smcstepcijvar = smc_step_cij.compute_variance()
                smcstepcijstddev = smc_step_cij.compute_std_dev()
                smcstepcijcovar = smc_step_cij.compute_covariance()

                output_file.writelines([f'\n \n All Propagated Means: \n {smcstepcijmean}'])
                output_file.writelines([f'\n All Propagated Variances \n {smcstepcijvar}'])
                output_file.writelines([f'\n All Propagted Std Devs: \n {smcstepcijstddev}'])
                output_file.writelines([f'\n Ordered Covariance Matrix (by input parameter order) \n {smcstepcijcovar}'])

                output_file.writelines([f'\n Propagated Mean Self Consistent Cij 6x6 Voigt Tensor: \n {rus.calc_forward_cm(smcstepcijmean,rusprop.ns)}'])
                output_file.writelines([f'\n Propagated Std Dev Self Consistent Cij 6x6 Voigt Tensor \n {rus.calc_forward_cm(smcstepcijstddev,rusprop.ns)}'])

                HSuppavg = rus.forward_tric_HSupp(smcstepcijmean)
                HSuppstd = rus.forward_tric_HSupp(smcstepcijstddev)
                print('Mean Propagated UpperHS 6x6 bound:')
                print(HSuppavg)
                output_file.writelines([f'\n \n Mean Propagated UpperHS 6x6 bound: \n {HSuppavg}'])
                print('Upper HS 6x6 bound Propagated Std Dev:')
                print(HSuppstd)
                output_file.writelines([f'\n Upper 6x6 HS bound Propagated Std Dev: \n {HSuppstd}'])

                HSlowavg = rus.forward_tric_HSlow(smcstepcijmean)
                HSlowstd = rus.forward_tric_HSlow(smcstepcijstddev)
                print('Mean Propagated Lower HS 6x6 bound:')
                print(HSlowavg)
                output_file.writelines([f'\n \n Mean Propagated Lower HS 6x6 bound: \n {HSlowavg}'])
                print('Lower HS bound 6x6 Propagated Std Dev:')
                print(HSlowstd)
                output_file.writelines([f'\n Lower 6x6 HS bound Propagated Std Dev: \n {HSlowstd}'])

                cijkeys = output_names[cijindex]
                hsuppkeys = output_names[hsuppslice]
                hslowkeys = output_names[hslowslice]
                clmnkeys = output_names[clmnslice]

                rus.outputmatlabclmnrawdata(smc_step_cij.params[:,clmnslice],smc_step_cij.weights,basefname)


                clmnmean = rus.ctodictimagcub(smcstepcijmean)
                clmnvariance = rus.ctodictimagcub(smcstepcijvar)
                clmnstd = rus.ctodictimagcub(smcstepcijstddev)
                print('Mean Propagated Clmn:')
                print(clmnmean)
                output_file.writelines([f'\n \n Mean Propagated Clmn: \n {clmnmean}'])
                output_file.writelines([f'\n \n Variance Propagated Clmn: \n {clmnvariance}'])
                print('Std Dev Propagated Clmn:')
                print(clmnstd)
                output_file.writelines([f'\n Std Dev Propagated Clmn: \n{clmnstd}'])

                frmean = rus.fr_toarray(smcstepcijmean,resfreqname)
                frstd = rus.fr_toarray(smcstepcijstddev,resfreqname)
                output_file.writelines([f'\n \n Mean Propagated F_r: \n {frmean}'])
                output_file.writelines([f'\n \n Std Dev Propagated F_r: \n {frstd}'])

                outputpngnamepropcij = outputpngname + '_propagatedcij'
                plot_pairwise(smc_step_cij.params[:,cijindex],weights=smc_step_cij.weights,save=True, show=False,
                                                  param_names=cijkeys,
                                                  prefix=outputpngnamepropcij)

                outputpngnameprop_clmn = outputpngname + '_propagated_clmn'
                clmnparamslice = smc_step_cij.params[:,clmnslice]
                clmnimagcub = np.concatenate((np.stack(((clmnparamslice[:,8]).imag,(clmnparamslice[:,7]).imag,(clmnparamslice[:,6]).imag,(clmnparamslice[:,5]).imag),axis=1),clmnparamslice[:,4:9]),axis=1)

                clmnkeysalt = np.array(['c440imag','c430imag','c420imag','c410imag','c400','c410','c420','c430','c440'])

                plot_pairwise(clmnimagcub,weights=smc_step_cij.weights,save=True, show=False,
                                                  param_names=clmnkeysalt,
                                                  prefix=outputpngnameprop_clmn)

            elif 'c400' in params:
                print('Clmn input will be supported in a future release, use V '
                      'tensor input for now')
            else:
                raise ValueError('Inputs not recognized as Cij, Clmn, or Vijkl, exiting.')

        end1 = time.time()
        total_time = end1-start_init
        output_file.writelines([f'\n \n Total Inversion Time: \n {total_time}'])

        if rank == 0:
            #print("Output of Inversion is:")
            #print(output_file.readlines())
            print('Time of inversion evaluations (s):')
            print(total_time)
