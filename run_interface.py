import numpy as np
from datetime import datetime
from scipy.stats import norm, invgamma
from rus_runinv import rus_runinv
from rusrun_forward import rusrun_forward
import rus_tools as rus
from smcpy import ImproperUniform


if __name__ == "__main__":
    # inputs - polynomial order (must be even), dims(m), density (kg/m^3),

    inputs = {'order': 12, 'd1': 0.006914 , 'd2': 0.008215, 'd3': 0.008672, 'density': 8589.062}
    problem_type = 'inverse'

    name = f"{str(datetime.now()).split(' ')[0]}"

    if problem_type == 'forward':
        inputs.update({'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1})
        params = {'v1111': 0.1, 'v1112': 0.08, 'v1113': -0.02, 'v1122': -0.05, 'v1123': 0.015, 'v1222': -0.01, 'v1223': 0.025, 'v2222': 0.125, 'v2223': -0.01}
        # number of freqnecies to calculate
        inputs.update({'nfreq': 95})
        forwardmodel = rusrun_forward(name, inputs, params)
        forwardmodel.runforward(params)

    elif problem_type == 'inverse':

        np.random.seed(4)
        inputs.update({'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1}) #coni SC

        #Directly calculate cij
        #param_priors = {'c11':['normal', 290.0, 50.0],'c22':['normal', 290.0, 50.0],'c33':['normal', 290.0, 50.0],'c12':['normal', 130.0, 30.0],'c13':['normal', 130.0, 30.0],'c23':['normal', 130.0, 30.0],'c44':['normal', 85.5,15.0],'c55':['normal', 85.5,15.0],'c66':['normal', 85.5,15.0],'c14':['normal', 0.1, 30.0],'c15':['normal', 0.1, 30.0],'c16':['normal', 0.1, 30.0],'c24':['normal', 0.1, 30.0],'c25':['normal', 0.1, 30.0],'c26':['normal', 0.1, 30.0],'c34':['normal', 0.1, 30.0],'c35':['normal', 0.1, 30.0],'c36':['normal', 0.1, 30.0],'c45':['normal', 0.1, 30.0],'c46':['normal', 0.1, 30.0],'c56':['normal', 0.1, 30.0]}

        param_priors = {'v1111': ImproperUniform(),
                        'v1112': ImproperUniform(),
                        'v1113': ImproperUniform(),
                        'v1122': ImproperUniform(),
                        'v1123': ImproperUniform(),
                        'v1222': ImproperUniform(),
                        'v1223': ImproperUniform(),
                        'v2222': ImproperUniform(),
                        'v2223': ImproperUniform(),
                        'rs': ImproperUniform()
                        }
        proposals = {'v1111': norm(0.18, 0.004),
                     'v1112': norm(-0.008, 0.016),
                     'v1113': norm(0.0013, 0.014),
                     'v1122': norm(-0.1, 0.003),
                     'v1123': norm(0.002, 0.011),
                     'v1222': norm(0.005, 0.012),
                     'v1223': norm(-0.002, 0.011),
                     'v2222': norm(0.18, 0.004),
                     'v2223': norm(-0.0045, 0.015),
                     'rs': norm(-0.0101,0.0008),
                     'std_dev': invgamma(25, scale=100)} # invgamma centered ~4

        #name of file (in same folder) with frequency data
        filename = 'freq_R2_CoNi_20deg_ab_70_polished'
        '''
        Set number of particles, timesteps, and number of markov chain steps.
        Good place to believe results is particles: 4000, timesteps 45,
        MCMC 5-8
        '''
        inputs.update({'particles': 5000, 'timesteps': 30, 'MCMC_steps': 6})

        inversemodel = rus_runinv(name, 1, inputs, param_priors, filename)
        inversemodel.runinverse(param_priors, proposals)
