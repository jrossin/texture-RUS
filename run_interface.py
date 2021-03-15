import numpy as np
from datetime import datetime
from scipy.stats import norm, invgamma
from rus_runinv import rus_runinv
from rusrun_forward import rusrun_forward
import rus_tools as rus
from smcpy import ImproperUniform


if __name__ == "__main__":
    # inputs - polynomial order (must be even), dims(m), density (kg/m^3),
    #A55 S2 SR
    #inputs = {'order': 12, 'd1': 0.002642, 'd2': 0.007055, 'd3': 0.0244225, 'density': 8535.899}
    #Ti64 brent isotropic
    #inputs = {'order': 12, 'd1': 0.00775333, 'd2': 0.00905666, 'd3':0.01319866, 'density': 4401.7}
    #E08 constants
    #inputs = {'order': 12, 'd1': 0.004858 , 'd2': 0.00682, 'd3': 0.018248, 'density': 8450.925}
    #CoNi R2 constants
    #inputs = {'order': 12, 'd1': 0.00691 , 'd2': 0.008249, 'd3': 0.008667, 'density': 8591.4}
    #CoNi R2 constants polished ab
    #inputs = {'order': 12, 'd1': 0.006914 , 'd2': 0.008215, 'd3': 0.008672, 'density': 8589.062}
    #CoNi R2 sr constants
    #inputs = {'order': 12, 'd1': 0.0069123, 'd2': 0.008214, 'd3': 0.008673, 'density': 8580.06}
    #CoNi R1 constants
    #inputs = {'order': 12, 'd1': 0.00838625, 'd2': 0.00887375, 'd3': 0.010325714, 'density': 8593.678145}
    #CoNi R4 constants
    inputs = {'order': 12, 'd1': 0.00837, 'd2': 0.008882857, 'd3': 0.010355714, 'density': 8593.955373}
    #CoNi R4 constants sr
    #inputs = {'order': 12, 'd1': 0.00837, 'd2': 0.008882857, 'd3': 0.010355714, 'density': 8594.54}
    #E70 constants
    #inputs = {'order': 12, 'd1': 0.00371833, 'd2': 0.005852, 'd3':0.01539, 'density': 8440.148}
    #E56_sectioned_s2
    #inputs = {'order': 12, 'd1': 0.001536, 'd2': 0.002890, 'd3':0.005252, 'density': 8445.628}
    #T64 sample B4 Z7 S1
    #inputs = {'order': 12, 'd1': 0.008928, 'd2': 0.009904, 'd3':0.01192, 'density': 4415.365365}

    problem_type = 'inverse'


    name = f"{str(datetime.now()).split(' ')[0]}"



    if problem_type == 'forward':

        #inputs.update({'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1}) #coni SC constants used
        inputs.update({'sc11': 243.3,'sc12': 156.7, 'sc44': 117.8})

        #cubic MICROSYMM
        #params = {'v1111': 0.1, 'v1112': 0.08, 'v1113': -0.02, 'v1122': -0.05, 'v1123': 0.015, 'v1222': -0.01, 'v1223': 0.025, 'v2222': 0.125, 'v2223': -0.01}
        params = rus.mctov({'c4-40':-0.0894337147196525 + 0.00418523311931692j,
'c4-30':0.00140163289627949 - 0.0150268399606209j,
'c4-20':-0.198399902597017 + 0.0131128463994099j,
'c4-10':-0.00174523794112499 - 0.0170905466923035j,
'c400':-0.0319998415787382 - 2.48940659263435e-17j,
'c410': -0.00174523794112497 + 0.0170905466923035j,
'c420':-0.198399902597017 - 0.0131128463994099j,
'c430':0.00140163289627951 + 0.0150268399606209j,
'c440':-0.0894337147196525 - 0.00418523311931695j
                        })
        print(rus.voigt((rus.gen_4th_varr(params)).real))
        #params = {'c11':276.43 ,'c22': 270.8 ,'c33': 271.68 ,'c44': 104.83 ,'c55': 98.73 ,'c66': 98.75 ,'c12': 131.2 ,'c13': 130.3 ,'c23': 135.97 ,'c14':2.272 ,'c15':4.184 ,'c16':16.3 ,'c24':1.57927 ,'c25':4.906 ,'c26':2.251 ,'c34':0.6936 ,'c35': 0.7222 ,'c36':14.117 ,'c45':17.113 ,'c46':6.25 ,'c56':4.26}
 # number of freqnecies to calculate
        inputs.update({'nfreq': 95})
        forwardmodel = rusrun_forward(name, inputs, params)
        forwardmodel.runforward(params)

    elif problem_type == 'inverse':

        np.random.seed(4)
        #Ti64 from Dawson et al.
        inputs.update({'sc11': 169.0, 'sc12': 89.0, 'sc13': 62.0, 'sc33': 196.0,'sc44': 43.0})
        #inputs.update({'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1}) #coni SC constants used
        #{'sc11': 227.6, 'sc12': 148.5, 'sc44': 126.0} prior coni constants - DO NOT USE
        #in625 from beese et al
        #inputs.update({'sc11': 243.3,'sc12': 156.7, 'sc44': 117.8})


        #triclinic symm
        #param_priors = {'c11':['normal', 290.0, 50.0],'c22':['normal', 290.0, 50.0],'c33':['normal', 290.0, 50.0],'c12':['normal', 130.0, 30.0],'c13':['normal', 130.0, 30.0],'c23':['normal', 130.0, 30.0],'c44':['normal', 85.5,15.0],'c55':['normal', 85.5,15.0],'c66':['normal', 85.5,15.0],'c14':['normal', 0.1, 30.0],'c15':['normal', 0.1, 30.0],'c16':['normal', 0.1, 30.0],'c24':['normal', 0.1, 30.0],'c25':['normal', 0.1, 30.0],'c26':['normal', 0.1, 30.0],'c34':['normal', 0.1, 30.0],'c35':['normal', 0.1, 30.0],'c36':['normal', 0.1, 30.0],'c45':['normal', 0.1, 30.0],'c46':['normal', 0.1, 30.0],'c56':['normal', 0.1, 30.0]}


        #HEXAGONAL MICROSYMM
        # param_priors = {'v211': ImproperUniform(),
        #                 'v212': ImproperUniform(),
        #                 'v213': ImproperUniform(),
        #                 'v222': ImproperUniform(),
        #                 'v223': ImproperUniform(),
        #                 'v1111': ImproperUniform(),
        #                 'v1112': ImproperUniform(),
        #                 'v1113': ImproperUniform(),
        #                 'v1122': ImproperUniform(),
        #                 'v1123': ImproperUniform(),
        #                 'v1222': ImproperUniform(),
        #                 'v1223': ImproperUniform(),
        #                 'v2222': ImproperUniform(),
        #                 'v2223': ImproperUniform()}
        # proposals = {'v211': norm(0.0, 0.2),
        #              'v212': norm(0.0, 0.2),
        #              'v213': norm(0.0, 0.2),
        #              'v222': norm(0.0, 0.2),
        #              'v223': norm(0.0, 0.2),
        #              'v1111': norm(0.0, 0.2),
        #              'v1112': norm(0.0, 0.2),
        #              'v1113': norm(0.0, 0.2),
        #              'v1122': norm(0.0, 0.2),
        #              'v1123': norm(0.0, 0.2),
        #              'v1222': norm(0.0, 0.2),
        #              'v1223': norm(0.0, 0.2),
        #              'v2222': norm(0.0, 0.2),
        #              'v2223': norm(0.0, 0.2),
        #               'std_dev': invgamma(25, scale=100)}


        #CUBIC MICROSYMM 9 independent coefficients at 4th order for cubic
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
                     #'std_dev': norm(1.28,0.078)}
                     'std_dev': invgamma(25, scale=100)} # invgamma centered ~4

        #name of file (in same folder) with frequency data

        filename = 'freq_R4_CoNi_0deg_ab_70'

        #filename = 'A55_s2_sr_55N'
        '''
        Set number of particles, timesteps, and number of markov chain steps.
        Good place to believe results is particles: 4000, timesteps 45,
        MCMC 5-8
        '''

        inputs.update({'particles': 500, 'timesteps': 2, 'MCMC_steps': 2})



        inversemodel = rus_runinv(name, 1, inputs, param_priors, filename)
        inversemodel.runinverse(param_priors, proposals)
