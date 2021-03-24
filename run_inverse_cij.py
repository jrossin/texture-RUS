import numpy as np
from datetime import datetime
from scipy.stats import norm, invgamma
from rus.runinv import runinv
import rus.tools as rus
from smcpy import ImproperUniform


if __name__ == "__main__":
    # inputs - polynomial order (must be even), dims(m), density (kg/m^3),
    inputs = {'order': 12, 'd1': 0.006914 , 'd2': 0.008215, 'd3': 0.008672, 'density': 8589.062}
    #this is the output name for the simulation
    name = f"{str(datetime.now()).split(' ')[0]}"

    #Priors are set to -inf,inf for all parameters
    param_priors = {'c11':ImproperUniform(), 'c22':ImproperUniform(), 'c33':ImproperUniform(), 'c12':ImproperUniform(), 'c13':ImproperUniform(), 'c23':ImproperUniform(), 'c44':ImproperUniform(), 'c55':ImproperUniform(), 'c66':ImproperUniform(), 'c14':ImproperUniform(), 'c15':ImproperUniform(), 'c16':ImproperUniform(), 'c24':ImproperUniform(), 'c25':ImproperUniform(), 'c26':ImproperUniform(), 'c34':ImproperUniform(), 'c35':ImproperUniform(), 'c36':ImproperUniform(), 'c45':ImproperUniform(), 'c46':ImproperUniform(), 'c56':ImproperUniform(), 'rs':ImproperUniform()}
    #initialize the simulation with a distribution for each parameter - normal distributions in this case
    proposals = {'c11':norm(280.0, 30.0),
        'c22':norm(280.0, 30.0),
        'c33':norm(280.0, 30.0),
        'c12':norm(125.0, 20.0),
        'c13':norm(125.0, 20.0),
        'c23':norm(125.0, 20.0),
        'c44':norm(90.0,10.0),
        'c55':norm(90.0,10.0),
        'c66':norm(90.0,10.0),
        'c14':norm(0.01, 2.0),
        'c15':norm(0.01, 30.0),
        'c16':norm(0.01, 1.0),
        'c24':norm(0.01, 1.0),
        'c25':norm(0.01, 10.0),
        'c26':norm(0.01, 1.0),
        'c34':norm(0.01, 1.0),
        'c35':norm(0.01, 30.0),
        'c36':norm(0.01, 1.0),
        'c45':norm(0.01, 1.0),
        'c46':norm(0.01, 10.0),
        'c56':norm(0.01, 1.0),
        'rs':norm(-0.010,0.005),
        'std_dev': invgamma(25, scale=100)}

    #name of file (in same folder) with frequency data
    filename = 'freq_R2_CoNi_20deg_ab_70_polished'

    #Set SMCPy parameters - particles, timesteps, and number of markov chain mutation steps.
    inputs.update({'particles': 3000, 'timesteps': 30, 'MCMC_steps': 6})

    inversemodel = runinv(name, 1, inputs, param_priors, filename)
    inversemodel.runinverse(param_priors, proposals)
