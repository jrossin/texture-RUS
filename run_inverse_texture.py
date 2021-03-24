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

    #Single crystal elastic constants C_11, C_12, C_44
    inputs.update({'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1})

    #Priors are set to -inf,inf for all parameters
    param_priors = {'v1111': ImproperUniform(), 'v1112': ImproperUniform(), 'v1113': ImproperUniform(), 'v1122': ImproperUniform(), 'v1123': ImproperUniform(), 'v1222': ImproperUniform(), 'v1223': ImproperUniform(), 'v2222': ImproperUniform(), 'v2223': ImproperUniform(), 'rs': ImproperUniform()}
    #initialize the simulation with a distribution for each parameter - normal distributions for all parameters besides the error.
    proposals = {'v1111': norm(0.001, 0.1),
                 'v1112': norm(0.001, 0.1),
                 'v1113': norm(0.001, 0.1),
                 'v1122': norm(0.001, 0.1),
                 'v1123': norm(0.001, 0.1),
                 'v1222': norm(0.001, 0.1),
                 'v1223': norm(0.001, 0.1),
                 'v2222': norm(0.001, 0.1),
                 'v2223': norm(0.001, 0.1),
                 'rs': norm(-0.01,0.005),
                 'std_dev': invgamma(25, scale=100)} # invgamma centered ~4

    #name of file (in same folder) with frequency data
    filename = 'freq_R2_CoNi_20deg_ab_70_polished'

    #Set SMCPy parameters - particles, timesteps, and number of markov chain mutation steps.
    inputs.update({'particles': 600, 'timesteps': 2, 'MCMC_steps': 2})

    inversemodel = runinv(name, 1, inputs, param_priors, filename)
    inversemodel.runinverse(param_priors, proposals)
