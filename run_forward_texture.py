import numpy as np
from datetime import datetime
from rus.run_forward import run_forward
import rus.tools as rus

if __name__ == "__main__":
    # inputs - polynomial order (must be even), dims(m), density (kg/m^3),
    inputs = {'order': 12, 'd1': 0.006914 , 'd2': 0.008215, 'd3': 0.008672, 'density': 8589.062}

    #this is the output name for the simulation
    name = f"{str(datetime.now()).split(' ')[0]}"

    #Single crystal elastic constants C_11, C_12, C_44
    inputs.update({'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1})
    #Tensorial texture coefficients V_klmn
    params = {'v1111': 0.1, 'v1112': 0.08, 'v1113': -0.02, 'v1122': -0.05, 'v1123': 0.015, 'v1222': -0.01, 'v1223': 0.025, 'v2222': 0.125, 'v2223': -0.01}

    # number of frequencies to calculate
    inputs.update({'nfreq': 95})


    forwardmodel = run_forward(name, inputs, params)
    forwardmodel.runforward(params)
