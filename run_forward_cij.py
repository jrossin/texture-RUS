import numpy as np
from datetime import datetime
from rus.run_forward import run_forward
import rus.tools as rus

if __name__ == "__main__":
    # inputs - polynomial order (must be even), dims(m), density (kg/m^3),
    inputs = {'order': 12, 'd1': 0.006914 , 'd2': 0.008215, 'd3': 0.008672, 'density': 8589.062}

    #this is the output name for the simulation
    name = f"{str(datetime.now()).split(' ')[0]}"

    #Cij parameters. Units GPa. Can use various symmetries as detailed in rus/tools.py, triclinic shown here.
    params = {'c11': 276.43 ,'c22': 270.8 ,'c33': 271.68 ,'c44': 104.83 ,'c55': 98.73 ,'c66': 98.75 ,'c12': 131.2 ,'c13': 130.3 ,'c23': 135.97 ,'c14':2.272 ,'c15':4.184 ,'c16':16.3 ,'c24':1.57927 ,'c25':4.906 ,'c26':2.251 ,'c34':0.6936 ,'c35': 0.7222 ,'c36':14.117 ,'c45':17.113 ,'c46':6.25 ,'c56':4.26}

    # number of frequencies to calculate
    inputs.update({'nfreq': 95})


    forwardmodel = run_forward(name, inputs, params)
    forwardmodel.runforward(params)
