import numpy as np
import os
import rus_tools as rus
from rus_forward_full import formodel
import time


class rusrun_forward():

    def __init__(self, name, input, params):

        self.name = name
        self.dicttot = input.copy()
        self.freqnum = self.dicttot.get('nfreq')
        self.params = params
        print(params)
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
                                 'cubic arbitrary sample symmetry '
                                 '(respectively) - NOTE RS term is an additional degree of freedom')
        else:
            raise ValueError('Elastic constants (1-9 independent constants) or '
                             'texture coefficients (3 or 9) must be '
                             'specified as params')

    def runforward(self, params):


        prefix = 'results/formodel_' + self.name
        outputfolder = '_output/'

        fname = prefix + outputfolder  +'_results.txt'

        if not os.path.exists(os.path.dirname(fname)):

            try:
                os.makedirs(os.path.dirname(fname))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        output_file = open(fname, 'a+')
        output_file.truncate(0)

        start_init = time.time()

        param_order = list(params.keys())
        param_values = np.array([[params[key] for key in param_order]])
        model = formodel(self.name, self.dicttot, params, param_order)

        start_model_eval = time.time()
        print("Time to initialize model(s):")
        model_init_time = start_model_eval-start_init
        print(model_init_time)
        output_file.writelines([f'Time to Initialize Model: \n {model_init_time}'])

        print('Parameters used in Model:')
        print('Dimensions in mm: ' + str(model.d1) +','+ str(model.d2) +','+ \
              str(model.d3) +', Density:' + str(model.density))

        polynomial_order = model.polynomial_order

        output_file.writelines([f'\n Dimension 1 in mm: {model.d1}'])
        output_file.writelines([f'\n Dimension 2 in mm: {model.d2}'])
        output_file.writelines([f'\n Dimension 3 in mm: {model.d3}'])
        output_file.writelines([f'\n Density: {model.density}'])
        output_file.writelines([f'\n Polynomial Order: {polynomial_order}'])

        outputfreqs = model.evaluate(param_values)

        if 'v1111' in params:
            print('Single Crystal Elastic Constants used in Model:')
            print(model.sc)
            output_file.writelines([f'\n Single Crystal Elastic Constants used in Model: \n {model.sc}'])
            output_file.writelines([f'\n Upper zero bound iso eigs, lower zero bounds iso eigs, self consistent iso eigs used in Model: \n {model.zeroboundhigh} {model.zeroboundlow} {model.csci_e}'])

            print('Input Parameters:')
            print(params)
            output_file.writelines([f'\n Input Parameters to Forward Model: \n',
                                   str(params)])
            output_file.writelines([f'\n 6x6 4th order Vtens \n {rus.voigt(rus.gen_4th_varr(params))}'])

            print('Upper HS Bound:')
            print(model.upper_HS)
            output_file.writelines([f'\n Upper HS bound from V input: \n {model.upper_HS}'])
            print('Lower HS Bound:')
            print(model.lower_HS)
            output_file.writelines([f'\n Lower HS bound from V input: \n {model.lower_HS}'])
            print('Reuss Bound:')
            print(model.reuss)
            output_file.writelines([f'\n Reuss bound from V input: \n {model.reuss}'])
            print('Voigt Bound:')
            print(model.voigt)
            output_file.writelines([f'\n Voigt bound from V input: \n {model.voigt}'])

            clmn = rus.mvtoc(params)

            output_file.writelines([f'\n Clmn: \n {clmn}'])
            print('Clmn:')
            print(clmn)

        print('Input Cij or Calculated Cij(Self Consistent Solution) Tensor:')
        print(model.cmat)
        output_file.writelines([f'\n Input or Calculated 6x6 C Tensor:\n {model.cmat}'])

        print('Calculated ' + str(self.freqnum) + ' Frequencies:')
        print(outputfreqs)
        output_file.writelines([f'\n Calculated {self.freqnum} \n Frequencies:\n {outputfreqs}'])

        endtime = time.time()
        print('Time of forward evaluation(s):')
        model_full_time = endtime - start_model_eval
        print(model_full_time)
        output_file.writelines([f'\n Time to Run Single Forward Model: \n {model_full_time}'])
