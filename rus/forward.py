import sys
import numpy as np
import rus.tools as rus

class formodel:

    def __init__(self, name, inputdict, params, param_order):
        '''
        :param param_order: a list of strings corresponding to model random
            variable names, in the order they will be passed to evaluate
        :type param_order: list of strings
        '''
        self.name = name
        self.param_order = param_order
        self.polynomial_order = inputdict.get('order', 8)

        self.density = inputdict.get('density', 6)
        self.ns = inputdict.get('num_moduli')
        self.nfreq = inputdict.get('nfreq', 100)
        #sample dimensions directly used
        self.d1 = inputdict.get('d1')
        self.d2 = inputdict.get('d2')
        self.d3 = inputdict.get('d3')

        self.sc = dict()
        if 'sc11' in inputdict:
            self.sc['sc11'] = inputdict.get('sc11')
            self.sc['sc12'] = inputdict.get('sc12')
            self.sc['sc44'] = inputdict.get('sc44')

        if 'v1111' in params or 'c400' in params:
            if len(self.sc) == 3:
                if len(params) in (9,10):
                    self.zeroboundlow, self.zeroboundhigh, self.csci_e, self.cs = rus.calc_zero_bounds(self.sc)
                else:
                    raise ValueError('Cubic materials require 9 4th order texture variables (V1111) to be specified for inverse/forward calcualtion or 9 4th order texture coefficients and 1 residual stress term')

            elif len(self.sc) != 0:
                raise ValueError('Wrong number of single crystal constants given'
                                 'for cubic -- other'
                                 'symmetries will be supported in a later release')

        #lookup is a vector of matrices (ends up being 3d data)
        self.cdict = dict()
        if 'c11' in params:
            for p in params:
                self.cdict[p] = params.get(p)

        self.cmat = np.zeros((6,6), dtype=np.float64)

        if self.polynomial_order == None:
            #set high polynomial order as convergent set of frequencies.
            poly_order_converge = 20
            self.polynomial_order = rus.poly_optimize(self.d1,self.d2,self.d3,self.density,self.nfreq,20)

        self.M, self.K_arr = rus.build_basis(self.polynomial_order, self.d1,
                                             self.d2,self.d3, self.density)

        self.outeigen = sys.stdout

    def evaluate(self, param_array):
        '''
        Handles both inputted cij or inputted tex vals
        '''
        outputs = []
        for param in param_array.copy():

            params = dict(zip(self.param_order, param))

            if 'v1111' in params:
                if rus.checkfrob4th_cub(params) == True:
                    outputs.append(np.ones(self.nfreq) * np.inf)

                else:


                    self.cmat = \
                            rus.texture_to_c_cub_run(params,self.csci_e,
                                                 self.cs)

                    self.cvect = rus.c_vect_create_mat(self.cmat)
                    freqs = rus.mech_rus(self.nfreq, self.M, self.K_arr,
                                        0.01 * self.cvect, self.polynomial_order)

                    if 'rs' in params:
                        npfreq = np.array(freqs)
                        freqs = npfreq + params.get('rs') * npfreq

                    freqs = freqs.tolist()
                    outputs.append(freqs)

            elif 'c11' in params:

                for p in params:
                    self.cdict[p] = params.get(p)
                self.cmat = rus.calc_forward_cm(self.cdict,self.ns)
                self.cvect = rus.c_vect_create_mat(self.cmat)
                freqs = rus.mech_rus(self.nfreq, self.M, self.K_arr,
                                        0.01 * self.cvect, self.polynomial_order)
                if 'rs' in params:
                        npfreq = np.array(freqs)
                        freqs = npfreq + params.get('rs') * npfreq
                freqs = freqs.tolist()

                outputs.append(freqs)
            else:
                raise ValueError('Invalid Input: texture values or elastic '
                                 'constants given are insufficient for '
                                 'calculation')

        outputs = np.array(outputs)
        axes = np.where(np.any(np.isnan(outputs)==True,axis=1))
        outputs[axes,:] = np.full(outputs.shape[1],np.inf)

        return outputs
