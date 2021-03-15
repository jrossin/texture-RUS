import sys
import scipy
import scipy.linalg.lapack as lapack
import numpy as np
from math import sqrt
import rus_tools as rus

from smcpy.model.base_model import BaseModel

class rus_propagator(BaseModel):
    def __init__(self, name, inputdict, params, param_order):

        self.name = name
        self.param_order = param_order
        self.polynomial_order = inputdict.get('order', 8)
        self.density = inputdict.get('density', 6)
        self.ns = inputdict.get('num_moduli', 9)

        #sample dimensions directly used
        self.d1 = inputdict.get('d1')
        self.d2 = inputdict.get('d2')
        self.d3 = inputdict.get('d3')

        self.sc = dict()
        self.sc['sc11'] = inputdict.get('sc11')
        self.sc['sc12'] = inputdict.get('sc12')
        self.sc['sc44'] = inputdict.get('sc44')
        if 'sc13' in inputdict:
            self.sc['sc13'] = inputdict.get('sc13')
        if 'sc33' in inputdict:
            self.sc['sc33'] = inputdict.get('sc33')
        if len(self.sc) in (3,5):
            self.zeroboundlow, self.zeroboundhigh, self.csci_e, self.cs = rus.calc_zero_bounds(self.sc)

        elif len(self.sc) != 0:
            print('Wrong number of single crystal constants given for cubic single xtal - other symmetries will be supported in a later release')
            sys.exit(0)


        self.cmat = np.zeros((6,6),dtype=np.float64)
        self.M, self.K_arr = rus.build_basis(self.polynomial_order, self.d1,
                                             self.d2,self.d3, self.density)

        self.nfreq = inputdict.get('nfreq', 100)

        self.outeigen = sys.stdout

    def evaluate(self, param_array):

        outputs = []
        i = 0
        for param in param_array.copy(): # evaluate now can take an array of inputs
            # print('propagated input parameter')
            # print(param)

            texvaldict = dict(zip(self.param_order, param))

            if 'c400' in texvaldict:
                vdict = rus.mctov(texvaldict)
            elif 'v1111' in texvaldict:
                vdict = texvaldict
            if rus.checkfrob4th_cub(vdict) == True:
                #outputs.append(np.ones(self.nfreq) * np.inf)
                i += 1
                print('Patricles in propagator failing frob check #: ' + str(i))
            if len(self.sc) == 3:
                self.cmat, self.upper_HS, self.lower_HS = rus.texture_to_c_cub(vdict, self.zeroboundlow, self.zeroboundhigh,self.csci_e, self.cs)
                clmn = rus.mvtoc(vdict)

            elif len(self.sc) == 5:
                self.cmat, self.upper_HS, self.lower_HS = rus.texture_to_c_hex(vdict, self.zeroboundlow, self.zeroboundhigh,self.csci_e, self.cs)
                clmn = rus.mvtoc_hex(vdict)
                clmnaddname = np.array([clmn.get('c2-20'),clmn.get('c2-10'),clmn.get('c200'),clmn.get('c210'),clmn.get('c220')])

            #self.cmat = 0.01 * self.cmat
            #for RUS calculation only, not here
            if self.ns == 9:
                cijout = np.array([self.cmat[0][0], self.cmat[0][1], self.cmat[0][2], self.cmat[1][1], self.cmat[1][2], self.cmat[2][2], self.cmat[3][3], self.cmat[4][4], self.cmat[5][5]])
                hsout = np.array([self.upper_HS[0][0], self.upper_HS[0][1], self.upper_HS[0][2], self.upper_HS[1][1], self.upper_HS[1][2], self.upper_HS[2][2], self.upper_HS[3][3], self.upper_HS[4][4], self.upper_HS[5][5],self.lower_HS[0][0], self.lower_HS[0][1], self.lower_HS[0][2], self.lower_HS[1][1], self.lower_HS[1][2], self.lower_HS[2][2], self.lower_HS[3][3], self.lower_HS[4][4], self.lower_HS[5][5]])
                clmnout = np.array([clmn.get('c400'),clmn.get('c420'),clmn.get('c440')])
            elif self.ns == 21:
                cijout = np.array([self.cmat[0][0], self.cmat[0][1], self.cmat[0][2], self.cmat[1][1], self.cmat[1][2], self.cmat[2][2], self.cmat[3][3], self.cmat[4][4], self.cmat[5][5],self.cmat[0][3],self.cmat[0][4],self.cmat[0][5],self.cmat[1][3],self.cmat[1][4],self.cmat[1][5],self.cmat[2][3],self.cmat[2][4],self.cmat[2][5],self.cmat[3][4],self.cmat[3][5],self.cmat[4][5]])
                hsout = np.array([self.upper_HS[0][0], self.upper_HS[0][1], self.upper_HS[0][2], self.upper_HS[1][1], self.upper_HS[1][2], self.upper_HS[2][2], self.upper_HS[3][3], self.upper_HS[4][4], self.upper_HS[5][5],self.upper_HS[0][3],self.upper_HS[0][4],self.upper_HS[0][5],self.upper_HS[1][3],self.upper_HS[1][4],self.upper_HS[1][5],self.upper_HS[2][3],self.upper_HS[2][4],self.upper_HS[2][5],self.upper_HS[3][4],self.upper_HS[3][5],self.upper_HS[4][5],self.lower_HS[0][0], self.lower_HS[0][1], self.lower_HS[0][2], self.lower_HS[1][1], self.lower_HS[1][2], self.lower_HS[2][2], self.lower_HS[3][3], self.lower_HS[4][4], self.lower_HS[5][5],self.lower_HS[0][3],self.lower_HS[0][4],self.lower_HS[0][5],self.lower_HS[1][3],self.lower_HS[1][4],self.lower_HS[1][5],self.lower_HS[2][3],self.lower_HS[2][4],self.lower_HS[2][5],self.lower_HS[3][4],self.lower_HS[3][5],self.lower_HS[4][5]])
                clmnout = np.array([clmn.get('c4-40'),clmn.get('c4-30'),clmn.get('c4-20'),clmn.get('c4-10'),clmn.get('c400'),clmn.get('c410'),clmn.get('c420'),clmn.get('c430'),clmn.get('c440')])
            addhs = np.append(cijout,hsout)
            addclmn = np.append(addhs,clmnout)
            if len(self.sc) == 5:
                addclmn = np.append(addclmn, clmnaddname)



            self.cvect = rus.c_vect_create_mat(self.cmat)
            freqs = rus.mech_rus(self.nfreq, self.M, self.K_arr,
                                        0.01 * self.cvect, self.polynomial_order)

            if 'rs' in texvaldict:
                npfreq = np.array(freqs)
                freqs = npfreq + texvaldict.get('rs') * npfreq


            npoutall = np.append(addclmn,freqs)

            outputs.append(npoutall)


        return np.array(outputs)
