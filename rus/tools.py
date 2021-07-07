import itertools
import time
import sys
import scipy
import scipy.io
import numpy as np
from math import sqrt
import scipy.linalg as la
from decimal import *
from scipy.optimize import minimize, fsolve, root
import sympy as sp
from sympy.solvers.solveset import nonlinsolve
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy_backports.sympy_backports import linear_eq_to_matrix
from numba import jit
from collections import OrderedDict


def calc_forward_cm(cxx,ns):

    # isotropic
    if ns == 2:
        return forward_iso(cxx)
    # cubic
    if ns == 3:
        return forward_cub(cxx)
    # orthorhombic
    if ns == 9:
        return forward_orth(cxx)
    #triclinic
    if ns == 21:
        return forward_tric(cxx)

def read_input(infile):

    # get measured frequencies from file
    try:
        file_handle = open(infile, "rU")
    except IOError:
        print('Could not open frequency file.')
        sys.exit(-1)

    number_of_freqs = int(file_handle.readline())
    #print('nfreq={}'.format(number_of_freqs))
    freq_list = []

    for i in range(number_of_freqs):
        line = file_handle.readline()
        freq_list.append(float(line))
        #print(' freq={0:.6f}'.format(float(line)))
    file_handle.close()
    print('Experimental Frequencies to compare to:')
    print(freq_list)
    if len(freq_list) != number_of_freqs:
        print('Unexpected number of frequencies.')
        print('Expected {} but read {}'.format(number_of_freqs, len(freq_list)))
        raise ValueError('Exiting')

    return freq_list, number_of_freqs

# --------------------------------------------------
# Forward model translation of dictionary inputs to voigt notation
# --------------------------------------------------
def forward_iso(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.float64)
    try:
        cm[0,0] = cxx.get('c11')
        cm[3,3] = cxx.get('c44')
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[0,0] = cm[0,0]
        cm[3,3] = cm[3,3]
        cm[1,1] = cm[2,2] = cm[0,0]
        cm[4,4] = cm[5,5] = cm[3,3]
        cm[0,1] = cm[0,2] = cm[1,2] = cm[0,0] - 2.0 * cm[3,3]
        cm[1,0] = cm[2,0] = cm[2,1] = cm[0,0] - 2.0 * cm[3,3]
        return cm

# --------------------------------------------------
# CUBIC CALCULATIONS
# --------------------------------------------------
def forward_cub(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.float64)
    try:
        cm[0,0] = cxx.get('c11')
        cm[0,1] = cxx.get('c12')
        cm[3,3] = cxx.get('c44')
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[0,0] = cm[0,0]
        cm[0,1] = cm[0,1]
        cm[3,3] = cm[3,3]
        cm[1,1] = cm[2,2] = cm[0,0]
        cm[4,4] = cm[5,5] = cm[3,3]
        cm[0,2] = cm[1,2] = cm[0,1]
        cm[2,0] = cm[2,1] = cm[1,0] = cm[0,1]
        return cm
#----------------------------------------
# ORTHORHOMBIC CALCULATIONS
# --------------------------------------------------
def forward_orth(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.float64)
    try:
        cm[0,0] = cxx.get('c11')
        cm[1,1] = cxx.get('c22')
        cm[2,2] = cxx.get('c33')
        cm[1,2] = cxx.get('c23')
        cm[0,2] = cxx.get('c13')
        cm[0,1] = cxx.get('c12')
        cm[3,3] = cxx.get('c44')
        cm[4,4] = cxx.get('c55')
        cm[5,5] = cxx.get('c66')
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[0,0] = cm[0,0]
        cm[1,1] = cm[1,1]
        cm[2,2] = cm[2,2]
        cm[1,2] = cm[1,2]
        cm[0,2] = cm[0,2]
        cm[0,1] = cm[0,1]
        cm[3,3] = cm[3,3]
        cm[4,4] = cm[4,4]
        cm[5,5] = cm[5,5]
        cm[2,0] = cm[0,2]
        cm[1,0] = cm[0,1]
        cm[2,1] = cm[1,2]
        return cm

def forward_tric(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.float64)
    try:
        cm[0,0] = np.real(cxx.get('c11'))
        cm[1,1] = np.real(cxx.get('c22'))
        cm[2,2] = np.real(cxx.get('c33'))
        cm[1,2] = np.real(cxx.get('c23'))
        cm[0,2] = np.real(cxx.get('c13'))
        cm[0,1] = np.real(cxx.get('c12'))
        cm[3,3] = np.real(cxx.get('c44'))
        cm[4,4] = np.real(cxx.get('c55'))
        cm[5,5] = np.real(cxx.get('c66'))
        #
        cm[0,3] = np.real(cxx.get('c14'))
        cm[0,4] = np.real(cxx.get('c15'))
        cm[0,5] = np.real(cxx.get('c16'))
        cm[1,3] = np.real(cxx.get('c24'))
        cm[1,4] = np.real(cxx.get('c25'))
        cm[1,5] = np.real(cxx.get('c26'))
        cm[2,3] = np.real(cxx.get('c34'))
        cm[2,4] = np.real(cxx.get('c35'))
        cm[2,5] = np.real(cxx.get('c36'))
        cm[3,4] = np.real(cxx.get('c45'))
        cm[3,5] = np.real(cxx.get('c46'))
        cm[4,5] = np.real(cxx.get('c56'))
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[2,0] = cm[0,2]
        cm[1,0] = cm[0,1]
        cm[2,1] = cm[1,2]
        #
        cm[3,0] = cm[0,3]
        cm[4,0] = cm[0,4]
        cm[5,0] = cm[0,5]
        cm[3,1] = cm[1,3]
        cm[4,1] = cm[1,4]
        cm[5,1] = cm[1,5]
        cm[3,2] = cm[2,3]
        cm[4,2] = cm[2,4]
        cm[5,2] = cm[2,5]
        cm[4,3] = cm[3,4]
        cm[5,3] = cm[3,5]
        cm[5,4] = cm[4,5]
        return cm

def forward_tric_comp(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.cdouble)
    try:
        cm[0,0] = cxx.get('c11')
        cm[1,1] = cxx.get('c22')
        cm[2,2] = cxx.get('c33')
        cm[1,2] = cxx.get('c23')
        cm[0,2] = cxx.get('c13')
        cm[0,1] = cxx.get('c12')
        cm[3,3] = cxx.get('c44')
        cm[4,4] = cxx.get('c55')
        cm[5,5] = cxx.get('c66')
        cm[0,3] = cxx.get('c14')
        cm[0,4] = cxx.get('c15')
        cm[0,5] = cxx.get('c16')
        cm[1,3] = cxx.get('c24')
        cm[1,4] = cxx.get('c25')
        cm[1,5] = cxx.get('c26')
        cm[2,3] = cxx.get('c34')
        cm[2,4] = cxx.get('c35')
        cm[2,5] = cxx.get('c36')
        cm[3,4] = cxx.get('c45')
        cm[3,5] = cxx.get('c46')
        cm[4,5] = cxx.get('c56')
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[2,0] = cm[0,2]
        cm[1,0] = cm[0,1]
        cm[2,1] = cm[1,2]
        #
        cm[3,0] = cm[0,3]
        cm[4,0] = cm[0,4]
        cm[5,0] = cm[0,5]
        cm[3,1] = cm[1,3]
        cm[4,1] = cm[1,4]
        cm[5,1] = cm[1,5]
        cm[3,2] = cm[2,3]
        cm[4,2] = cm[2,4]
        cm[5,2] = cm[2,5]
        cm[4,3] = cm[3,4]
        cm[5,3] = cm[3,5]
        cm[5,4] = cm[4,5]
        return cm
def forward_tric_HSupp(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.float64)
    try:
        cm[0,0] = np.real(cxx.get('HS_Upper_C11'))
        cm[1,1] = np.real(cxx.get('HS_Upper_C22'))
        cm[2,2] = np.real(cxx.get('HS_Upper_C33'))
        cm[1,2] = np.real(cxx.get('HS_Upper_C23'))
        cm[0,2] = np.real(cxx.get('HS_Upper_C13'))
        cm[0,1] = np.real(cxx.get('HS_Upper_C12'))
        cm[3,3] = np.real(cxx.get('HS_Upper_C44'))
        cm[4,4] = np.real(cxx.get('HS_Upper_C55'))
        cm[5,5] = np.real(cxx.get('HS_Upper_C66'))
        cm[0,3] = np.real(cxx.get('HS_Upper_C14'))
        cm[0,4] = np.real(cxx.get('HS_Upper_C15'))
        cm[0,5] = np.real(cxx.get('HS_Upper_C16'))
        cm[1,3] = np.real(cxx.get('HS_Upper_C24'))
        cm[1,4] = np.real(cxx.get('HS_Upper_C25'))
        cm[1,5] = np.real(cxx.get('HS_Upper_C26'))
        cm[2,3] = np.real(cxx.get('HS_Upper_C34'))
        cm[2,4] = np.real(cxx.get('HS_Upper_C35'))
        cm[2,5] = np.real(cxx.get('HS_Upper_C36'))
        cm[3,4] = np.real(cxx.get('HS_Upper_C45'))
        cm[3,5] = np.real(cxx.get('HS_Upper_C46'))
        cm[4,5] = np.real(cxx.get('HS_Upper_C56'))
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[2,0] = cm[0,2]
        cm[1,0] = cm[0,1]
        cm[2,1] = cm[1,2]
        #
        cm[3,0] = cm[0,3]
        cm[4,0] = cm[0,4]
        cm[5,0] = cm[0,5]
        cm[3,1] = cm[1,3]
        cm[4,1] = cm[1,4]
        cm[5,1] = cm[1,5]
        cm[3,2] = cm[2,3]
        cm[4,2] = cm[2,4]
        cm[5,2] = cm[2,5]
        cm[4,3] = cm[3,4]
        cm[5,3] = cm[3,5]
        cm[5,4] = cm[4,5]
        return cm

def forward_tric_HSlow(cxx):
    """

    """
    cm = np.zeros((6,6),dtype=np.float64)
    try:
        cm[0,0] = np.real(cxx.get('HS_Lower_C11'))
        cm[1,1] = np.real(cxx.get('HS_Lower_C22'))
        cm[2,2] = np.real(cxx.get('HS_Lower_C33'))
        cm[1,2] = np.real(cxx.get('HS_Lower_C23'))
        cm[0,2] = np.real(cxx.get('HS_Lower_C13'))
        cm[0,1] = np.real(cxx.get('HS_Lower_C12'))
        cm[3,3] = np.real(cxx.get('HS_Lower_C44'))
        cm[4,4] = np.real(cxx.get('HS_Lower_C55'))
        cm[5,5] = np.real(cxx.get('HS_Lower_C66'))
        cm[0,3] = np.real(cxx.get('HS_Lower_C14'))
        cm[0,4] = np.real(cxx.get('HS_Lower_C15'))
        cm[0,5] = np.real(cxx.get('HS_Lower_C16'))
        cm[1,3] = np.real(cxx.get('HS_Lower_C24'))
        cm[1,4] = np.real(cxx.get('HS_Lower_C25'))
        cm[1,5] = np.real(cxx.get('HS_Lower_C26'))
        cm[2,3] = np.real(cxx.get('HS_Lower_C34'))
        cm[2,4] = np.real(cxx.get('HS_Lower_C35'))
        cm[2,5] = np.real(cxx.get('HS_Lower_C36'))
        cm[3,4] = np.real(cxx.get('HS_Lower_C45'))
        cm[3,5] = np.real(cxx.get('HS_Lower_C46'))
        cm[4,5] = np.real(cxx.get('HS_Lower_C56'))
    except KeyError:
        raise ValueError('Missing cxx value.')
    else:
        cm[2,0] = cm[0,2]
        cm[1,0] = cm[0,1]
        cm[2,1] = cm[1,2]

        cm[3,0] = cm[0,3]
        cm[4,0] = cm[0,4]
        cm[5,0] = cm[0,5]
        cm[3,1] = cm[1,3]
        cm[4,1] = cm[1,4]
        cm[5,1] = cm[1,5]
        cm[3,2] = cm[2,3]
        cm[4,2] = cm[2,4]
        cm[5,2] = cm[2,5]
        cm[4,3] = cm[3,4]
        cm[5,3] = cm[3,5]
        cm[5,4] = cm[4,5]
        return cm
## -----------------------------------------------------------------
## translated from CMDSTAN-RUS to avoid block diagonalization
## -----------------------------------------------------------------

@jit(nopython=True)
def build_basis(poly_order,d1,d2,d3,density):

    n_arr = []
    m_arr = []
    l_arr = []
    poly_order = np.int64(poly_order)
    L = np.int64((poly_order+1)*(poly_order+2)*(poly_order+3)/6)
    lookup_size = L * L * 3 * 3 * 21
    maxpoly = poly_order+1
    for i in range(maxpoly):
        for j in range(maxpoly):
            for k in range(maxpoly):
                if i + j + k <= poly_order:
                    n_arr.append(i)
                    m_arr.append(j)
                    l_arr.append(k)
    ns = np.array(n_arr)
    ms = np.array(m_arr)
    ls = np.array(l_arr)
    len_ns = ns.size
    if len_ns != L:
        raise ValueError("This should not ever happen, make Polynomial order even and try again")

    dp = np.zeros((L*3*3 + L*L*3*3,3,3),dtype=np.float64)

    pv = np.zeros((L,L),dtype=np.float64)

    Xs = np.zeros((2*poly_order+3),dtype=np.float64)
    Ys = np.zeros((2*poly_order+3),dtype=np.float64)
    Zs = np.zeros((2*poly_order+3),dtype=np.float64)
    for i in range(-1,2*poly_order+2):
        Xs[i+1] = d1**i
        Ys[i+1] = d2**i
        Zs[i+1] = d3**i

    dpm = np.zeros((3,3),dtype=np.float64)

    for i in range(len_ns):
        for j in range(len_ns):
            n0 = ns[i]
            m0 = ms[i]
            l0 = ls[i]
            n1 = ns[j]
            m1 = ms[j]
            l1 = ls[j]

            dpm[0,0] = Xs[n1 + n0] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 2, m1 + m0, l1 + l0) * n0 * n1
            dpm[0,1] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * n0 * m1
            dpm[0,2] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * n0 * l1

            dpm[1,0] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * m0 * n1
            dpm[1,1] = Xs[n1 + n0 + 2] * Ys[m1 + m0] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0 - 2, l1 + l0) * m0 * m1
            dpm[1,2] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * m0 * l1

            dpm[2,0] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * l0 * n1
            dpm[2,1] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * l0 * m1
            dpm[2,2] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0] * polyint(n1 + n0, m1 + m0, l1 + l0 - 2) * l0 * l1

            pv[i,j] = density * Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0, l1 + l0)
            dp[i * 3 * 3 + j * len_ns * 3 * 3,:,:] = dpm

    M = buildM(poly_order,pv)

    lookupind = np.zeros((21,6,6))
    k_arr = []


    ij = 0
    for i in range(6):
        for j in range(i+1):
            dCdcij = np.zeros((6,6),dtype=np.float64)
            dCdcij[i,j] = 1.0
            dCdcij[j,i] = 1.0

            dktmp = buildK(poly_order,dCdcij,dp)

            k_arr.append(dktmp)


    if (len(k_arr) % 21) != 0:
       raise ValueError("k array.len() must be a multiple of 21!")


    return M,k_arr

@jit(nopython=True)
def buildK(poly_order, Ch, dp):

      L = np.int64((poly_order + 1) * (poly_order + 2) * (poly_order + 3) / 6)

      K = np.zeros((L * 3, L * 3),dtype = np.float64)

      C = voigt_stan(Ch)


      for n in range(L):
        for m in range(L):

          for i in range(3):
            for k in range(3):
                total = 0.0

                for j in range(3):
                  for l in range(3):

                    total = total + (C[i + 3 * j, k + 3 * l] * dp[n * 3 * 3 + m * L * 3 * 3,j,l])

                K[n * 3 + i, m * 3 + k] = total

      return K

@jit(nopython=True)
def buildM(poly_order,pv):

    L = np.int64((poly_order + 1) * (poly_order + 2) * (poly_order + 3) / 6)

    M = np.zeros((L * 3, L * 3),dtype=np.float64)

    for n in range(L):
        for m in range(L):
            M[n * 3 + 0, m * 3 + 0] = pv[n, m]
            M[n * 3 + 1, m * 3 + 1] = pv[n, m]
            M[n * 3 + 2, m * 3 + 2] = pv[n, m]

    return M
def c_vect_create_dict(cdict):

    c_vect = np.array(([cdict['c11'],cdict['c12'],cdict['c22'],cdict['c13'],cdict['c23'],cdict['c33'],cdict['c14'],cdict['c24'],cdict['c34'],cdict['c44'],cdict['c15'],cdict['c25'],cdict['c35'],cdict['c45'],cdict['c55'],cdict['c16'],cdict['c26'],cdict['c36'],cdict['c46'],cdict['c56'],cdict['c66']]),dtype=np.float64)
    return c_vect
def c_vect_create_mat(cmat):

    c_vect = np.array(([cmat[0,0],cmat[0,1],cmat[1,1],cmat[0,2],cmat[1,2],cmat[2,2],cmat[0,3],cmat[1,3],cmat[2,3],cmat[3,3],cmat[4,0],cmat[4,1],cmat[2,4],cmat[3,4],cmat[4,4],cmat[0,5],cmat[1,5],cmat[2,5],cmat[3,5],cmat[4,5],cmat[5,5]]),dtype=np.float64)
    return c_vect
def flatten_stan(Ksize,lookup, C_, L):
    lookup_ = [scipy.sparse.csc_matrix((3*L,3*L),dtype=np.float64)] * Ksize * C_.size

    for ij in range(C_.size):
        for k in range(Ksize):
            lookup_[ij*Ksize+k] += lookup[ij * Ksize + k]

    return lookup_
@jit(nopython=True)
def polyint(n,m,l):
    if (n < 0) or (m < 0) or (l < 0) or (n % 2 > 0) or (m % 2 > 0) or (l % 2 > 0):
        return 0.0
    xtmp = 2.0 * (0.5**(n + 1))
    ytmp = 2.0 * xtmp * (0.5**(m + 1))
    result = 2.0 * (0.5**(l + 1)) * ytmp / ((n + 1) * (m + 1) * (l + 1))
    return result

def mech_rus(N, M,K_arr, C,P):

    freqs, dfreqsdCij = mechanics(C,M,K_arr,N,P)


    return freqs
def mechanics(C,M,K_arr,nevs,P):

    L = np.int64((P+1)*(P+2)*(P+3)/6)
    K = np.zeros((3*L,3*L),dtype=np.float64)
    for i in range(C.size):
        K += K_arr[i] * C[i]
    evals,evecs = la.eigh(K, M, eigvals = (0,6+nevs-1), check_finite=True)
    dfreqsdCij = np.zeros((nevs, C.size),dtype=np.float64)

    for i in range(6):
        if (evals[i])>1E-2:
            print("Eigenvalue " + str(i) + " is " + str(evals[i]) +  "(should be near zeros for the first 6 given tralations and rotations, tolerance for this calcualtion is 1e-2)")
            raise ValueError('Exiting')
    freqstmp = np.zeros((nevs),dtype=np.float64)
    for i in range(nevs):
        freqstmp[i] = evals[i+6]

    freqs = np.sqrt(freqstmp * 1.0E11) / (np.pi * 2000.0)
    return freqs, dfreqsdCij
## ----------------------------------------------------------------------------
####### 9x9 VOIGT transforms from CMDSTAN RUS
## ----------------------------------------------------------------------------
@jit(nopython=True)
def voigt_stan(Ch):
    #take in 6x6 Ch numpy and spit out 9x9 C np array
    C = np.zeros((9,9),dtype=np.float64)
    C[0 + 0 * 3, 0 + 0 * 3] = Ch[0, 0]
    C[0 + 0 * 3, 1 + 1 * 3] = Ch[0, 1]
    C[0 + 0 * 3, 2 + 2 * 3] = Ch[0, 2]
    C[0 + 0 * 3, 1 + 2 * 3] = Ch[0, 3]
    C[0 + 0 * 3, 2 + 1 * 3] = Ch[0, 3]
    C[0 + 0 * 3, 0 + 2 * 3] = Ch[0, 4]
    C[0 + 0 * 3, 2 + 0 * 3] = Ch[0, 4]
    C[0 + 0 * 3, 0 + 1 * 3] = Ch[0, 5]
    C[0 + 0 * 3, 1 + 0 * 3] = Ch[0, 5]
    C[1 + 1 * 3, 0 + 0 * 3] = Ch[1, 0]
    C[1 + 1 * 3, 1 + 1 * 3] = Ch[1, 1]
    C[1 + 1 * 3, 2 + 2 * 3] = Ch[1, 2]
    C[1 + 1 * 3, 1 + 2 * 3] = Ch[1, 3]
    C[1 + 1 * 3, 2 + 1 * 3] = Ch[1, 3]
    C[1 + 1 * 3, 0 + 2 * 3] = Ch[1, 4]
    C[1 + 1 * 3, 2 + 0 * 3] = Ch[1, 4]
    C[1 + 1 * 3, 0 + 1 * 3] = Ch[1, 5]
    C[1 + 1 * 3, 1 + 0 * 3] = Ch[1, 5]
    C[2 + 2 * 3, 0 + 0 * 3] = Ch[2, 0]
    C[2 + 2 * 3, 1 + 1 * 3] = Ch[2, 1]
    C[2 + 2 * 3, 2 + 2 * 3] = Ch[2, 2]
    C[2 + 2 * 3, 1 + 2 * 3] = Ch[2, 3]
    C[2 + 2 * 3, 2 + 1 * 3] = Ch[2, 3]
    C[2 + 2 * 3, 0 + 2 * 3] = Ch[2, 4]
    C[2 + 2 * 3, 2 + 0 * 3] = Ch[2, 4]
    C[2 + 2 * 3, 0 + 1 * 3] = Ch[2, 5]
    C[2 + 2 * 3, 1 + 0 * 3] = Ch[2, 5]
    C[1 + 2 * 3, 0 + 0 * 3] = Ch[3, 0]
    C[2 + 1 * 3, 0 + 0 * 3] = Ch[3, 0]
    C[1 + 2 * 3, 1 + 1 * 3] = Ch[3, 1]
    C[2 + 1 * 3, 1 + 1 * 3] = Ch[3, 1]
    C[1 + 2 * 3, 2 + 2 * 3] = Ch[3, 2]
    C[2 + 1 * 3, 2 + 2 * 3] = Ch[3, 2]
    C[1 + 2 * 3, 1 + 2 * 3] = Ch[3, 3]
    C[1 + 2 * 3, 2 + 1 * 3] = Ch[3, 3]
    C[2 + 1 * 3, 1 + 2 * 3] = Ch[3, 3]
    C[2 + 1 * 3, 2 + 1 * 3] = Ch[3, 3]
    C[1 + 2 * 3, 0 + 2 * 3] = Ch[3, 4]
    C[1 + 2 * 3, 2 + 0 * 3] = Ch[3, 4]
    C[2 + 1 * 3, 0 + 2 * 3] = Ch[3, 4]
    C[2 + 1 * 3, 2 + 0 * 3] = Ch[3, 4]
    C[1 + 2 * 3, 0 + 1 * 3] = Ch[3, 5]
    C[1 + 2 * 3, 1 + 0 * 3] = Ch[3, 5]
    C[2 + 1 * 3, 0 + 1 * 3] = Ch[3, 5]
    C[2 + 1 * 3, 1 + 0 * 3] = Ch[3, 5]
    C[0 + 2 * 3, 0 + 0 * 3] = Ch[4, 0]
    C[2 + 0 * 3, 0 + 0 * 3] = Ch[4, 0]
    C[0 + 2 * 3, 1 + 1 * 3] = Ch[4, 1]
    C[2 + 0 * 3, 1 + 1 * 3] = Ch[4, 1]
    C[0 + 2 * 3, 2 + 2 * 3] = Ch[4, 2]
    C[2 + 0 * 3, 2 + 2 * 3] = Ch[4, 2]
    C[0 + 2 * 3, 1 + 2 * 3] = Ch[4, 3]
    C[0 + 2 * 3, 2 + 1 * 3] = Ch[4, 3]
    C[2 + 0 * 3, 1 + 2 * 3] = Ch[4, 3]
    C[2 + 0 * 3, 2 + 1 * 3] = Ch[4, 3]
    C[0 + 2 * 3, 0 + 2 * 3] = Ch[4, 4]
    C[0 + 2 * 3, 2 + 0 * 3] = Ch[4, 4]
    C[2 + 0 * 3, 0 + 2 * 3] = Ch[4, 4]
    C[2 + 0 * 3, 2 + 0 * 3] = Ch[4, 4]
    C[0 + 2 * 3, 0 + 1 * 3] = Ch[4, 5]
    C[0 + 2 * 3, 1 + 0 * 3] = Ch[4, 5]
    C[2 + 0 * 3, 0 + 1 * 3] = Ch[4, 5]
    C[2 + 0 * 3, 1 + 0 * 3] = Ch[4, 5]
    C[0 + 1 * 3, 0 + 0 * 3] = Ch[5, 0]
    C[1 + 0 * 3, 0 + 0 * 3] = Ch[5, 0]
    C[0 + 1 * 3, 1 + 1 * 3] = Ch[5, 1]
    C[1 + 0 * 3, 1 + 1 * 3] = Ch[5, 1]
    C[0 + 1 * 3, 2 + 2 * 3] = Ch[5, 2]
    C[1 + 0 * 3, 2 + 2 * 3] = Ch[5, 2]
    C[0 + 1 * 3, 1 + 2 * 3] = Ch[5, 3]
    C[0 + 1 * 3, 2 + 1 * 3] = Ch[5, 3]
    C[1 + 0 * 3, 1 + 2 * 3] = Ch[5, 3]
    C[1 + 0 * 3, 2 + 1 * 3] = Ch[5, 3]
    C[0 + 1 * 3, 0 + 2 * 3] = Ch[5, 4]
    C[0 + 1 * 3, 2 + 0 * 3] = Ch[5, 4]
    C[1 + 0 * 3, 0 + 2 * 3] = Ch[5, 4]
    C[1 + 0 * 3, 2 + 0 * 3] = Ch[5, 4]
    C[0 + 1 * 3, 0 + 1 * 3] = Ch[5, 5]
    C[0 + 1 * 3, 1 + 0 * 3] = Ch[5, 5]
    C[1 + 0 * 3, 0 + 1 * 3] = Ch[5, 5]
    C[1 + 0 * 3, 1 + 0 * 3] = Ch[5, 5]

    return C

def unvoigt_stan(Ch):
  #take in 9x9 Ch numpy and spit out 6x6 C
  C = np.zeros((6,6),dtype=np.float64)
  C[0, 1] = Ch[0 + 0 * 3, 1 + 1 * 3]
  C[0, 2] = Ch[0 + 0 * 3, 2 + 2 * 3]
  C[0, 3] = Ch[0 + 0 * 3, 1 + 2 * 3]
  C[0, 3] = Ch[0 + 0 * 3, 2 + 1 * 3]
  C[0, 4] = Ch[0 + 0 * 3, 0 + 2 * 3]
  C[0, 4] = Ch[0 + 0 * 3, 2 + 0 * 3]
  C[0, 5] = Ch[0 + 0 * 3, 0 + 1 * 3]
  C[0, 5] = Ch[0 + 0 * 3, 1 + 0 * 3]
  C[1, 0] = Ch[1 + 1 * 3, 0 + 0 * 3]
  C[1, 1] = Ch[1 + 1 * 3, 1 + 1 * 3]
  C[1, 2] = Ch[1 + 1 * 3, 2 + 2 * 3]
  C[1, 3] = Ch[1 + 1 * 3, 1 + 2 * 3]
  C[1, 3] = Ch[1 + 1 * 3, 2 + 1 * 3]
  C[1, 4] = Ch[1 + 1 * 3, 0 + 2 * 3]
  C[1, 4] = Ch[1 + 1 * 3, 2 + 0 * 3]
  C[1, 5] = Ch[1 + 1 * 3, 0 + 1 * 3]
  C[1, 5] = Ch[1 + 1 * 3, 1 + 0 * 3]
  C[2, 0] = Ch[2 + 2 * 3, 0 + 0 * 3]
  C[2, 1] = Ch[2 + 2 * 3, 1 + 1 * 3]
  C[2, 2] = Ch[2 + 2 * 3, 2 + 2 * 3]
  C[0, 0] = Ch[0 + 0 * 3, 0 + 0 * 3]
  C[2, 3] = Ch[2 + 2 * 3, 1 + 2 * 3]
  C[2, 3] = Ch[2 + 2 * 3, 2 + 1 * 3]
  C[2, 4] = Ch[2 + 2 * 3, 0 + 2 * 3]
  C[2, 4] = Ch[2 + 2 * 3, 2 + 0 * 3]
  C[2, 5] = Ch[2 + 2 * 3, 0 + 1 * 3]
  C[2, 5] = Ch[2 + 2 * 3, 1 + 0 * 3]
  C[3, 0] = Ch[1 + 2 * 3, 0 + 0 * 3]
  C[3, 0] = Ch[2 + 1 * 3, 0 + 0 * 3]
  C[3, 1] = Ch[1 + 2 * 3, 1 + 1 * 3]
  C[3, 1] = Ch[2 + 1 * 3, 1 + 1 * 3]
  C[3, 2] = Ch[1 + 2 * 3, 2 + 2 * 3]
  C[3, 2] = Ch[2 + 1 * 3, 2 + 2 * 3]
  C[3, 3] = Ch[1 + 2 * 3, 1 + 2 * 3]
  C[3, 3] = Ch[1 + 2 * 3, 2 + 1 * 3]
  C[3, 3] = Ch[2 + 1 * 3, 1 + 2 * 3]
  C[3, 3] = Ch[2 + 1 * 3, 2 + 1 * 3]
  C[3, 4] = Ch[1 + 2 * 3, 0 + 2 * 3]
  C[3, 4] = Ch[1 + 2 * 3, 2 + 0 * 3]
  C[3, 4] = Ch[2 + 1 * 3, 0 + 2 * 3]
  C[3, 4] = Ch[2 + 1 * 3, 2 + 0 * 3]
  C[3, 5] = Ch[1 + 2 * 3, 0 + 1 * 3]
  C[3, 5] = Ch[1 + 2 * 3, 1 + 0 * 3]
  C[3, 5] = Ch[2 + 1 * 3, 0 + 1 * 3]
  C[3, 5] = Ch[2 + 1 * 3, 1 + 0 * 3]
  C[4, 0] = Ch[0 + 2 * 3, 0 + 0 * 3]
  C[4, 0] = Ch[2 + 0 * 3, 0 + 0 * 3]
  C[4, 1] = Ch[0 + 2 * 3, 1 + 1 * 3]
  C[4, 1] = Ch[2 + 0 * 3, 1 + 1 * 3]
  C[4, 2] = Ch[0 + 2 * 3, 2 + 2 * 3]
  C[4, 2] = Ch[2 + 0 * 3, 2 + 2 * 3]
  C[4, 3] = Ch[0 + 2 * 3, 1 + 2 * 3]
  C[4, 3] = Ch[0 + 2 * 3, 2 + 1 * 3]
  C[4, 3] = Ch[2 + 0 * 3, 1 + 2 * 3]
  C[4, 3] = Ch[2 + 0 * 3, 2 + 1 * 3]
  C[4, 4] = Ch[0 + 2 * 3, 0 + 2 * 3]
  C[4, 4] = Ch[0 + 2 * 3, 2 + 0 * 3]
  C[4, 4] = Ch[2 + 0 * 3, 0 + 2 * 3]
  C[4, 4] = Ch[2 + 0 * 3, 2 + 0 * 3]
  C[4, 5] = Ch[0 + 2 * 3, 0 + 1 * 3]
  C[4, 5] = Ch[0 + 2 * 3, 1 + 0 * 3]
  C[4, 5] = Ch[2 + 0 * 3, 0 + 1 * 3]
  C[4, 5] = Ch[2 + 0 * 3, 1 + 0 * 3]
  C[5, 0] = Ch[0 + 1 * 3, 0 + 0 * 3]
  C[5, 0] = Ch[1 + 0 * 3, 0 + 0 * 3]
  C[5, 1] = Ch[0 + 1 * 3, 1 + 1 * 3]
  C[5, 1] = Ch[1 + 0 * 3, 1 + 1 * 3]
  C[5, 2] = Ch[0 + 1 * 3, 2 + 2 * 3]
  C[5, 2] = Ch[1 + 0 * 3, 2 + 2 * 3]
  C[5, 3] = Ch[0 + 1 * 3, 1 + 2 * 3]
  C[5, 3] = Ch[0 + 1 * 3, 2 + 1 * 3]
  C[5, 3] = Ch[1 + 0 * 3, 1 + 2 * 3]
  C[5, 3] = Ch[1 + 0 * 3, 2 + 1 * 3]
  C[5, 4] = Ch[0 + 1 * 3, 0 + 2 * 3]
  C[5, 4] = Ch[0 + 1 * 3, 2 + 0 * 3]
  C[5, 4] = Ch[1 + 0 * 3, 0 + 2 * 3]
  C[5, 4] = Ch[1 + 0 * 3, 2 + 0 * 3]
  C[5, 5] = Ch[0 + 1 * 3, 0 + 1 * 3]
  C[5, 5] = Ch[0 + 1 * 3, 1 + 0 * 3]
  C[5, 5] = Ch[1 + 0 * 3, 0 + 1 * 3]
  C[5, 5] = Ch[1 + 0 * 3, 1 + 0 * 3]
  return C


## --------------------------------------------
## Regular voigt transforms
## --------------------------------------------

def recover4order(voigt):
    tensor3s = np.zeros((3,3,3,3),dtype=np.float64)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        v_i = help_3x3(i, j)
        v_j = help_3x3(k, l)
        tensor3s[i, j, k, l] = voigt[v_i, v_j]
    return tensor3s
def recover4ordercomplex(voigt):

    tensor3s = np.zeros((3,3,3,3),dtype=complex)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        v_i = help_3x3(i, j)
        v_j = help_3x3(k, l)
        tensor3s[i, j, k, l] = voigt[v_i, v_j]
    return tensor3s
def voigt(tensor3s):
    #regular voigt notation switch (unnormalized)
    c_voigt = np.zeros((6,6), dtype=np.float64)
    voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    for i,j in itertools.product(range(6),range(6)):
        k, l = voigt_notation[i]
        m, n = voigt_notation[j]
        c_voigt[i,j] = tensor3s[k,l,m,n]
    return c_voigt

def help_3x3(i, j):
    if i == j:
        return i
    return 6-i-j

## ----------------------------------------------------------------------------
## these functions are the normalized voigt (mandel notation transformations) to remain consistent within the fernandez HS framework
## ----------------------------------------------------------------------------

def gen_4th_varr(vdict):

    v = np.zeros((6,6),dtype=np.float64)


    v[0,0] = vdict.get('v1111')
    v[1,1] = vdict.get('v2222')
    v[2,2] = vdict.get('v1111') + 2.0*vdict.get('v1122') + vdict.get('v2222') #v1111 + 2*v1122 + v2222

    v[3,3] = -vdict.get('v1122') - vdict.get('v2222') #v44 = -v12-v22
    v[4,4] = -vdict.get('v1111') - vdict.get('v1122') #v55 = -v11-v12
    v[5,5] = vdict.get('v1122')

    v[0,1] = v[1,0] = vdict.get('v1122') #v1122
    v[2,0] = v[0,2] = -(vdict.get('v1111') + vdict.get('v1122')) #v13 = -v11-v12
    v[1,2] = v[2,1] = -(vdict.get('v1122') + vdict.get('v2222')) #v23 = -v12-v22

    v[0,3] = v[3,0] = vdict.get('v1123') #v1123
    v[0,4] = v[4,0] = vdict.get('v1113') #v1113
    v[0,5] = v[5,0] = vdict.get('v1112') #v1112

    v[1,3] = v[3,1] = vdict.get('v2223') #v2223
    v[1,4] = v[4,1] = vdict.get('v1223') #v1223
    v[1,5] = v[5,1] = vdict.get('v1222') #v1222

    v[2,3] = v[3,2] = -(vdict.get('v1123') + vdict.get('v2223')) #-v1123 - v2223
    v[2,4] = v[4,2] = -(vdict.get('v1113') + vdict.get('v1223')) #-v1113 - v1223
    v[2,5] = v[5,2] = -(vdict.get('v1112') + vdict.get('v1222')) #-v1112 - v1222

    v[3,4] = v[4,3] = -(vdict.get('v1112')+vdict.get('v1222')) #-(v1112 + v1222)
    v[3,5] = v[5,3] = vdict.get('v1223') #v1223

    v[4,5] = v[5,4] = vdict.get('v1123') #v1123

    fourthorder = recover4order(v)

    return fourthorder


def recovernormalized4th(sixsix):

    fourth_order3smat = np.zeros((3,3,3,3),dtype=np.float64)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        v_i = help_3x3(i, j)
        v_j = help_3x3(k, l)
        if (v_i == v_j == 3) or (v_i == v_j == 4) or (v_i == v_j == 5) or (v_i == 3 and v_j == 4) or (v_i == 4 and v_j == 3) or \
        (v_i == 3 and v_j == 5) or (v_i == 5 and v_j == 3) or (v_i == 4 and v_j == 5) or (v_i == 5 and v_j == 4):
            fourth_order3smat[i, j, k, l] = 1./2. * sixsix[v_i, v_j]
        elif (v_i == v_j == 0) or (v_i == v_j == 1) or (v_i == v_j == 2) or (v_i == 0 and v_j == 1) or (v_i == 1 and v_j == 0) or \
        (v_i == 0 and v_j == 2) or (v_i == 2 and v_j == 0) or (v_i == 1 and v_j == 2) or (v_i == 2 and v_j == 1):
            fourth_order3smat[i, j, k, l] = sixsix[v_i, v_j]
        else:
            fourth_order3smat[i, j, k, l] = 1./np.sqrt(2.) * sixsix[v_i, v_j]
    return fourth_order3smat

def recovernormalizedvoigt(tensor3s):
    voigt = np.zeros((6,6), dtype=np.float64)

    voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    for i,j in itertools.product(range(6),range(6)):
        k, l = voigt_notation[i]
        m, n = voigt_notation[j]
        if (i == j == 3) or (i == j == 4) or (i == j == 5) or (i == 3 and j == 4) or (i == 4 and j == 3) or (i == 3 and j == 5) or \
        (i == 5 and j == 3) or (i == 4 and j == 5) or (i == 5 and j == 4):
            voigt[i,j] = 2.* tensor3s[k,l,m,n]
        elif (i == j == 0) or (i == j == 1) or (i == j == 2) or (i == 0 and j == 1) or (i == 1 and j == 0) or (i == 0 and j == 2) or \
        (i == 2 and j == 0) or (i == 1 and j == 2) or (i == 2 and j == 1):
            voigt[i,j] = tensor3s[k,l,m,n]
        else:
            voigt[i,j] = np.sqrt(2.)* tensor3s[k,l,m,n]
    return voigt



## -----------------------------------------
## setting zero bounds based on single crystal values
## ------------------------------------------

def calc_zero_bounds(sc_c):
    p_1 = 1./3.*np.array([[1., 1., 1., 0., 0., 0.], [1., 1., 1., 0., 0., 0.], [1., 1., 1., 0., 0.,
     0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]);
    ident = np.identity(6)
    p_2 = p2(p_1,ident)
    if len(sc_c) == 3:
        c11 = sc_c.get('sc11')
        c12 = sc_c.get('sc12')
        c44 = sc_c.get('sc44')
        #set as array
        csnvn = np.array([[c11, c12, c12, 0., 0., 0.],[c12,c11,c12, 0.,0.,0.],[c12,c12,c11,0.,0.,0.],\
            [0.,0.,0.,2.*c44,0.,0.],[0.,0.,0.,0.,2.*c44,0.],[0.,0.,0.,0.,0.,2.*c44]])
    elif len(sc_c) == 5:
        c11 = sc_c.get('sc11')
        c12 = sc_c.get('sc12')
        c13 = sc_c.get('sc13')
        c33 = sc_c.get('sc33')
        c44 = sc_c.get('sc44')
        csnvn = np.array([[c11, c12, c13, 0., 0., 0.],[c12,c11,c13, 0.,0.,0.],[c13,c13,c33,0.,0.,0.],\
            [0.,0.,0.,2.*c44,0.,0.],[0.,0.,0.,0.,2.*c44,0.],[0.,0.,0.,0.,0.,2.*(c11-c12)/2.]])
    else:

        raise ValueError("This should never happen, something is wrong with the I/O of files. See calc_zero_bounds function for details")

    isouppout = isoboundupp(csnvn,p_1,ident,0)

    isoupp = np.array([(isouppout[0]).real,(isouppout[1]).real])
    inversecnvn = np.linalg.inv(csnvn)
    invuplow = isoboundupp(inversecnvn,p_1,ident,1)

    isolow = np.array([1./((invuplow[0]).real),1./((invuplow[1]).real)])


    cs = recovernormalized4th(csnvn)

    cscihighlow = csci_iso(cs)

    return isolow,isoupp, cscihighlow, cs

def csci_iso(cs):
    p_1 = p1()
    p_2 = p2(p_1,identitymat())

    def eqs(variables):
        c1,c2 = variables
        eq1 = np.array([c1,c2]) - np.array(1.0/(liso4s(p0(c1,c2,p_1,p_2)))) + np.array(1.0/(liso4s(inv336633((cs)-(isomat_fromeig(np.array([c1,c2]),p_1,p_2)-inv336633(p0(c1,c2,p_1,p_2)))))))
        print(eq1)
        eq2 = np.array([c1,c2])
        eq3=eq1-eq2
        return [eq3[0],eq3[1]]

    c1,c2 = fsolve(eqs,[500,150])
    eigvalarr = np.array([c1,c2])

    return eigvalarr


def isoboundupp(cs_arr,p1,ident,lowhigh):

    if lowhigh == 0:
        x0 = 100
        xtol = 0.1
    if lowhigh == 1:
        x0 = -0.05
        xtol = 1.e-7
    def isobound(z):
        matcalc = np.subtract(cs_arr,z*p1)
        eigval = (la.eigvals(matcalc))
        b = max(eigval)
        t = np.trace(np.add(z*p1,b*ident))
        return t.real

    res = scipy.optimize.minimize(isobound, x0, method='nelder-mead',options={'xtol': xtol})
    afinal = res.x[0]
    bfinal = max(la.eigvals(np.subtract(cs_arr,afinal*p1)))
    isoboundupp = [afinal+bfinal,bfinal]
    return isoboundupp

## -----------------------------------------
## Calculations of HS, Voigt, Reuss bounds based on texture coefficients
## ------------------------------------------
def checkfrob4th_cub(vdict):
    vtens = gen_4th_varr(vdict)
    norm = frobnormtens(vtens)
    if norm > 1.0 or norm < 0.0:
        return True
    else:
        return False

def texture_to_c_cub_run(vdict,csci_e,cs3s):

    p_1 = p1()
    ident = identitymat()
    p_2 = p2(p_1,ident)

    v4tens = gen_4th_varr(vdict)
    v4tensreal = v4tens.real

    cscimat = isomat_fromeig(csci_e,p_1,p_2)

    CSCI = CSCI_cub(cs3s,v4tensreal,cscimat,csci_e[0],csci_e[1],p_1,p_2)


    csci_voigt = voigt(CSCI)

    return csci_voigt

def texture_to_c_cub(vdict,zbl,zbh,csci_e,cs3s):

    p_1 = p1()
    ident = identitymat()
    p_2 = p2(p_1,ident)

    v4tens = gen_4th_varr(vdict)
    v4tensreal = v4tens.real

    isomatupp = isomat_fromeig(zbh,p_1,p_2)
    isomatlow = isomat_fromeig(zbl,p_1,p_2)
    cscimat = isomat_fromeig(csci_e,p_1,p_2)

    CHS_upper = CHS(cs3s,v4tensreal,isomatupp,zbh[0],zbh[1],p_1,p_2)
    CHS_lower = CHS(cs3s,v4tensreal,isomatlow,zbl[0],zbl[1],p_1,p_2)
    CSCI = CSCI_cub(cs3s,v4tensreal,cscimat,csci_e[0],csci_e[1],p_1,p_2)

    upper_HS_voigt = voigt(CHS_upper)
    lower_HS_voigt = voigt(CHS_lower)
    csci_voigt = voigt(CSCI)

    return csci_voigt, upper_HS_voigt, lower_HS_voigt

def average_HS(uppervoi,lowervoi):
    upp = uppervoi.flatten()
    low = lowervoi.flatten()
    output = np.concatenate([[upp],[low]],axis = 0)
    mean = np.mean(output, axis = 0)
    cmean = mean.reshape((6,6))
    return cmean

def CHS(cs3s,vtens,C0,zb1,zb2,p1,p2):

    input2 = inv336633(p0(zb1,zb2,p1,p2))

    input4 = inv336633(cs3s-C0+input2)

    input3 = inv336633(oafull(input4,vtens))

    HS_bound = C0 - input2 + input3
    return HS_bound
def CSCI_cub(cs3s,vtens,C0CI,e1,e2,p1,p2):
    input2 = inv336633(p0(e1,e2,p1,p2))

    input4 = inv336633(cs3s-C0CI+input2)

    input3 = inv336633(oafull(input4,vtens))

    csci = C0CI - input2 + input3

    return csci


def oafull(tensor3s,vtens):
    hi1,hi2,H21,H22,H4 = hd(tensor3s)
    ret21,ret22,ret41 = coeffhd24(H21,H22,H4)

    h21v2 = ret21 * genH2alpha(2)
    h22v2 = ret22 * genH2alpha(2)

    h4v4 = ret41 * vtens


    oa = hdrec(hi1,hi2,h21v2,h22v2,h4v4)
    return oa

def genHONBcub(rank):
    if rank == 2:
        return np.array([np.zeros((3,3),dtype=np.float64)])
    elif rank == 4:
        ret = np.array([[[[[np.sqrt(2.0/15.0),0.0,0.0],[0.0,-1.0/np.sqrt(30.0),0.0],[0.0,0.0,-1.0/np.sqrt(30.0)]],[[0.0,-1.0/np.sqrt(30.0),0.0],[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,0.0,0.0],[-1.0/np.sqrt(30.0),0.0,0.0]]],[[[0.0,-1.0/np.sqrt(30.0),0.0],[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,0.0,0.0]],[[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,np.sqrt(2.0/15.0),0.0],[0.0,0.0,-1.0/np.sqrt(30.0)]],[[0.0,0.0,0.0],[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,-1.0/np.sqrt(30.0),0.0]]],[[[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,0.0,0.0],[-1.0/np.sqrt(30.0),0.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,-1.0/np.sqrt(30.0),0.0]],[[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,-1.0/np.sqrt(30.0),0.0],[0.0,0.0,np.sqrt(2.0/15.0)]]]]])
    else:
        raise ValueError('existing codes only work with rank 2 and 4 harmonic orthonormal bases.')
    return ret

def coeffhd24(H21,H22,H4):

    arrh21 = coeffONB(H21,genHONBcub(np.ndim(H21)-1))
    arrh22 = coeffONB(H22,genHONBcub(np.ndim(H22)-1))
    arrh4 = coeffONB(H4,genHONBcub(np.ndim(H4)-1))

    return arrh21,arrh22,arrh4

def genH2alpha(scalar):

    H = np.zeros((3,3),dtype=np.float64)
    return H

def vdicttov(vdict):
    v1111 = vdict.get('v1111')
    v1112 = vdict.get('v1112')
    v1113 = vdict.get('v1113')
    v1122 = vdict.get('v1122')
    v1123 = vdict.get('v1123')
    v1222 = vdict.get('v1222')
    v1223 = vdict.get('v1223')
    v2222 = vdict.get('v2222')
    v2223 = vdict.get('v2223')
    varr = np.array(([v1111,v1112,v1113,v1122,v1123,v1222,v1223,v2222,v2223]),dtype=np.float64)
    return varr
def ctodict(cdict):

    keyes = ['c4-40','c4-30','c4-20','c4-10','c400','c410','c420','c430','c440']
    newdict = OrderedDict()
    for desired in keyes:
        newdict[desired] = cdict.get(desired)
    return newdict
def ctodictimagcub(cdict):
    newdict = OrderedDict()
    keyes = ['c400','c410','c420','c430','c440']
    for desired in keyes:
        newdict[desired] = cdict.get(desired).real
    newdict['c440imag'] = cdict.get('c440').imag
    newdict['c430imag'] = cdict.get('c430').imag
    newdict['c420imag'] = cdict.get('c420').imag
    newdict['c410imag'] = cdict.get('c410').imag
    return newdict

def fr_toarray(propdict,frnames):
    freqs = []
    for name in frnames:
        freqs.append(propdict.get(name).real)
    return np.array(freqs)

def mvtoc(vdict):

    v11 = vdict.get('v1111')
    v16 = vdict.get('v1112')
    v15 = vdict.get('v1113')
    v12 = vdict.get('v1122')
    v14 = vdict.get('v1123')
    v26 = vdict.get('v1222')
    v46 = vdict.get('v1223')
    v22 = vdict.get('v2222')
    v24 = vdict.get('v2223')
    varr = np.array(([v11,v16,v15,v12,v14,v26,v46,v22,v24]),dtype=complex)


    trans_arr = np.array([[3.0*np.sqrt(21.0)/8.0,-(3.0j*np.sqrt(21.0))/2.0,0.0,-(9.0 * np.sqrt(21.0))/4.0,0.0,(3.0j*np.sqrt(21.0))/2.0,0.0,(3.0*np.sqrt(21.0))/8.0,0.0],
        [0.0, 0.0, -((3 * np.sqrt(21/2))/2),0.0,(9.0j/2.0)*np.sqrt(21/2),0.0, (9.0*np.sqrt(21.0/2.0))/2.0,0.0,-3.0j/2.0 * np.sqrt(21.0/2.0)],
        [-(21. * np.sqrt(3.))/4.,(21.0j*np.sqrt(3.))/2.,0.0,0.0,0.0,(21.0j * np.sqrt(3.0))/2.0 ,0.0 ,(21.0 * np.sqrt(3.0))/4.0,0.0],
        [0.0,0.0,(21.0*np.sqrt(3.0/2.0))/2.0,0.0, -21.0j/2.0 * np.sqrt(3.0/2.0),0.0,(21.0 * np.sqrt(3.0/2.0))/2.0,0.0, -21.0j/2.0 * np.sqrt(3.0/2.0)],
        [(21.0 * np.sqrt(15.0/2.0))/4.0 , 0.0  , 0.0,   (21.0 * np.sqrt(15.0/2.0))/2.0 ,0.0  ,0.0, 0.0 , (21.0 * np.sqrt(15.0/2.0))/4.0 ,  0.0],
        [0.0,0.0,(21.0*np.sqrt(3.0/2.0))/2.0,0.0, 21.0j/2.0 * np.sqrt(3.0/2.0),0.0,(21.0 * np.sqrt(3.0/2.0))/2.0,0.0, 21.0j/2.0 * np.sqrt(3.0/2.0)],
        [-(21.0 * np.sqrt(3.0))/4.0,-(21.0j * np.sqrt(3.0))/2.0, 0.0, 0.0,0.0,-((21.0j * np.sqrt(3.0))/2.0), 0.0,(21.0 * np.sqrt(3.0))/4.0,0.0],
        [0.0, 0.0, -((3 * np.sqrt(21/2))/2),0.0,-(9.0j/2.0)*np.sqrt(21/2),0.0, (9.0*np.sqrt(21.0/2.0))/2.0,0.0,3.0j/2.0 * np.sqrt(21.0/2.0)],
        [(3.0*np.sqrt(21.0))/8.0,(3.0j*np.sqrt(21.0))/2.0,0.0,-((9.0*np.sqrt(21.0))/4.0),0.0,-(3.0j*np.sqrt(21.0)/2.0),0.0,(3.0*np.sqrt(21.0))/8.0,0.0]])

    c_arr = np.zeros((9),dtype=complex)

    c_arr = np.dot(trans_arr,varr)
    tex_c = dict()
    tex_c['c4-40'] = c_arr[0]
    tex_c['c4-30'] = c_arr[1]
    tex_c['c4-20'] = c_arr[2]
    tex_c['c4-10'] = c_arr[3]
    tex_c['c400'] = c_arr[4]
    tex_c['c410'] = c_arr[5]
    tex_c['c420'] = c_arr[6]
    tex_c['c430'] = c_arr[7]
    tex_c['c440'] = c_arr[8]

    return tex_c


def mctov(tex_c):

    c4bar0 = tex_c.get('c4-40')
    c3bar0 = tex_c.get('c4-30')
    c2bar0 = tex_c.get('c4-20')
    c1bar0 = tex_c.get('c4-10')
    c00 = tex_c.get('c400')
    c10 = tex_c.get('c410')
    c20 = tex_c.get('c420')
    c30 = tex_c.get('c430')
    c40 = tex_c.get('c440')

    c4arr = np.array([c4bar0,c3bar0,c2bar0,c1bar0,c00,c10,c20,c30,c40])

    trans_arr = np.array([
        [1./(6.*np.sqrt(21.)),0.0,-1./(21.*np.sqrt(3.)),0.0,1./(7.*np.sqrt(30.)),0.0,-1./(21.*np.sqrt(3.)),0.0,1./(6.*np.sqrt(21.))],
        [1.0j/(6.*np.sqrt(21.)),0.0,-1.0j/(42.*np.sqrt(3.)),0.0,0.0,0.0,1.0j/(42.*np.sqrt(3.)),0.0,-1.0j/(6.*np.sqrt(21.))],
        [0.0,-1./(6.*np.sqrt(42.)),0.0,1./(14.*np.sqrt(6.)),0.0,1./(14.*np.sqrt(6.)),0.0,-1./(6.*np.sqrt(42.)),0.0],
        [-1./(6.*np.sqrt(21.)),0.0,0.0,0.0,1./(21.*np.sqrt(30.)),0.0,0.0,0.0,-1./(6.*np.sqrt(21.))],
        [0.0,-1.0j/(6.*np.sqrt(42.)),0.0,1.0j/(42.*np.sqrt(6.)),0.0,-1.0j/(42.*np.sqrt(6.)),0.0,1.0j/(6.*np.sqrt(42.)),0.0],
        [-1.0j/(6.*np.sqrt(21.)),0.0,-1.0j/(42.*np.sqrt(3.)),0.0,0.0,0.0,1.0j/(42.*np.sqrt(3.)),0.0,1.0j/(6.*np.sqrt(21.))],
        [0.0,1.0/(6.*np.sqrt(42.)),0.0,1.0/(42.*np.sqrt(6.)),0.0,1.0/(42.*np.sqrt(6.)),0.0,1.0/(6.*np.sqrt(42.)),0.0],
        [1./(6.*np.sqrt(21.)),0.0,1./(21.*np.sqrt(3.)),0.0,1./(7.*np.sqrt(30.)),0.0,1./(21.*np.sqrt(3.)),0.0,1./(6.*np.sqrt(21.))],
        [0.0,1.0j/(6.*np.sqrt(42.)),0.0,1.0j/(14.*np.sqrt(6.)),0.0,-1.0j/(14.*np.sqrt(6.)),0.0,-1.0j/(6.*np.sqrt(42.)),0.0]])

    v_arr = np.dot(trans_arr,c4arr)
    vdictout = dict()
    vdictout['v1111'] = v_arr[0]
    vdictout['v1112'] = v_arr[1]
    vdictout['v1113'] = v_arr[2]
    vdictout['v1122'] = v_arr[3]
    vdictout['v1123'] = v_arr[4]
    vdictout['v1222'] = v_arr[5]
    vdictout['v1223'] = v_arr[6]
    vdictout['v2222'] = v_arr[7]
    vdictout['v2223'] = v_arr[8]

    return vdictout

def makeH():

    H = 1./(np.sqrt(30.))*(np.array([[2., -1., -1., 0., 0., 0.],[-1.,2.,-1.,0.,0.,0.],[-1.,-1.,2.,0.,0.,0.], \
    [0.,0.,0.,-2.,0.,0.],[0.,0.,0.,0.,-2.,0.],[0.,0.,0.,0.,0.,-2.]]))
    return H

def lm(A,B):

    temp = int(A.ndim -np.ndim(B))
    mult = 1.0
    dims = list()
    for i in range(temp):
        dims.append(A.shape[i])
    t1 = tuple(dims)
    for i in range(B.ndim):
        mult = mult * B.shape[i]
    dimadd = (int(mult),)
    shape = t1 + dimadd
    reshaped = np.reshape(A,shape)
    result = np.dot(reshaped,B.flatten())

    return result

def hd(A):
    A = np.array(A,dtype=np.float64)

    hi1, hi2 = sp.symbols('hi1 hi2',real=True)
    h2111,h2112,h2113,h2122,h2123 = sp.symbols('h2111 h2112 h2113 h2122 h2123',real=True)
    h2211,h2212,h2213,h2222,h2223 = sp.symbols('h2211 h2212 h2213 h2222 h2223',real=True)

    h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223 = sp.symbols('h41111 h41112 h41113 h41122 h41123 h41222 h41223 h42222 h42223',real=True)

    eq1 = sp.Matrix(A.reshape((81,1)))

    eq2 = hi1 * sp.Matrix(p1().reshape((81,1))) + hi2 * sp.Matrix(p2(p1(),identitymat()).reshape((81,1))) + H21(h2111,h2112,h2113,h2122,h2123) + H22(h2211,h2212,h2213,h2222,h2223) + H4sp(h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223)
    varset = [hi1,hi2,h2111,h2112,h2113,h2122,h2123,h2211,h2212,h2213,h2222,h2223,h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223]

    system_to_solve, system_rhs = linear_eq_to_matrix((eq1-eq2), varset)
    system_to_solve = np.array(system_to_solve, dtype=np.float64)
    system_rhs = np.array(system_rhs, dtype=np.float64)
    result = np.linalg.lstsq(system_to_solve, system_rhs,rcond=None)[0]

    solveout = {}
    for var_idx, i in enumerate(varset):
        solveout[str(i)] = result[var_idx]
    hi1out = solveout.get('hi1')
    hi2out = solveout.get('hi2')


    #input h2111,h2112,h2113,h2122,h2123
    H21res = H2(solveout.get('h2111'),solveout.get('h2112'),solveout.get('h2113'),solveout.get('h2122'),solveout.get('h2123'))
    #input h2211,h2212,h2213,h2222,h2223
    H22res = H2(solveout.get('h2211'),solveout.get('h2212'),solveout.get('h2213'),solveout.get('h2222'),solveout.get('h2223'))
    #h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223
    H4res = H4(solveout.get('h41111'),solveout.get('h41112'),solveout.get('h41113'),solveout.get('h41122'),solveout.get('h41123'),solveout.get('h41222'),solveout.get('h41223'),solveout.get('h42222'),solveout.get('h42223'))

    return hi1out,hi2out,H21res,H22res,H4res
def H2(h11,h12,h13,h22,h23):
    H2 = np.array(([[h11,h12,h13],[h12,h22,h23],[h13,h23,-1.0*h11 - h22]]),dtype=np.float64)
    return H2
def H21(h11,h12,h13,h22,h23):

    H2 = sp.Matrix([(6.0 * h11)/7.0,(3.0* h12)/7.0,(3.0* h13)/7.0,(3.0* h12)/7.0, h11/7.0 + h22 /7.0,h23/7.0,(3.0* h13)/7.0,h23/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),(3.0* h12)/7.0,h11/7.0+h22/7.0,h23/7.0,h11/7.0+h22/7.0,(3.0 *h12)/7.0,h13/7.0,h23/7.0,h13/7.0,h12/7.0,(3.0* h13)/7.0,h23/7.0,h11/7.0+1.0/7.0* (-1.0*h11-h22),h23/7.0,h13/7.0,h12/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h12/7.0,(3.0 * h13)/7.0,(3.0* h12)/7.0,h11/7.0+h22/7.0,h23/7.0,h11/7.0+h22/7.0,(3.0* h12)/7.0,h13/7.0,h23/7.0,h13/7.0,h12/7.0,h11/7.0+h22/7.0,(3.0 *h12)/7.0,h13/7.0,(3.0* h12)/7.0,(6.0 *h22)/7.0,(3.0* h23)/7.0,h13/7.0,(3.0 *h23)/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,h23/7.0,h13/7.0,h12/7.0,h13/7.0,(3.0 *h23)/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,h12/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,(3.0 *h23)/7.0,(3.0* h13)/7.0,h23/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h23/7.0,h13/7.0,h12/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h12/7.0,(3.0* h13)/7.0,h23/7.0,h13/7.0,h12/7.0,h13/7.0,(3.0* h23)/7.0,1.0/7.0* (-1.0*h11-h22)+h22/7.0,h12/7.0,1.0/7.0* (-1.0*h11-h22)+h22/7.0,(3.0* h23)/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h12/7.0,(3.0* h13)/7.0,h12/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,(3.0* h23)/7.0,(3.0* h13)/7.0,(3.0 *h23)/7.0,6.0/7.0* (-1.0* h11-h22)])
    return H2
def H22(h11,h12,h13,h22,h23):

    H2 = sp.Matrix([0.0,0.0,0.0,0.0,-(h11/3.0)-h22/3.0,-(h23/3.0),0.0,-(h23/3.0),-(h11/3.0)+1.0/3.0 * (h11+h22),0.0,h11/6.0+h22/6.0,h23/6.0,h11/6.0+h22/6.0,0.0,h13/6.0,h23/6.0,h13/6.0,-(h12/3.0),0.0,h23/6.0,h11/6.0+1.0/6.0* (-h11-h22),h23/6.0,-(h13/3.0),h12/6.0,h11/6.0+1.0/6.0 *(-h11-h22),h12/6.0,0.0,0.0,h11/6.0+h22/6.0,h23/6.0,h11/6.0+h22/6.0,0.0,h13/6.0,h23/6.0,h13/6.0,-(h12/3.0),-(h11/3.0)-h22/3.0,0.0,-(h13/3.0),0.0,0.0,0.0,-(h13/3.0),0.0,-(h22/3.0)+1.0/3.0* (h11+h22),-(h23/3.0),h13/6.0,h12/6.0,h13/6.0,0.0,1.0/6.0* (-h11-h22)+h22/6.0,h12/6.0,1.0/6.0 *(-h11-h22)+h22/6.0,0.0,0.0,h23/6.0,h11/6.0+1.0/6.0* (-1.0*h11-h22),h23/6.0,-(h13/3.0),h12/6.0,h11/6.0+1.0/6.0 *(-h11-h22),h12/6.0,0.0,-(h23/3.0),h13/6.0,h12/6.0,h13/6.0,0.0,1.0/6.0 *(-h11-h22)+h22/6.0,h12/6.0,1.0/6.0* (-h11-h22)+h22/6.0,0.0,-(h11/3.0)+1.0/3.0 *(h11+h22),-(h12/3.0),0.0,-(h12/3.0),-(h22/3.0)+1.0/3.0 *(h11+h22),0.0,0.0,0.0,0.0])
    return H2
def H4(h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223):

    H4 = np.array(([[[[h41111,h41112, h41113], [h41112, h41122, h41123], [h41113, h41123, -1.0*h41111 - h41122]], [[h41112, h41122, h41123], [h41122,h41222, h41223], [h41123, h41223, -1.0*h41112 - h41222]], [[h41113, h41123, -1.0*h41111 - h41122], [h41123, h41223, -1.0*h41112 - h41222], [-1.0*h41111 - h41122, -1.0*h41112 - h41222, -1.0*h41113 - h41223]]], [[[h41112, h41122, h41123], [h41122, h41222, h41223], [h41123, h41223, -1.0*h41112 - h41222]], [[h41122, h41222, h41223], [h41222, h42222, h42223], [h41223, h42223, -1.0*h41122 - h42222]], [[h41123, h41223, -1.0*h41112 - h41222], [h41223, h42223, -1.0*h41122 - h42222], [-1.0*h41112 - h41222, -h41122 - h42222, -1.0*h41123 - h42223]]], [[[h41113, h41123, -h41111 - h41122], [h41123, h41223, -1.0*h41112 - h41222], [-1.0*h41111 - h41122, -1.0*h41112 - h41222, -1.0*h41113 - h41223]], [[h41123, h41223, -1.0*h41112 - h41222], [h41223, h42223, -1.0*h41122 - h42222], [-h41112 - h41222, -h41122 - h42222, -h41123 - h42223]], [[-1.0*h41111 - h41122, -1.0*h41112 - h41222, -1.0*h41113 - h41223], [-1.0*h41112 - h41222, -1.0*h41122 - h42222, -1.0*h41123 - h42223], [-1.0*h41113 - h41223, -1.0*h41123 - h42223, h41111 + 2.0 * h41122 + h42222]]]]),dtype=np.float64)
    return H4
def H4sp(h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223):
    #flattened form
    H4 = sp.Matrix([h41111,h41112, h41113,h41112, h41122, h41123, h41113, h41123, -h41111 - h41122, h41112, h41122, h41123, h41122,h41222, h41223, h41123, h41223, -h41112 - h41222, h41113, h41123, -h41111 - h41122, h41123, h41223, -h41112 - h41222, -h41111 - h41122, -h41112 - h41222, -h41113 - h41223, h41112, h41122, h41123, h41122, h41222, h41223, h41123, h41223, -h41112 - h41222, h41122, h41222, h41223, h41222, h42222, h42223, h41223, h42223, -h41122 - h42222, h41123, h41223, -h41112 - h41222, h41223, h42223, -h41122 - h42222, -h41112 - h41222, -h41122 - h42222, -h41123 - h42223, h41113, h41123, -h41111 - h41122, h41123, h41223, -h41112 - h41222, -h41111 - h41122, -h41112 - h41222, -h41113 - h41223, h41123, h41223, -h41112 - h41222, h41223, h42223, -h41122 - h42222, -h41112 - h41222, -h41122 - h42222, -h41123 - h42223, -h41111 - h41122, -h41112 - h41222, -h41113 - h41223, -h41112 - h41222, -h41122 - h42222, -h41123 - h42223, -h41113 - h41223, -h41123 - h42223, h41111 + 2.0 * h41122 + h42222])
    return H4
def coeffONB(A,onb):
    scalarprod = sum(np.multiply(A,onb).flatten())
    return scalarprod
def hdrec(hi1,hi2,h21v2,h22v2,h4v4):

    rectens = hi1 * p1() + hi2 * p2(p1(),identitymat()) + lm(J61(),h21v2) + lm(J62(),h22v2) + h4v4
    return rectens

def frobnormtens(A):
    fn = np.sqrt(sum(np.multiply(A,A).flatten()))
    return fn
def scalarprod(A,B):
    sp = sum(np.multiply(A,B).flatten())
    return sp


def p0(c1,c2,p1,p2):

    temp1 = 1./(c1 + 2.*c2)
    temp2 = (2./(5. * c2))*(c1 + 3. * c2)/(c1 + 2. * c2)
    tempinput = np.array([temp1,temp2])
    polarization = isomat_fromeig(tempinput,p1,p2)
    #in 3x3x3x3 notation
    return polarization

def inv336633(tensor3s):

    invtensor3s = recovernormalized4th(np.linalg.inv(recovernormalizedvoigt(tensor3s)))
    return invtensor3s

def p1():
    id = np.identity(3)
    p_uno = np.array(1/3.*np.tensordot(id,id,0))
    return p_uno

def identitymat():

    ident = np.array([[[[1.,0.,0.],[0.,0.,0.],[0.,0.,0.]],[[0.,1./2.,0.],[1./2.,0.,0.],[0.,0.,0.]],[[0.,0.,1./2.],[0.,0.,0.],[1./2.,0.,0.]]],\
        [[[0.,1./2.,0.],[1./2.,0.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,1.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,1./2.],[0.,1./2.,0.]]],\
        [[[0.,0.,1./2.],[0.,0.,0.],[1./2.,0.,0.]],[[0.,0.,0.],[0.,0.,1./2.],[0.,1./2.,0.]],[[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]]])
    return ident

def p2(p1,ident):
    pdos = np.subtract(ident,p1)
    return pdos
def liso4s(tensor3s):

    p_1 = p1()
    p_2 = p2(p_1,identitymat())


    sp1 = scalarprod(tensor3s,p_1)
    sp2 = scalarprod(p_1,p_1)
    liso = sp1/sp2

    sp3 = scalarprod(tensor3s,p_2)
    sp4 = scalarprod(p_2,p_2)
    liso2 = sp3/sp4

    lisoarr = np.array([liso,liso2])
    return lisoarr

def isomat_fromeig(zblist,p_1,p_2):

    isomat3s = np.multiply(zblist[0],p_1) + np.multiply(zblist[1],p_2)

    return isomat3s

def J60(i,j,k,l,m,n):
    I2 = np.identity(3,dtype=np.float64)
    J60 = I2[i][j] * I2[k][l] * I2[m][n]
    return J60

def J61():

    jay61 = np.zeros((3,3,3,3,3,3),dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    for m in range(3):
                        for n in range(3):
                            jay61[i,j,k,l,m,n] = 1.0/7.0 * (J60(i, j, k, n, l, m) + J60(i, k, j, n, l, m) + J60(i, l, j, n, k, m) + J60(i, n, j, k, l, m) + J60(i, n, j, l, k, m) + J60(i, n, j, m, k, l))
    return jay61
def J62():
    jay62 = np.zeros((3,3,3,3,3,3),dtype=np.float64)
    for i,j,k,l,m,n in itertools.product(range(3), range(3), range(3), range(3), range(3),range(3)):
        jay62[i,j,k,l,m,n] = 1.0/6.0 * (-2.0 * J60(i,j,k,n,l,m) + J60(i,k,j,n,l,m) + J60(i,l,j,n,k,m)+ J60(i,n,j,k,l,m) + J60(i,n,j,l,k,m) - 2.0 * J60(i,n,j,m,k,l))
    return jay62

def voigtcalc(cs,vtens):
    voigt = oafull(cs,vtens)
    return voigt
def reusscalc(cs,vtens):
    oaret = oafull(inv336633(cs),vtens)
    reuss = inv336633(oaret)
    return reuss

def out9param():
    out = np.array(['c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c44', 'c55', 'c66','HS_Upper_C11', 'HS_Upper_C12', 'HS_Upper_C13', 'HS_Upper_C22', 'HS_Upper_C23', 'HS_Upper_C33', 'HS_Upper_C44', 'HS_Upper_C55', 'HS_Upper_C66','HS_Lower_C11', 'HS_Lower_C12', 'HS_Lower_C13', 'HS_Lower_C22', 'HS_Lower_C23', 'HS_Lower_C33', 'HS_Lower_C44', 'HS_Lower_C55', 'HS_Lower_C66','c400','c420','c440'])
    return out
def out21param():
    out = np.array(['c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c44', 'c55', 'c66','c14','c15','c16','c24','c25','c26','c34','c35','c36','c45','c46','c56','HS_Upper_C11', 'HS_Upper_C12', 'HS_Upper_C13', 'HS_Upper_C22', 'HS_Upper_C23', 'HS_Upper_C33', 'HS_Upper_C44', 'HS_Upper_C55', 'HS_Upper_C66','HS_Upper_C14','HS_Upper_C15','HS_Upper_C16','HS_Upper_C24','HS_Upper_C25','HS_Upper_C26','HS_Upper_C34','HS_Upper_C35','HS_Upper_C36','HS_Upper_C45','HS_Upper_C46','HS_Upper_C56','HS_Lower_C11', 'HS_Lower_C12', 'HS_Lower_C13', 'HS_Lower_C22', 'HS_Lower_C23', 'HS_Lower_C33', 'HS_Lower_C44', 'HS_Lower_C55', 'HS_Lower_C66','HS_Lower_C14','HS_Lower_C15','HS_Lower_C16','HS_Lower_C24','HS_Lower_C25','HS_Lower_C26','HS_Lower_C34','HS_Lower_C35','HS_Lower_C36','HS_Lower_C45','HS_Lower_C46','HS_Lower_C56','c4-40','c4-30','c4-20','c4-10','c400','c410','c420','c430','c440'])
    return out



def outputmatlabclmnrawdata(propagatedstepsliced,propstepwts,filename):
    clmnoutvar = np.append(propagatedstepsliced,propstepwts,axis=1)
    prefix = filename
    dictout = {'Clmn_ordered':clmnoutvar}
    scipy.io.savemat(prefix +'_clmn_weights.mat', dictout)

def varralttodict(varr):

    vdict = {}
    vdict['v1111'] = varr[0]
    vdict['v1112'] = varr[1]
    vdict['v1113'] = varr[2]
    vdict['v1122'] = varr[3]
    vdict['v1123'] = varr[4]
    vdict['v1222'] = varr[5]
    vdict['v1223'] = varr[6]
    vdict['v2222'] = varr[7]
    vdict['v2223'] = varr[8]
    return vdict

def v4_to_arr(v4tensreal):
    vv = voigt(v4tensreal)
    v11 = vv[0,0]
    v16 = vv[0,5]
    v15 = vv[0,4]
    v12 = vv[0,1]
    v14 = vv[0,3]
    v26 = vv[1,5]
    v46 = vv[3,5]
    v22 = vv[1,1]
    v24 = vv[1,3]
    varr = np.array(([v11,v16,v15,v12,v14,v26,v46,v22,v24]),dtype=np.float64)
    return varr
def mvtocarr(vdict):

    v11 = vdict.get('v1111')
    v16 = vdict.get('v1112')
    v15 = vdict.get('v1113')
    v12 = vdict.get('v1122')
    v14 = vdict.get('v1123')
    v26 = vdict.get('v1222')
    v46 = vdict.get('v1223')
    v22 = vdict.get('v2222')
    v24 = vdict.get('v2223')
    varr = np.array(([v11,v16,v15,v12,v14,v26,v46,v22,v24]),dtype=np.float64)

    trans_arr = np.array([[3.0*np.sqrt(21.0)/8.0,-(3.0j*np.sqrt(21.0))/2.0,0.0,-(9.0 * np.sqrt(21.0))/4.0,0.0,(3.0j*np.sqrt(21.0))/2.0,0.0,(3.0*np.sqrt(21.0))/8.0,0.0],
        [0.0, 0.0, -((3 * np.sqrt(21/2))/2),0.0,(9.0j/2.0)*np.sqrt(21/2),0.0, (9.0*np.sqrt(21.0/2.0))/2.0,0.0,-3.0j/2.0 * np.sqrt(21.0/2.0)],
        [-(21. * np.sqrt(3.))/4.,(21.0j*np.sqrt(3.))/2.,0.0,0.0,0.0,(21.0j * np.sqrt(3.0))/2.0 ,0.0 ,(21.0 * np.sqrt(3.0))/4.0,0.0],
        [0.0,0.0,(21.0*np.sqrt(3.0/2.0))/2.0,0.0, -21.0j/2.0 * np.sqrt(3.0/2.0),0.0,(21.0 * np.sqrt(3.0/2.0))/2.0,0.0, -21.0j/2.0 * np.sqrt(3.0/2.0)],
        [(21.0 * np.sqrt(15.0/2.0))/4.0 , 0.0  , 0.0,   (21.0 * np.sqrt(15.0/2.0))/2.0 ,0.0  ,0.0, 0.0 , (21.0 * np.sqrt(15.0/2.0))/4.0 ,  0.0],
        [0.0,0.0,(21.0*np.sqrt(3.0/2.0))/2.0,0.0, 21.0j/2.0 * np.sqrt(3.0/2.0),0.0,(21.0 * np.sqrt(3.0/2.0))/2.0,0.0, 21.0j/2.0 * np.sqrt(3.0/2.0)],
        [-(21.0 * np.sqrt(3.0))/4.0,-(21.0j * np.sqrt(3.0))/2.0, 0.0, 0.0,0.0,-((21.0j * np.sqrt(3.0))/2.0), 0.0,(21.0 * np.sqrt(3.0))/4.0,0.0],
        [0.0, 0.0, -((3 * np.sqrt(21/2))/2),0.0,-(9.0j/2.0)*np.sqrt(21/2),0.0, (9.0*np.sqrt(21.0/2.0))/2.0,0.0,3.0j/2.0 * np.sqrt(21.0/2.0)],
        [(3.0*np.sqrt(21.0))/8.0,(3.0j*np.sqrt(21.0))/2.0,0.0,-((9.0*np.sqrt(21.0))/4.0),0.0,-(3.0j*np.sqrt(21.0)/2.0),0.0,(3.0*np.sqrt(21.0))/8.0,0.0]])

    c_arr = np.zeros((9),dtype=complex)

    c_arr = np.dot(trans_arr,varr)

    return c_arr

def rotsymmnegx():
    mat = np.array([[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]])
    return mat
def rotsymmnegy():
    mat = np.array([[-1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-1.0]])
    return mat
def rotsymmnegz():
    mat = np.array([[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]])
    return mat
def rotarbit(vdict,str):
    vten4 = gen_4th_varr(vdict)
    if str == 'y':
        return rotT4th(vten4,rotsymmnegy())
    elif str == 'z':
        return rotT4th(vten4,rotsymmnegz())
    elif str == 'x':
        return rotT4th(vten4,rotsymmnegx())
def rotT4th(tens,g):
    gg = np.outer(g, g)
    gggg = np.outer(gg, gg).reshape(4 * g.shape)
    axes = ((0, 2, 4, 6), (0, 1, 2, 3))
    return np.tensordot(gggg, tens, axes)

def poly_optimize(d1,d2,d3,density,nfreq,n):
        #using random elastic constants from CoNi alloy
        print('Running adaptive determination of polynomial order.')
        cvect = rus.c_vect_create_dict({'c11':276.43 ,'c22': 270.8 ,'c33': 271.68 ,'c44': 104.83 ,'c55': 98.73 ,'c66': 98.75 ,'c12': 131.2 ,'c13': 130.3 ,'c23': 135.97 ,'c14':2.272 ,'c15':4.184 ,'c16':16.3 ,'c24':1.57927 ,'c25':4.906 ,'c26':2.251 ,'c34':0.6936 ,'c35': 0.7222 ,'c36':14.117 ,'c45':17.113 ,'c46':6.25 ,'c56':4.26})

        M_ref, K_arr_ref = rus.build_basis(n, d1,
                                             d2,d3, density)
        comparisonfreqs = rus.mech_rus(nfreq, M_ref, K_arr_ref,
                                    0.01 * cvect, n)
        for i in range(2,16,2):
            curr_poly = n - i
            M_curr, K_arr_curr = rus.build_basis(curr_poly, d1,
                                             d2,d3, density)
            freqscurr = rus.mech_rus(nfreq, M_curr, K_arr_curr,
                                    0.01 * cvect, curr_poly)
            percdiff_modewise = (freqscurr - comparisonfreqs)/comparisonfreqs * 100
            maxabspercdiff = np.amax(np.absolute(percdiff_modewise))
            #when the maximum percent difference of any single mode is larger than 0.05%, break out of loop and use the previous polynomial order.
            if maxabspercdiff > 0.05:
                ideal_polyorder = curr_poly + 2
                print('Optimized polynomial order for a maximum differential of 0.05 percent at any mode set as: ' + str(ideal_polyorder))
                break

        return ideal_polyorder
