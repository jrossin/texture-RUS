
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
from sympy_backports import linear_eq_to_matrix
from numba import jit
from collections import OrderedDict


def calc_forward_cm(cxx,ns):
    """ Calls appropriate function based on ns value. """
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
    Translate dictionary values to voigt notation for forward model
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
    Translate dictionary values to voigt notation for forward model
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
    Translate dictionary values to voigt notation for forward model
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
   Translate dictionary values to voigt notation for forward model
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
        #cxxddon for tric symmetry below
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
        #addon for tric symmetry below W
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
   Translate dictionary values to voigt notation for forward model
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
        #addon for tric symmetry below W
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
   Translate dictionary values to voigt notation for forward model
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
        #addon for tric symmetry below W
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
   Translate dictionary values to voigt notation for forward model
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
        #addon for tric symmetry below W
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
## several of these functions are translated directly and will be optimized for fitting into the SMC framework at a later date.

@jit(nopython=True)
def build_basis(poly_order,d1,d2,d3,density):
    #d1 is smallest dimension (X), d2 next Y, d3 largest dimension (z) - they are the double side lengths
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

    #perhaps define separate function to get the lookup table here
    dp = np.zeros((L*3*3 + L*L*3*3,3,3),dtype=np.float64)

    pv = np.zeros((L,L),dtype=np.float64)

    #vectors of x y and z
    #VOLUME integral here
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
            # pv is
            # initialized length of each i j iteration of the dpm vector
            #[i * 3 * 3 + j * ns.size() * 3 * 3]
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

    #potential for Dask parallelization here
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

    #print(k_arr[0])

    #ij end iteration is 21, so last call uses ij=20 for computations
    #cvect is keyed to the same

    if (len(k_arr) % 21) != 0:
       raise ValueError("k array.len() must be a multiple of 21!")

    #Ksize = len(lookup) / 21
    #left here for if lower symmetry problems are desired to be used later, redundant when Cij = 21 a given
    #lookup_ = flatten_stan(Ksize, lookup, cdict_init,L)

    #lookup_ = lookup

    return M,k_arr

@jit(nopython=True)
def buildK(poly_order, Ch, dp):
      #Ch is 6x6
      L = np.int64((poly_order + 1) * (poly_order + 2) * (poly_order + 3) / 6)

      K = np.zeros((L * 3, L * 3),dtype = np.float64)
      #voigt transforms to C 9x9 parametrization (mandel?)
      C = voigt_stan(Ch)


      for n in range(L):
        for m in range(L):
          #if n == 5 and m == 5:
          # print("dpmbuildk")

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
    #c_init = np.zeros((len(cdict)),dtype=np.float64)
    c_vect = np.array(([cdict['c11'],cdict['c12'],cdict['c22'],cdict['c13'],cdict['c23'],cdict['c33'],cdict['c14'],cdict['c24'],cdict['c34'],cdict['c44'],cdict['c15'],cdict['c25'],cdict['c35'],cdict['c45'],cdict['c55'],cdict['c16'],cdict['c26'],cdict['c36'],cdict['c46'],cdict['c56'],cdict['c66']]),dtype=np.float64)
    return c_vect

def c_vect_create_mat(cmat):
    #c_init = np.zeros((len(cmat)),dtype=np.float64)
    c_vect = np.array(([cmat[0,0],cmat[0,1],cmat[1,1],cmat[0,2],cmat[1,2],cmat[2,2],cmat[0,3],cmat[1,3],cmat[2,3],cmat[3,3],cmat[4,0],cmat[4,1],cmat[2,4],cmat[3,4],cmat[4,4],cmat[0,5],cmat[1,5],cmat[2,5],cmat[3,5],cmat[4,5],cmat[5,5]]),dtype=np.float64)
    return c_vect

def flatten_stan(Ksize,lookup, C_, L):
    lookup_ = [scipy.sparse.csc_matrix((3*L,3*L),dtype=np.float64)] * Ksize * C_.size
    #leaves space to flatten and reshape lookup if fewer Cij values are used.
    #not necessary here, unless the P,N despande framework becomes a thing
    #otherwise, simply links ij representation of each variable in C_ vectorto the position of the lookup table
    for ij in range(C_.size):
        for k in range(Ksize):
            lookup_[ij*Ksize+k] += lookup[ij * Ksize + k]

    #normally C_ is the number of independent variables in Cmat - so always 21 in our case or we'd use block diag
    #final form is a vector lookup_ of size Ksize*21 that corresponds to a unique 3*L x 3*L matrix for each unique i,j param of Cij (21 independent)
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
    #this function is the the forward calculation
    # N is number of resonance modes to calculate, lookup is the vector of length L*L*3*3*C_size (full of 3*L x 3*L arrays)
    #dfreqsdCij = np.zeros((N,len(C_)),dtype=np.float64)
    #freqs will be output as column vector
    freqs, dfreqsdCij = mechanics(C,M,K_arr,N,P)

    #if mechanics was given doubles, no need for gradients, that is only for symbolic outputs----return dfreqsdcij later if desired
    return freqs


def mechanics(C,M,K_arr,nevs,P):



    L = np.int64((P+1)*(P+2)*(P+3)/6)
    K = np.zeros((3*L,3*L),dtype=np.float64)

    for i in range(C.size):
        K += K_arr[i] * C[i]

    #evals,evecs = scipy.sparse.linalg.eigs(K, M, k = 6+nevs, which = 'LM', ncv = 12+ 2*nevs,  tol=1.0E-4, return_eigenvectors=True)
    evals,evecs = la.eigh(K, M, eigvals = (0,6+nevs-1), check_finite=True)
    #dfreqs dcij is gradients dependent on all parameters in C(K)
    dfreqsdCij = np.zeros((nevs, C.size),dtype=np.float64)

    for i in range(6):
        if (evals[i])>1E-2:
            print("Eigenvalue " + str(i) + " is " + str(evals[i]) +  "(should be near zeros for the first 6 given tralations and rotations, tolerance for this calcualtion is 1e-2)")
            raise ValueError('Exiting')

    freqstmp = np.zeros((nevs),dtype=np.float64)
    for i in range(nevs):
        freqstmp[i] = evals[i+6]
        #for j in range(evecs.rows()):
            #evectemp[j,i] = evecs[j,nevs-j-1]
    freqs = np.sqrt(freqstmp * 1.0E11) / (np.pi * 2000.0)
    return freqs, dfreqsdCij



## ----------------------------------------------------------------------------------------------
####### 9x9 VOIGT transforms from CMDSTAN RUS
## ----------------------------------------------------------------------------------------------
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
    #input voigt tensor is not in normalized form according to fernandez
    #So C44 == C2323 - no factor of 2 applied
    tensor3s = np.zeros((3,3,3,3),dtype=np.float64)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        v_i = help_3x3(i, j)
        v_j = help_3x3(k, l)
        tensor3s[i, j, k, l] = voigt[v_i, v_j]
    return tensor3s
def recover4ordercomplex(voigt):
    #input voigt tensor is not in normalized form according to fernandez
    #So C44 == C2323 - no factor of 2 applied
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

## ----------------------------------------------------------------------------------------------------------------
## these functions deal iwth the normalized voigt (mandel notation transformations) to remain consistent within the fernandez HS framework
## ----------------------------------------------------------------------------------------------------------------
##hex functions
def gen_hex_varr(vdict):
    v2tens = gen_2nd_varr(vdict)
    v4tens = gen_4th_varr(vdict)
    return v2tens, v4tens

def gen_2nd_varr(vdict):
    v = np.zeros((3,3),dtype=np.float64)
    v[0,0] = vdict.get('v211')
    v[0,1] = v[1,0] = vdict.get('v212')
    v[0,2] = v[2,0] = vdict.get('v213')
    v[1,1] = vdict.get('v222')
    v[1,2] = v[2,1] = vdict.get('v223')
    v[2,2] = -vdict.get('v211') - vdict.get('v222')
    return v

    ## CUBIC functions (only)
def gen_4th_varr(vdict):
    #generate relations between V indices and transform into fernandez normalized 4th order V[4,1][alpha,beta]
    v = np.zeros((6,6),dtype=np.float64)
    #voigt index notation for V defined with the sqrt(2) and factors of 2 here, then translated to 4th order to add them

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
    #regular 4th order transform here (non nvn normalized in mathematica codes)
    fourthorder = recover4order(v)

    return fourthorder


def recovernormalized4th(sixsix):
    #relation turn a normalized voigt 6x6 back into its 3x3x3x3 parametrization relative to the fernandez framework normalized nvnrec function
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
    #takes tensor of order 3x3x3x3 and transforms it to voigt notation (normalized), also known as mandel notation. Corresponds to nvn function in fernandez HS framework
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
    #input is single crystal in non-normalized voigt notation - nvn form used here then transformed to 4th order correctly later

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
    #eig 1 and eig 3 in upper bound
    isoupp = np.array([(isouppout[0]).real,(isouppout[1]).real])
    inversecnvn = np.linalg.inv(csnvn)
    invuplow = isoboundupp(inversecnvn,p_1,ident,1)

    isolow = np.array([1./((invuplow[0]).real),1./((invuplow[1]).real)])
    #eig1x and eig 2x in lower bound

    cs = recovernormalized4th(csnvn)
    #using self consistent solution as output elastic constants
    cscihighlow = csci_iso(cs)

    return isolow,isoupp, cscihighlow, cs

def csci_iso(cs):
    p_1 = p1()
    p_2 = p2(p_1,identitymat())
    #CHSLisofunction
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
    #initial guesses for the optimization
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
    #shortened function to only calculate csci bounds during inversion
    p_1 = p1()
    ident = identitymat()
    p_2 = p2(p_1,ident)
    #capability to get 1st order bounds (voigt reuss) as well, but wont bother with computation expense as 2nd are better
    #convert vdict to 4th order vtensor
    v4tens = gen_4th_varr(vdict)
    v4tensreal = v4tens.real

    cscimat = isomat_fromeig(csci_e,p_1,p_2)

    CSCI = CSCI_cub(cs3s,v4tensreal,cscimat,csci_e[0],csci_e[1],p_1,p_2)
    #all tensors in 3x3x3x3 notation for input
    #regular voigt notation for 6x6 output

    csci_voigt = voigt(CSCI)

    return csci_voigt

def texture_to_c_hex_run(vdict,csci_e,cs3s):
    #shortened function to only calculate csci bounds during inversion
    p_1 = p1()
    ident = identitymat()
    p_2 = p2(p_1,ident)
    #capability to get 1st order bounds (voigt reuss) as well, but wont bother with computation expense as 2nd are better

    #convert vdict to 2nd and 4th order vtensor
    v2tens, v4tens = gen_hex_varr(vdict)
    v2tensreal = v2tens.real
    v4tensreal = v4tens.real

    cscimat = isomat_fromeig(csci_e,p_1,p_2)

    CSCI = CSCI_hex(cs3s,v2tensreal,v4tensreal,cscimat,csci_e[0],csci_e[1],p_1,p_2)
    csci_voigt = voigt(CSCI)
    #all tensors in 3x3x3x3 notation
    #changed to voigt rather than normalized voigt for output numerics

    return csci_voigt

def texture_to_c_cub(vdict,zbl,zbh,csci_e,cs3s):
    #zb is zeroth order bounds
    #all cs inputs are in nvn notation (normalized) - conversions to and from 3x3x3x3 use normalized function recovernormalized4th()

    p_1 = p1()
    ident = identitymat()
    p_2 = p2(p_1,ident)
    #we have the capability to get 1st order bounds (voigt reuss) as well, but wont bother with computation expense

    #convert vdict to 4th order vtensor
    v4tens = gen_4th_varr(vdict)
    v4tensreal = v4tens.real

    #isotropic matrices of zeroth order bounds
    isomatupp = isomat_fromeig(zbh,p_1,p_2)
    isomatlow = isomat_fromeig(zbl,p_1,p_2)
    cscimat = isomat_fromeig(csci_e,p_1,p_2)

    CHS_upper = CHS(cs3s,v4tensreal,isomatupp,zbh[0],zbh[1],p_1,p_2)
    CHS_lower = CHS(cs3s,v4tensreal,isomatlow,zbl[0],zbl[1],p_1,p_2)
    CSCI = CSCI_cub(cs3s,v4tensreal,cscimat,csci_e[0],csci_e[1],p_1,p_2)
    #all tensors in 3x3x3x3 notation
    #changed to voigt rather than normalized voigt for output numerics
    upper_HS_voigt = voigt(CHS_upper)
    lower_HS_voigt = voigt(CHS_lower)
    csci_voigt = voigt(CSCI)



    return csci_voigt, upper_HS_voigt, lower_HS_voigt

def texture_to_c_hex(vdict,zbl,zbh,csci_e,cs3s):
    #zb is zeroth order bounds
    #all inputs are in voigt notation (unnormalized) - conversions to and from 3x3x3x3 use normalized nvn notation after initial transfer
    p_1 = p1()
    ident = identitymat()
    p_2 = p2(p_1,ident)
    #we have the capability to get 1st order bounds (voigt reuss) as well, but wont bother with computation expense

    #convert vdict to 4th order vtensor
    v2tens, v4tens = gen_hex_varr(vdict)
    v2tensreal = v2tens.real
    v4tensreal = v4tens.real


    #isotropic matrices of zeroth order bounds
    isomatupp = isomat_fromeig(zbh,p_1,p_2)
    isomatlow = isomat_fromeig(zbl,p_1,p_2)
    cscimat = isomat_fromeig(csci_e,p_1,p_2)

    CHS_upper = CHS_hex(cs3s,v2tensreal,v4tensreal,isomatupp,zbh[0],zbh[1],p_1,p_2)
    upper_HS_voigt = voigt(CHS_upper)

    CHS_lower = CHS_hex(cs3s,v2tensreal,v4tensreal,isomatlow,zbl[0],zbl[1],p_1,p_2)
    lower_HS_voigt = voigt(CHS_lower)

    CSCI = CSCI_hex(cs3s,v2tensreal,v4tensreal,cscimat,csci_e[0],csci_e[1],p_1,p_2)
    csci_voigt = voigt(CSCI)
    #all tensors in 3x3x3x3 notation
    #changed to voigt rather than normalized voigt for output numerics



    return csci_voigt, upper_HS_voigt, lower_HS_voigt

def average_HS(uppervoi,lowervoi):
    upp = uppervoi.flatten()
    low = lowervoi.flatten()
    output = np.concatenate([[upp],[low]],axis = 0)
    mean = np.mean(output, axis = 0)
    cmean = mean.reshape((6,6))
    return cmean

def CHS(cs3s,vtens,C0,zb1,zb2,p1,p2):
    #cubic microsymmetry


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

def CHS_hex(cs3s,vtens2,vtens4,C0,zb1,zb2,p1,p2):

    input2 = inv336633(p0(zb1,zb2,p1,p2))

    input4 = inv336633(cs3s-C0+input2)

    input3 = inv336633(oafullhex(input4,vtens2,vtens4))

    HS_bound = C0 - input2 + input3

    return HS_bound
def CSCI_hex(cs3s,vtens2,vtens4,C0CI,e1,e2,p1,p2):
    input2 = inv336633(p0(e1,e2,p1,p2))

    input4 = inv336633(cs3s-C0CI+input2)

    input3 = inv336633(oafullhex(input4,vtens2,vtens4))

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
def oafullhex(tensor3s,vtens2,vtens4):
    hi1,hi2,H21,H22,H4 = hd(tensor3s)

    ret21,ret22,ret41 = coeffhd24hex(H21,H22,H4)

    h21v2 = ret21 * vtens2
    h22v2 = ret22 * vtens2

    h4v4 = ret41 * vtens4


    oa = hdrec(hi1,hi2,h21v2,h22v2,h4v4)
    return oa
def genHONBhex(rank):


    if rank == 2:
        ret = np.array([[[1.0/np.sqrt(6.0), 0.0, 0.0], [0.0, 1.0/np.sqrt(6.0), 0.0], [0.0, 0.0, -np.sqrt(2.0/3.0)]]])
    elif rank == 4:
        ret = np.array([[[[[3.0/(2.0*np.sqrt(70.0)),0.0,0.0],[0.0,1.0/(2.0 *np.sqrt(70.0)),0.0],[0.0,0.0,-np.sqrt(2.0/35.0)]],[[0.0,1.0/(2.0 *np.sqrt(70.0)),0.0],[1.0/(2.0 *np.sqrt(70.0)),0.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,-np.sqrt(2.0/35.0)],[0.0,0.0,0.0],[-np.sqrt(2.0/35.0),0.0,0.0]]],[[[0.0,1.0/(2.0 *np.sqrt(70.0)),0.0],[1.0/(2.0 *np.sqrt(70.0)),0.0,0.0],[0.0,0.0,0.0]],[[1.0/(2.0 *np.sqrt(70.0)),0.0,0.0],[0.0,3.0/(2.0 *np.sqrt(70.0)),0.0],[0.0,0.0,-np.sqrt(2.0/35.0)]],[[0.0,0.0,0.0],[0.0,0.0,-np.sqrt(2.0/35.0)],[0.0,-np.sqrt(2.0/35.0),0.0]]],[[[0.0,0.0,-np.sqrt(2.0/35.0)],[0.0,0.0,0.0],[-np.sqrt(2.0/35.0),0.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,-np.sqrt(2.0/35.0)],[0.0,-np.sqrt(2.0/35.0),0.0]],[[-np.sqrt(2.0/35.0),0.0,0.0],[0.0,-np.sqrt(2.0/35.0),0.0],[0.0,0.0,2.0*np.sqrt(2.0/35.0)]]]]])
    else:
        raise ValueError('existing codes only work with rank 2 and 4 harmonic orthonormal bases.')
    return ret
def genHONBcub(rank):


    if rank == 2:
        return np.array([np.zeros((3,3),dtype=np.float64)])
    elif rank == 4:
        ret = np.array([[[[[np.sqrt(2.0/15.0),0.0,0.0],[0.0,-1.0/np.sqrt(30.0),0.0],[0.0,0.0,-1.0/np.sqrt(30.0)]],[[0.0,-1.0/np.sqrt(30.0),0.0],[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,0.0,0.0]],[[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,0.0,0.0],[-1.0/np.sqrt(30.0),0.0,0.0]]],[[[0.0,-1.0/np.sqrt(30.0),0.0],[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,0.0,0.0]],[[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,np.sqrt(2.0/15.0),0.0],[0.0,0.0,-1.0/np.sqrt(30.0)]],[[0.0,0.0,0.0],[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,-1.0/np.sqrt(30.0),0.0]]],[[[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,0.0,0.0],[-1.0/np.sqrt(30.0),0.0,0.0]],[[0.0,0.0,0.0],[0.0,0.0,-1.0/np.sqrt(30.0)],[0.0,-1.0/np.sqrt(30.0),0.0]],[[-1.0/np.sqrt(30.0),0.0,0.0],[0.0,-1.0/np.sqrt(30.0),0.0],[0.0,0.0,np.sqrt(2.0/15.0)]]]]])
    else:
        raise ValueError('existing codes only work with rank 2 and 4 harmonic orthonormal bases.')
    return ret

def coeffhd24(H21,H22,H4):
    #cubic microsymmetry

    arrh21 = coeffONB(H21,genHONBcub(np.ndim(H21)-1))
    arrh22 = coeffONB(H22,genHONBcub(np.ndim(H22)-1))
    arrh4 = coeffONB(H4,genHONBcub(np.ndim(H4)-1))

    #array of elements 3-5 of the harmonic decomposition of a 4th rank tensor(elastic)
    #elements are the tensors H21,H22, and H4 respectively for 2: ---- each of different length


    return arrh21,arrh22,arrh4
def coeffhd24hex(H21,H22,H4):
    #hexagonal symmetry

    arrh21 = coeffONB(H21,genHONBhex(np.ndim(H21)-1))
    arrh22 = coeffONB(H22,genHONBhex(np.ndim(H22)-1))
    arrh4 = coeffONB(H4,genHONBhex(np.ndim(H4)-1))

    #array of elements 3-5 of the harmonic decomposition of a 4th rank tensor(elastic)
    #elements are the tensors H21,H22, and H4 respectively for 2: ---- each of different length


    return arrh21,arrh22,arrh4

def genH2alpha(scalar):
    #micro symm cubic zeros the 2nd order component
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
    #cubic microsymmetry
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
def ctodicthex(cdict):
    keyes = ['c2-20','c2-10','c200','c210','c220','c4-40','c4-30','c4-20','c4-10','c400','c410','c420','c430','c440']
    newdict = OrderedDict()
    for desired in keyes:
        newdict[desired] = cdict.get(desired)
    return newdict
def ctodictheximag(cdict):
    keyes = ['c200','c210','c220','c400','c410','c420','c430','c440']
    newdict = OrderedDict()
    for desired in keyes:
        newdict[desired] = cdict.get(desired)
    newdict['c220imag'] = cdict.get('c220').imag
    newdict['c210imag'] = cdict.get('c210').imag
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
    #cubic microsymm, triclinic macrosymm
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

    #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
    #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
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

def mvtoc_hex(vdict):
    #hex microsymm, triclinic macrosymm
    #2nd order
    v211 = vdict.get('v211')
    v212 = vdict.get('v212')
    v213 = vdict.get('v213')
    v222 = vdict.get('v222')
    v223 = vdict.get('v223')
    #4th order

    v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233 =hex_translate_correct_v4val(vdict)

    varr = np.array(([v211,v212,v213,v222,v223,v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233]),dtype=np.float64)

    #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
    #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
    trans_arr = np.array([[-1.0*(81.0/10.0), (81.0j)/5.0, 0.0, 81.0/10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0* (81.0/5.0), 0.0, -(81.0j)/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [( 81.0* np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, (81.0 * np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,1.0* (81.0/5.0),0.0,(81.0j)/5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-81.0/10.0,-((81.0j)/5.0),0.0,81.0/10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,18,-9.0j,0.0,0.0,63.0/4.0,9.0j,0.0,0.0,-1.0*(9.0/4.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*9.0/np.sqrt(2.0),1.0*(27.0j)/(np.sqrt(2.0)),0.0,0.0,27.0/(np.sqrt(2.0)),-9.0j/(np.sqrt(2.0)),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),-9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),-9.0j*np.sqrt(7.0/2.0),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0),0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,-9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,-9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*(9.0/np.sqrt(2.0)),-1.0*((27.0j)/np.sqrt(2.0)),0.0,0.0,27.0/np.sqrt(2.0),(9.0j)/np.sqrt(2.0),0.0],[0.0,0.0,0.0,0.0,0.0,18.0,9.0j,0.0,0.0,63.0/4.0,-9.0j,0.0,0.0,-1.0*(9.0/4.0)]])

    c_arr = np.zeros((14),dtype=complex)

    c_arr = np.dot(trans_arr,varr)
    tex_c = dict()
    tex_c['c2-20'] = c_arr[0]
    tex_c['c2-10'] = c_arr[1]
    tex_c['c200'] = c_arr[2]
    tex_c['c210'] = c_arr[3]
    tex_c['c220'] = c_arr[4]
    tex_c['c4-40'] = c_arr[5]
    tex_c['c4-30'] = c_arr[6]
    tex_c['c4-20'] = c_arr[7]
    tex_c['c4-10'] = c_arr[8]
    tex_c['c400'] = c_arr[9]
    tex_c['c410'] = c_arr[10]
    tex_c['c420'] = c_arr[11]
    tex_c['c430'] = c_arr[12]
    tex_c['c440'] = c_arr[13]

    return tex_c
def hex_translate_correct_v4val(vdict):
    #simple conversion to alternate v tensor components for use with mathematica 12 implementation (Fernandez et al. 2019) of mvtoC code for hexagonal microsymmetry materials
    #keylist = ['v1111','v1112','v1113','v1122','v1123','v1222','v1223','v2222','v2223']
    #vdictalt = {yourkey:vdict.get(yourkey) for yourkey in keylist}
    vv = voigt(gen_4th_varr(vdict))

    v41111 = vv[0,0]
    v41112 = vv[0,5]
    v41113 = vv[0,4]
    v41123 = vv[0,3]
    v41133 = vv[0,2]
    v41222 = vv[1,5]
    v41223 = vv[3,5]
    v42223 = vv[1,3]
    v42233 = vv[1,2]

    return v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233

def hex_untranslate_correct_v4val(v):

    #corresponds to varr[5:]
    #keylist = [v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233]
    #vdictalt = {yourkey:vdict.get(yourkey) for yourkey in keylist}

    v41111 = v[5]
    v41112 = v[6]
    v41113 = v[7]
    v41122 = -v[5]-v[9] #-(v1111+v1133)
    v41123 = v[8]
    v41222 = v[10]
    v41223 = v[11]
    v42222 = v[5] + v[9] - v[13] #v1111+v1133-v2233
    v42223 = v[12]

    vdict = dict()
    vdict['v211'] = v[0]
    vdict['v212'] = v[1]
    vdict['v213'] = v[2]
    vdict['v222'] = v[3]
    vdict['v223'] = v[4]
    vdict['v1111'] = v41111
    vdict['v1112'] = v41112
    vdict['v1113'] = v41113
    vdict['v1122'] = v41122
    vdict['v1123'] = v41123
    vdict['v1222'] = v41222
    vdict['v1223'] = v41223
    vdict['v2222'] = v42222
    vdict['v2223'] = v42223

    return vdict
def mctov(tex_c):
    #cubic micro, triclinic macro
    #updated to fix complex conjugate interpolation
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
    #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
    #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
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
def mctov_hex(tex_c):
    c2bar0 = tex_c.get('c2-20')
    c1bar0 = tex_c.get('c2-10')
    c00 = tex_c.get('c200')
    c10 = tex_c.get('c210')
    c20 = tex_c.get('c220')
    c44bar0 = tex_c.get('c4-40')
    c43bar0 = tex_c.get('c4-30')
    c42bar0 = tex_c.get('c4-20')
    c41bar0 = tex_c.get('c4-10')
    c400 = tex_c.get('c400')
    c410 = tex_c.get('c410')
    c420 = tex_c.get('c420')
    c430 = tex_c.get('c430')
    c440 = tex_c.get('c440')

    c_arr = np.array(([c2bar0,c1bar0,c00,c10,c20,c44bar0,c43bar0,c42bar0,c41bar0,c400,c410,c420,c430,c440]),dtype=complex)

    trans_arr = np.array([[-1.0*(81.0/10.0), (81.0j)/5.0, 0.0, 81.0/10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0* (81.0/5.0), 0.0, -(81.0j)/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [( 81.0* np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, (81.0 * np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,1.0* (81.0/5.0),0.0,(81.0j)/5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-81.0/10.0,-((81.0j)/5.0),0.0,81.0/10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,18,-9.0j,0.0,0.0,63.0/4.0,9.0j,0.0,0.0,-1.0*(9.0/4.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*9.0/np.sqrt(2.0),1.0*(27.0j)/(np.sqrt(2.0)),0.0,0.0,27.0/(np.sqrt(2.0)),-9.0j/(np.sqrt(2.0)),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),-9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),-9.0j*np.sqrt(7.0/2.0),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0),0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,-9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,-9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*(9.0/np.sqrt(2.0)),-1.0*((27.0j)/np.sqrt(2.0)),0.0,0.0,27.0/np.sqrt(2.0),(9.0j)/np.sqrt(2.0),0.0],[0.0,0.0,0.0,0.0,0.0,18.0,9.0j,0.0,0.0,63.0/4.0,-9.0j,0.0,0.0,-1.0*(9.0/4.0)]])
    invtrans = np.linalg.inv(trans_arr)

    v_arr = np.dot(invtrans,c4arr)
    # ordering v211,v212,v213,v222,v223,v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233

    varralt = hex_untranslate_correct_v4val(v_arr)
    vdictout = varralttodict_hex(varralt)

    return vdictout
def makeH():

    H = 1./(np.sqrt(30.))*(np.array([[2., -1., -1., 0., 0., 0.],[-1.,2.,-1.,0.,0.,0.],[-1.,-1.,2.,0.,0.,0.], \
    [0.,0.,0.,-2.,0.,0.],[0.,0.,0.,0.,-2.,0.],[0.,0.,0.,0.,0.,-2.]]))
    return H

def lm(A,B):
    #linear mapping of A to B
    #reshape A and then flatten it to have dimensions of A(1-1+trB),Dimensions n of B * itself * itself...
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
    #takes tensor A and spits out constants hi1,hi2,H21,H22,H4 - each of different lengths/sizes
    hi1, hi2 = sp.symbols('hi1 hi2',real=True)
    h2111,h2112,h2113,h2122,h2123 = sp.symbols('h2111 h2112 h2113 h2122 h2123',real=True)
    h2211,h2212,h2213,h2222,h2223 = sp.symbols('h2211 h2212 h2213 h2222 h2223',real=True)
    #set to 0 by definition of cubic micro symmetry #lm(J61(),H21) #lm(J62(),H22) not included in solve - substituted by H21,H22 which evaluate to the same tensor 3x3
    h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223 = sp.symbols('h41111 h41112 h41113 h41122 h41123 h41222 h41223 h42222 h42223',real=True)
    #convert sympy to numpy  J=np.array(J_mat).astype(np.float64)

    #hi1 * p1() + hi2 * p2(p1(),identitymat()) + lm(J61(),H21) + lm(J62(),H22) + H4

    eq1 = sp.Matrix(A.reshape((81,1)))

    eq2 = hi1 * sp.Matrix(p1().reshape((81,1))) + hi2 * sp.Matrix(p2(p1(),identitymat()).reshape((81,1))) + H21(h2111,h2112,h2113,h2122,h2123) + H22(h2211,h2212,h2213,h2222,h2223) + H4sp(h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223)
    varset = [hi1,hi2,h2111,h2112,h2113,h2122,h2123,h2211,h2212,h2213,h2222,h2223,h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223]
    #increasing precision should get the H21 and H22 to evaluate as nonzero(e-14 ish)
    system_to_solve, system_rhs = linear_eq_to_matrix((eq1-eq2), varset)
    system_to_solve = np.array(system_to_solve, dtype=np.float64)
    system_rhs = np.array(system_rhs, dtype=np.float64)
    result = np.linalg.lstsq(system_to_solve, system_rhs,rcond=None)[0]

    #hi1,hi2,H21,H22,H4
    #py3 use list(results.values()) or direct dict indexing

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
    #lm(J61(),H21) hard coded
    H2 = sp.Matrix([(6.0 * h11)/7.0,(3.0* h12)/7.0,(3.0* h13)/7.0,(3.0* h12)/7.0, h11/7.0 + h22 /7.0,h23/7.0,(3.0* h13)/7.0,h23/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),(3.0* h12)/7.0,h11/7.0+h22/7.0,h23/7.0,h11/7.0+h22/7.0,(3.0 *h12)/7.0,h13/7.0,h23/7.0,h13/7.0,h12/7.0,(3.0* h13)/7.0,h23/7.0,h11/7.0+1.0/7.0* (-1.0*h11-h22),h23/7.0,h13/7.0,h12/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h12/7.0,(3.0 * h13)/7.0,(3.0* h12)/7.0,h11/7.0+h22/7.0,h23/7.0,h11/7.0+h22/7.0,(3.0* h12)/7.0,h13/7.0,h23/7.0,h13/7.0,h12/7.0,h11/7.0+h22/7.0,(3.0 *h12)/7.0,h13/7.0,(3.0* h12)/7.0,(6.0 *h22)/7.0,(3.0* h23)/7.0,h13/7.0,(3.0 *h23)/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,h23/7.0,h13/7.0,h12/7.0,h13/7.0,(3.0 *h23)/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,h12/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,(3.0 *h23)/7.0,(3.0* h13)/7.0,h23/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h23/7.0,h13/7.0,h12/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h12/7.0,(3.0* h13)/7.0,h23/7.0,h13/7.0,h12/7.0,h13/7.0,(3.0* h23)/7.0,1.0/7.0* (-1.0*h11-h22)+h22/7.0,h12/7.0,1.0/7.0* (-1.0*h11-h22)+h22/7.0,(3.0* h23)/7.0,h11/7.0+1.0/7.0 *(-1.0*h11-h22),h12/7.0,(3.0* h13)/7.0,h12/7.0,1.0/7.0 *(-1.0*h11-h22)+h22/7.0,(3.0* h23)/7.0,(3.0* h13)/7.0,(3.0 *h23)/7.0,6.0/7.0* (-1.0* h11-h22)])
    return H2
def H22(h11,h12,h13,h22,h23):
    #lm(J62(),H22) hard coded
    H2 = sp.Matrix([0.0,0.0,0.0,0.0,-(h11/3.0)-h22/3.0,-(h23/3.0),0.0,-(h23/3.0),-(h11/3.0)+1.0/3.0 * (h11+h22),0.0,h11/6.0+h22/6.0,h23/6.0,h11/6.0+h22/6.0,0.0,h13/6.0,h23/6.0,h13/6.0,-(h12/3.0),0.0,h23/6.0,h11/6.0+1.0/6.0* (-h11-h22),h23/6.0,-(h13/3.0),h12/6.0,h11/6.0+1.0/6.0 *(-h11-h22),h12/6.0,0.0,0.0,h11/6.0+h22/6.0,h23/6.0,h11/6.0+h22/6.0,0.0,h13/6.0,h23/6.0,h13/6.0,-(h12/3.0),-(h11/3.0)-h22/3.0,0.0,-(h13/3.0),0.0,0.0,0.0,-(h13/3.0),0.0,-(h22/3.0)+1.0/3.0* (h11+h22),-(h23/3.0),h13/6.0,h12/6.0,h13/6.0,0.0,1.0/6.0* (-h11-h22)+h22/6.0,h12/6.0,1.0/6.0 *(-h11-h22)+h22/6.0,0.0,0.0,h23/6.0,h11/6.0+1.0/6.0* (-1.0*h11-h22),h23/6.0,-(h13/3.0),h12/6.0,h11/6.0+1.0/6.0 *(-h11-h22),h12/6.0,0.0,-(h23/3.0),h13/6.0,h12/6.0,h13/6.0,0.0,1.0/6.0 *(-h11-h22)+h22/6.0,h12/6.0,1.0/6.0* (-h11-h22)+h22/6.0,0.0,-(h11/3.0)+1.0/3.0 *(h11+h22),-(h12/3.0),0.0,-(h12/3.0),-(h22/3.0)+1.0/3.0 *(h11+h22),0.0,0.0,0.0,0.0])
    return H2
def H4(h41111,h41112,h41113,h41122,h41123,h41222,h41223,h42222,h42223):
    #true 4th order form
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
    #hd[1] is hd of 4th order tensor, hd[2] is coeff24 of hd[1], hdarr[3-5] are various sized arrays

    rectens = hi1 * p1() + hi2 * p2(p1(),identitymat()) + lm(J61(),h21v2) + lm(J62(),h22v2) + h4v4
    return rectens
#### --------------------------------------------------------------
####space groups for triclinic and cubic symmetries, precomputed
#### --------------------------------------------------------------
def frobnormtens(A):
    fn = np.sqrt(sum(np.multiply(A,A).flatten()))
    return fn
def scalarprod(A,B):
    sp = sum(np.multiply(A,B).flatten())
    return sp

def sgtric():
    tenstric = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    return tenstric
def sghex():
    tenshex = np.array([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],[[1./2.,-np.sqrt(3.0)/2.0,0.0],[np.sqrt(3.0)/2.0,1.0/2.0,0.0],[0.0,0.0,1.0]],[[-1.0/2.0,-np.sqrt(3.0)/2.0,0.0],[np.sqrt(3.0)/2.0,-1.0/2.0,0.0],[0.0,0.0,1.0]],[[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]],[[-1.0/2.0,np.sqrt(3.0)/2.0,0.0],[-np.sqrt(3.0)/2.0,-1.0/2.0,0.0],[0.0,0.0,1.0]],[[1.0/2.0,np.sqrt(3.0)/2.0,0.0],[-np.sqrt(3.0)/2.0,1.0/2.0,0.0],[0.0,0.0,1.0]]])
    return tenshex
def sgcub():
    tenscub = np.array([[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,1.0,0.0]],[[0.0,0.0,1.0],[0.0,1.0,0.0],[-1.0,0.0,0.0]],
    [[0.0,-1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,-1.0]],[[-1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-1.0]],
    [[-1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]],[[1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,-1.0,0.0]],[[0.0,0.0,-1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]],
    [[0.0,1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,1.0]],[[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,-1.0]],[[0.0,0.0,1.0],[0.0,-1.0,0.0],[1.0,0.0,0.0]],
    [[-1.0,0.0,0.0],[0.0,0.0,1.0],[0.0,1.0,0.0]],[[0.0,-1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,-1.0]],[[0.0,0.0,-1.0],[0.0,-1.0,0.0],[-1.0,0.0,0.0]],
    [[-1.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,-1.0,0.0]],[[0.0,0.0,1.0],[1.0,0.0,0.0],[0.0,1.0,0.0]],[[0.0,1.0,0.0],[0.0,0.0,-1.0],[-1.0,0.0,0.0]],
    [[0.0,-1.0,0.0],[0.0,0.0,-1.0],[1.0,0.0,0.0]],[[0.0,0.0,-1.0],[-1.0,0.0,0.0],[0.0,1.0,0.0]],[[0.0,-1.0,0.0],[0.0,0.0,1.0],[-1.0,0.0,0.0]],
    [[0.0,0.0,1.0],[-1.0,0.0,0.0],[0.0,-1.0,0.0]],[[0.0,0.0,-1.0],[1.0,0.0,0.0],[0.0,-1.0,0.0]],[[0.0,1.0,0.0],[0.0,0.0,1.0],[1.0,0.0,0.0]]])
    return tenscub


def p0(c1,c2,p1,p2):
    #polarization tensor calculation
    #input has already calculated eigvals
    temp1 = 1./(c1 + 2.*c2)
    temp2 = (2./(5. * c2))*(c1 + 3. * c2)/(c1 + 2. * c2)
    tempinput = np.array([temp1,temp2])
    polarization = isomat_fromeig(tempinput,p1,p2)
    #in 3x3x3x3 notation
    return polarization

def inv336633(tensor3s):
    #transforms 3x3x3x3 tensors to voigt notation to get their inverse more succinctly
    #important that normalized nvn notation from fernandez is utilized here
    invtensor3s = recovernormalized4th(np.linalg.inv(recovernormalizedvoigt(tensor3s)))
    return invtensor3s

def p1():
    id = np.identity(3)
    p_uno = np.array(1/3.*np.tensordot(id,id,0))
    return p_uno

def identitymat():
    #identity matrix IS (4th order)

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
    #listed as iso4s (fernandez)
    #takes in iso eigenvalues and results in iso matrix isomat in 3x3x3x3 notation
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
def out21paramhex():
    out = np.array(['c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c44', 'c55', 'c66','c14','c15','c16','c24','c25','c26','c34','c35','c36','c45','c46','c56','HS_Upper_C11', 'HS_Upper_C12', 'HS_Upper_C13', 'HS_Upper_C22', 'HS_Upper_C23', 'HS_Upper_C33', 'HS_Upper_C44', 'HS_Upper_C55', 'HS_Upper_C66','HS_Upper_C14','HS_Upper_C15','HS_Upper_C16','HS_Upper_C24','HS_Upper_C25','HS_Upper_C26','HS_Upper_C34','HS_Upper_C35','HS_Upper_C36','HS_Upper_C45','HS_Upper_C46','HS_Upper_C56','HS_Lower_C11', 'HS_Lower_C12', 'HS_Lower_C13', 'HS_Lower_C22', 'HS_Lower_C23', 'HS_Lower_C33', 'HS_Lower_C44', 'HS_Lower_C55', 'HS_Lower_C66','HS_Lower_C14','HS_Lower_C15','HS_Lower_C16','HS_Lower_C24','HS_Lower_C25','HS_Lower_C26','HS_Lower_C34','HS_Lower_C35','HS_Lower_C36','HS_Lower_C45','HS_Lower_C46','HS_Lower_C56','c4-40','c4-30','c4-20','c4-10','c400','c410','c420','c430','c440','c2-20','c2-10','c200','c210','c220'])
    return out


def outputmatlabclmnrawdata(propagatedstepsliced,propstepwts,filename):
    clmnoutvar = np.append(propagatedstepsliced,propstepwts,axis=1)
    prefix = filename
    dictout = {'Clmn_ordered':clmnoutvar}
    scipy.io.savemat(prefix +'_clmn_weights.mat', dictout)

def vdicttovarr_hex(vdict):
    #reg ordering for hex standard vars
    #not in regular notation from MM12
    v211 = vdict.get('v211')
    v212 = vdict.get('v212')
    v213 = vdict.get('v213')
    v222 = vdict.get('v222')
    v223 = vdict.get('v223')
    v1111 = vdict.get('v1111')
    v1112 = vdict.get('v1112')
    v1113 = vdict.get('v1113')
    v1122 = vdict.get('v1122')
    v1123 = vdict.get('v1123')
    v1222 = vdict.get('v1222')
    v1223 = vdict.get('v1223')
    v2222 = vdict.get('v2222')
    v2223 = vdict.get('v2223')
    varr = np.array(([v211,v212,v213,v222,v223,v1111,v1112,v1113,v1122,v1123,v1222,v1223,v2222,v2223]),dtype=np.float64)
    return varr


def varralttodict(varr):
    #varr = np.array(([v1123,v1122,v2223,v2222,v1112,v1113,v1222,v1223,v1111]),dtype=np.float64)
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

def varralttodict_hex(varr):
    #varr = np.array(([v1123,v1122,v2223,v2222,v1112,v1113,v1222,v1223,v1111]),dtype=np.float64)
    #hex_translate_correct_v4val
    #not in regular notation from MM12
    vdict = {}
    vdict['v211']  = varr[0]
    vdict['v212']  = varr[1]
    vdict['v213'] = varr[2]
    vdict['v222'] = varr[3]
    vdict['v223'] = varr[4]
    vdict['v1111'] = varr[5]
    vdict['v1112'] = varr[6]
    vdict['v1113'] = varr[7]
    vdict['v1122'] = varr[8]
    vdict['v1123'] = varr[9]
    vdict['v1222'] = varr[10]
    vdict['v1223'] = varr[11]
    vdict['v2222'] = varr[12]
    vdict['v2223'] = varr[13]
    return vdict

def mvtocarr(vdict):
    #cubic microsymm, triclinic macrosymm
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

    #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
    #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
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

    #form
    # tex_c = dict()
    # tex_c['c4-40'] = c_arr[0]
    # tex_c['c4-30'] = c_arr[1]
    # tex_c['c4-20'] = c_arr[2]
    # tex_c['c4-10'] = c_arr[3]
    # tex_c['c400'] = c_arr[4]
    # tex_c['c410'] = c_arr[5]
    # tex_c['c420'] = c_arr[6]
    # tex_c['c430'] = c_arr[7]
    # tex_c['c440'] = c_arr[8]
    return c_arr

def mvtoc_hex_arr(vdict):
    #hex microsymm, triclinic macrosymm
    #updated c2-10 signs (real term and imaginary term both switched) and same sitaution for c4-30,c4-10
    #2nd order
    v211 = vdict.get('v211')
    v212 = vdict.get('v212')
    v213 = vdict.get('v213')
    v222 = vdict.get('v222')
    v223 = vdict.get('v223')
    #4th order

    v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233 = hex_translate_correct_v4val(vdict)

    varr = np.array(([v211,v212,v213,v222,v223,v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233]),dtype=np.float64)

    #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
    #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
    trans_arr = np.array([[-1.0*(81.0/10.0), (81.0j)/5.0, 0.0, 81.0/10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0* (81.0/5.0), 0.0, -(81.0j)/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [( 81.0* np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, (81.0 * np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0,1.0* (81.0/5.0),0.0,(81.0j)/5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-81.0/10.0,-((81.0j)/5.0),0.0,81.0/10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        [0.0,0.0,0.0,0.0,0.0,18,-9.0j,0.0,0.0,63.0/4.0,9.0j,0.0,0.0,-1.0*(9.0/4.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*9.0/np.sqrt(2.0),1.0*(27.0j)/(np.sqrt(2.0)),0.0,0.0,27.0/(np.sqrt(2.0)),-9.0j/(np.sqrt(2.0)),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),-9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),-9.0j*np.sqrt(7.0/2.0),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0),0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0],
        [0.0,0.0,0.0,0.0,0.0,0.0,-9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,-9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*(9.0/np.sqrt(2.0)),-1.0*((27.0j)/np.sqrt(2.0)),0.0,0.0,27.0/np.sqrt(2.0),(9.0j)/np.sqrt(2.0),0.0],[0.0,0.0,0.0,0.0,0.0,18.0,9.0j,0.0,0.0,63.0/4.0,-9.0j,0.0,0.0,-1.0*(9.0/4.0)]])

    c_arr = np.zeros((14),dtype=complex)

    c_arr = np.dot(trans_arr,varr)
    # tex_c = dict()
    # tex_c['c2-20'] = c_arr[0]
    # tex_c['c2-10'] = c_arr[1]
    # tex_c['c200'] = c_arr[2]
    # tex_c['c210'] = c_arr[3]
    # tex_c['c220'] = c_arr[4]
    # tex_c['c4-40'] = c_arr[5]
    # tex_c['c4-30'] = c_arr[6]
    # tex_c['c4-20'] = c_arr[7]
    # tex_c['c4-10'] = c_arr[8]
    # tex_c['c400'] = c_arr[9]
    # tex_c['c410'] = c_arr[10]
    # tex_c['c420'] = c_arr[11]
    # tex_c['c430'] = c_arr[12]
    # tex_c['c440'] = c_arr[13]

    return c_arr

# def varralttodictpy2(varr):
#     #varr = np.array(([v1123,v1122,v2223,v2222,v1112,v1113,v1222,v1223,v1111]),dtype=np.float64)
#     vdict = {}
#     vdict['v1111'] = varr[8]
#     vdict['v1112'] = varr[4]
#     vdict['v1113'] = varr[5]
#     vdict['v1122'] = varr[1]
#     vdict['v1123'] = varr[7]
#     vdict['v1222'] = varr[6]
#     vdict['v1223'] = varr[0]
#     vdict['v2222'] = varr[3]
#     vdict['v2223'] = varr[2]
#     return vdict

# def vdicttovaltpy2(vdict):
#     #alternate ordering the py2 covariance prefers
#     v1111 = vdict.get('v1111')
#     v1112 = vdict.get('v1112')
#     v1113 = vdict.get('v1113')
#     v1122 = vdict.get('v1122')
#     v1123 = vdict.get('v1123')
#     v1222 = vdict.get('v1222')
#     v1223 = vdict.get('v1223')
#     v2222 = vdict.get('v2222')
#     v2223 = vdict.get('v2223')
#     varr = np.array(([v1123,v1122,v2223,v2222,v1112,v1113,v1222,v1223,v1111]),dtype=np.float64)
#     return varr

# def mvtoc_hex_old(vdict):
#     #hex microsymm, triclinic macrosymm
#     #2nd order
#     v211 = vdict.get('v211')
#     v212 = vdict.get('v212')
#     v213 = vdict.get('v213')
#     v222 = vdict.get('v222')
#     v223 = vdict.get('v223')
#     #4th order

#     v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233 =hex_translate_correct_v4val(vdict)

#     varr = np.array(([v211,v212,v213,v222,v223,v41111,v41112,v41113,v41123,v41133,v41222,v41223,v42223,v42233]),dtype=np.float64)

#     #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
#     #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
#     trans_arr = np.array([[-1.0*(81.0/10.0), (81.0j)/5.0, 0.0, 81.0/10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, -1.0* (81.0/5.0), 0.0, (81.0j)/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [( 81.0* np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, (81.0 * np.sqrt(3.0/2.0))/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0,1.0* (81.0/5.0),0.0,(81.0j)/5.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[-81.0/10.0,-((81.0j)/5.0),0.0,81.0/10.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
#         [0.0,0.0,0.0,0.0,0.0,18,-9.0j,0.0,0.0,63.0/4.0,9.0j,0.0,0.0,-1.0*(9.0/4.0)],
#         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0/np.sqrt(2.0),-1.0*(27.0j)/(np.sqrt(2.0)),0.0,0.0,-27.0/(np.sqrt(2.0)),9.0j/(np.sqrt(2.0)),0.0],
#         [0.0,0.0,0.0,0.0,0.0,0.0,9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
#         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0,0.0,-9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0],
#         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0),0.0,0.0,0.0,-1.0*((9.0*np.sqrt(35.0/2.0))/2.0)],
#         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0,0.0,9.0*np.sqrt(7.0/2.0),9.0j*np.sqrt(7.0/2.0),0.0],
#         [0.0,0.0,0.0,0.0,0.0,0.0,-9.0j*np.sqrt(7.0),0.0,0.0,(9.0*np.sqrt(7.0))/2.0,-9.0j*np.sqrt(7.0),0.0,0.0,-1.0*((9.0*np.sqrt(7.0))/2.0)],
#         [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0*(9.0/np.sqrt(2.0)),-1.0*((27.0j)/np.sqrt(2.0)),0.0,0.0,27.0/np.sqrt(2.0),(9.0j)/np.sqrt(2.0),0.0],[0.0,0.0,0.0,0.0,0.0,18.0,9.0j,0.0,0.0,63.0/4.0,-9.0j,0.0,0.0,-1.0*(9.0/4.0)]])

#     c_arr = np.zeros((14),dtype=complex)

#     c_arr = np.dot(trans_arr,varr)
#     tex_c = dict()
#     tex_c['c2-20'] = c_arr[0]
#     tex_c['c2-10'] = c_arr[1]
#     tex_c['c200'] = c_arr[2]
#     tex_c['c210'] = c_arr[3]
#     tex_c['c220'] = c_arr[4]
#     tex_c['c4-40'] = c_arr[5]
#     tex_c['c4-30'] = c_arr[6]
#     tex_c['c4-20'] = c_arr[7]
#     tex_c['c4-10'] = c_arr[8]
#     tex_c['c400'] = c_arr[9]
#     tex_c['c410'] = c_arr[10]
#     tex_c['c420'] = c_arr[11]
#     tex_c['c430'] = c_arr[12]
#     tex_c['c440'] = c_arr[13]
    #return tex_c

# def mvtoc_old(vdict):
#     #prior to conjugate error fix
#     #cubic microsymm, triclinic macrosymm
#     v11 = vdict.get('v1111')
#     v16 = vdict.get('v1112')
#     v15 = vdict.get('v1113')
#     v12 = vdict.get('v1122')
#     v14 = vdict.get('v1123')
#     v26 = vdict.get('v1222')
#     v46 = vdict.get('v1223')
#     v22 = vdict.get('v2222')
#     v24 = vdict.get('v2223')
#     varr = np.array(([v11,v16,v15,v12,v14,v26,v46,v22,v24]),dtype=complex)

#     #ordering is v1111(11),v1112(16),v1113(v15),v1122(v12),v1123(v14),v1222(v26),v1223(v46),v2222(v22),v2223(v24)
#     #row multiplication indices correspond to -4,0;-3,0;-2,0;-1,0;0,0;1,0;2,0;3,0;4,0
#     trans_arr = np.array([[3.0*np.sqrt(21.0)/8.0,-(3.0j*np.sqrt(21.0))/2.0,0.0,-(9.0 * np.sqrt(21.0))/4.0,0.0,(3.0j*np.sqrt(21.0))/2.0,0.0,(3.0*np.sqrt(21.0))/8.0,0.0],
#         [0.0, 0.0,(3.0 * np.sqrt(21./2.))/2.0, 0.0, -9.0j/2.0 * np.sqrt(21./2.), 0.0,-((9.0 * np.sqrt(21./2.))/2.), 0.0, 3.0j/2. * np.sqrt(21.0/2.0)],
#         [-(21. * np.sqrt(3.))/4.,(21.0j*np.sqrt(3.))/2.,0.0,0.0,0.0,(21.0j * np.sqrt(3.0))/2.0 ,0.0 ,(21.0 * np.sqrt(3.0))/4.0,0.0],
#         [0.0,0.0,-((21.0 * np.sqrt(3.0/2.0))/2.0),0.0,21.0j/2.0*np.sqrt(3.0/2.),0.0,-(21.0 *np.sqrt(3.0/2.0)/2.0),0.0,21.0j/2.0 * np.sqrt(3.0/2.0)],
#         [(21.0 * np.sqrt(15.0/2.0))/4.0 , 0.0  , 0.0,   (21.0 * np.sqrt(15.0/2.0))/2.0 ,0.0  ,0.0, 0.0 , (21.0 * np.sqrt(15.0/2.0))/4.0 ,  0.0],
#         [0.0,0.0,(21.0*np.sqrt(3.0/2.0))/2.0,0.0, 21.0j/2.0 * np.sqrt(3.0/2.0),0.0,(21.0 * np.sqrt(3.0/2.0))/2.0,0.0, 21.0j/2.0 * np.sqrt(3.0/2.0)],
#         [-(21.0 * np.sqrt(3.0))/4.0,-(21.0j * np.sqrt(3.0))/2.0, 0.0, 0.0,0.0,-((21.0j * np.sqrt(3.0))/2.0), 0.0,(21.0 * np.sqrt(3.0))/4.0,0.0],
#         [0.0, 0.0, -((3 * np.sqrt(21/2))/2),0.0,-(9.0j/2.0)*np.sqrt(21/2),0.0, (9.0*np.sqrt(21.0/2.0))/2.0,0.0,3.0j/2.0 * np.sqrt(21.0/2.0)],
#         [(3.0*np.sqrt(21.0))/8.0,(3.0j*np.sqrt(21.0))/2.0,0.0,-((9.0*np.sqrt(21.0))/4.0),0.0,-(3.0j*np.sqrt(21.0)/2.0),0.0,(3.0*np.sqrt(21.0))/8.0,0.0]])

#     c_arr = np.zeros((9),dtype=complex)

#     c_arr = np.dot(trans_arr,varr)
#     tex_c = dict()
#     tex_c['c4-40'] = c_arr[0]
#     tex_c['c4-30'] = c_arr[1]
#     tex_c['c4-20'] = c_arr[2]
#     tex_c['c4-10'] = c_arr[3]
#     tex_c['c400'] = c_arr[4]
#     tex_c['c410'] = c_arr[5]
#     tex_c['c420'] = c_arr[6]
#     tex_c['c430'] = c_arr[7]
#     tex_c['c440'] = c_arr[8]

    #return tex_c
# def outputmatlabclmnsdiscretefull(meandict,covarordered,N, filename):
#     np.random.seed(4)
#     vmeans_vect_altorder  = vdicttov(meandict)
#     if 'rs' in meandict:
#         #need to remove extra error output column when rs error added
#         covarordered = np.delete(np.delete(np.array(covarordered),10,0),10,1)
#     cutcovar = np.delete(np.delete(np.array(covarordered),9,0),9,1)
#     vtensoutvariate =  np.random.multivariate_normal(vmeans_vect_altorder,cutcovar,size=N)
#     clmnoutvar = np.zeros(vtensoutvariate.shape,dtype=complex)
#     for i in range(vtensoutvariate.shape[0]):
#         clmnoutvar[i,:] = mvtocarr(varralttodict(vtensoutvariate[i,:]))
#     prefix = filename
#     #clmn ordered as c4-40 -> c440
#     dictout = {'Clmn_ordered':clmnoutvar}
#     scipy.io.savemat(prefix +'_clmn.mat', dictout)

# def outputmatlabclmnsdiscretefull_hex(meandict,covarordered,N, filename):
#     np.random.seed(4)
#     #this is the normal ordering. check it.
#     varr = vdicttovarr_hex(meandict)

#     if 'rs' in meandict:
#         covarordered = np.delete(np.delete(np.array(covarordered),15,0),15,1)
#     cutcovar = np.delete(np.delete(np.array(covarordered),14,0),14,1)
#     vtensoutvariate =  np.random.multivariate_normal(varr,cutcovar,size=N)
#     clmnoutvar = np.zeros(vtensoutvariate.shape,dtype=complex)
#     for i in range(vtensoutvariate.shape[0]):
#         clmnoutvar[i,:] = mvtoc_hex_arr(varralttodict_hex(vtensoutvariate[i,:]))
#     prefix = filename
#     #clmn ordered as c4-40 -> c440
#     dictout = {'Clmn_ordered':clmnoutvar}
#     scipy.io.savemat(prefix +'_clmn.mat', dictout)

#def gen_4th_varr_old(vdict):
#     #generate relations between V indices and transform into fernandez normalized 4th order V[4,1][alpha,beta]
#     v = np.zeros((6,6),dtype=np.float64)
#     #voigt index notation for V defined with the sqrt(2) and factors of 2 here, then translated to 4th order to add them

#     v[0,0] = vdict.get('v1111')
#     v[1,1] = vdict.get('v2222')
#     v[2,2] = vdict.get('v1111') + 2.0*vdict.get('v1122') + vdict.get('v2222') #v1111 + 2*v1122 + v2222

#     v[3,3] = -vdict.get('v1122') - vdict.get('v2222') #v44 = -v12-v22
#     v[4,4] = -vdict.get('v1111') - vdict.get('v1122') #v55 = -v11-v12
#     v[5,5] = vdict.get('v1122')

#     v[0,1] = v[1,0] = vdict.get('v1122') #v1122
#     v[2,0] = v[0,2] = -(vdict.get('v1111') + vdict.get('v1122')) #v13 = -v11-v12
#     v[1,2] = v[2,1] = -(vdict.get('v1122') + vdict.get('v2222')) #v23 = -v12-v22

#     v[0,3] = v[3,0] = vdict.get('v1123') #v1123
#     v[0,4] = v[4,0] = vdict.get('v1113') #v1113
#     v[0,5] = v[5,0] = vdict.get('v1112') #v1112

#     v[1,3] = v[3,1] = vdict.get('v2223') #v2223
#     v[1,4] = v[4,1] = vdict.get('v1223') #v1223
#     v[1,5] = v[5,1] = vdict.get('v1222') #v1222

#     v[1,3] = v[3,1] = -(vdict.get('v1223') + vdict.get('v2223')) #-v1223 - v2223
#     v[1,4] = v[4,1] = -(vdict.get('v1113') + vdict.get('v1223')) #-v1113 - v1223
#     v[1,5] = v[5,1] = -(vdict.get('v1112') + vdict.get('v1222')) #-v1112 - v1222

#     v[3,4] = v[4,3] = -(vdict.get('v1112')+vdict.get('v1222')) #-(v1112 + v1222)
#     v[3,5] = v[5,3] = vdict.get('v1223') #v1223

#     v[4,5] = v[5,4] = vdict.get('v1123') #v1123
#     #regular 4th order transform here (non nvn normalized in mathematica codes)
#     fourthorder = recover4order(v)

#     return fourthorder
