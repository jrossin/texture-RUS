import numpy as np
import scipy.linalg as la
import scipy.io
from scipy.stats import norm
import rus_tools as rus
import sympy as sp
import pickle
import os
from scipy.stats import norm, invgamma
#from mpi4py import MPI
from sympy.physics.quantum import TensorProduct
#from smcpy.smc.smc_sampler import SMCSampler
#print(range(-1,2*12+2))

# for i in range(steplast.params.shape[1]):
#   print(rus.mvtoc(rus.varralttodict(steplast.params[i,:])))
# print(steplast.params.shape)
# print((steplast.param_names))
# print(type(steplast.weights))
# print((steplast.weights).shape)
# print()
params = {'c11':276.43 ,'c22': 270.8 ,'c33': 271.68 ,'c44': 104.83 ,'c55': 98.73 ,'c66': 98.75 ,'c12': 131.2 ,'c13': 130.3 ,'c23': 135.97 ,'c14':2.272 ,'c15':4.184 ,'c16':16.3 ,'c24':1.57927 ,'c25':4.906 ,'c26':2.251 ,'c34':0.6936 ,'c35': 0.7222 ,'c36':14.117 ,'c45':17.113 ,'c46':6.25 ,'c56':4.26}
print(rus.forward_tric(params))
temp = np.array([1.0+5.0j])
print(temp.imag)
prnt = {'v1111': norm(0.08, 0.1),
                     'v1112': norm(0.011, 0.1),
                     'v1113': norm(0.012, 0.1),
                     'v1122': norm(-0.043, 0.1),
                     'v1123': norm(-0.015, 0.1),
                     'v1222': norm(-0.012, 0.1),
                     'v1223': norm(0.0049, 0.1),
                     'v2222': norm(0.128, 0.1),
                     'v2223': norm(-0.008, 0.1),
                     #'rs': norm(-0.0125,0.002),
                     'std_dev': invgamma(25, scale=100)}
p_1 = rus.p1()
print(rus.voigt(rus.isomat_fromeig(np.array([538,268.2]),p_1,rus.p2(p_1,rus.identitymat()))))
# cdict = {'c4-40':0.260947738432631 + 0.193036844097765j, 'c4-30':0.0475175603703459 - 0.0546739021604664j, 'c4-20':0.0329815746148098 + 0.117546504456313j, 'c4-10':-0.0518092752070339 + 0.0652201070362246j, 'c400':1.65663247536998 + 0j, 'c410':-0.0518092752070340 - 0.0652201070362249j, 'c420':0.0329815746148098 - 0.117546504456313j,'c430':0.0475175603703459 + 0.0546739021604664j, 'c440':0.260947738432631 - 0.193036844097765j}
# texvaldict = rus.mctov(cdict)

texvaldict = {'v1111': 0.08843065552468005, 'v1112': -0.004054312883604645, 'v1113': 0.01732047596576196, 'v1122': -0.037248385401065356, 'v1123': -0.001535875410930425, 'v1222': -0.03843050100833657, 'v1223': -0.009471970862121783, 'v2222': 0.11466859322998971, 'v2223': 0.007694970253297949, 'std_dev': 3.6496367362537363}
scdict = {'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1}

print(rus.voigt(rus.gen_4th_varr(texvaldict)))
zeroboundlow, zeroboundhigh, csci_e, cs = rus.calc_zero_bounds(scdict)
print('break')
print(zeroboundlow)
print(zeroboundhigh)
print(csci_e)
print(rus.voigt(cs))
cmat, upper_HS, lower_HS = rus.texture_to_c_cub(texvaldict,zeroboundlow, zeroboundhigh, csci_e, cs)
v4 = rus.gen_4th_varr(texvaldict)
reuss = rus.reusscalc(cs,v4)
vv = rus.voigtcalc(cs,v4)
print(cmat)
print(upper_HS)
print(lower_HS)
print(rus.voigt(vv))
print(rus.voigt(reuss))

par = 15

q = norm(-0.037468247626918881, 0.010765307413027799)
proposal_pdfs = np.zeros((55))
samps = q.rvs(55)
print(samps)
proposal_pdfs[:] = q.pdf(samps)
print(proposal_pdfs)
cdic = {'v1123': -0.07157791323664324, 'v1122': 0.00825516800347421, 'v2223': 0.03852369879256294, 'v2222': -0.0005879184439700257, 'v1112': -0.04668760804651331, 'v1113': 0.006290282092454987, 'v1222': 0.033883771102722275, 'v1223': 0.005734323020316487, 'std_dev': 4.892596944700233, 'v1111': -0.06511102026470258}
varitest = np.array([[ 1.85371366e-04,  7.00261391e-05, -1.48914006e-04, -9.00603052e-05,
  -2.41740656e-05,  5.29373947e-05,  4.44552010e-05,  2.82413933e-05,
  -1.58495540e-05, -6.20125421e-06, -2.35677888e-03],
 [ 7.00261391e-05,  1.34352259e-03, -8.10632376e-04, -2.44933475e-05,
   1.94139218e-03, -4.44695925e-04,  5.28545672e-05,  1.54186993e-05,
  -6.09869928e-04,  1.14301658e-06,  4.82693555e-04],
 [-1.48914006e-04, -8.10632376e-04,  1.77524145e-03,  9.45596179e-05,
  -8.20495564e-04, -2.24746322e-04, -2.75458204e-04, -1.04055737e-04,
   4.45382672e-04, -1.60806092e-06, -1.10710867e-03],
 [-9.00603052e-05, -2.44933475e-05,  9.45596179e-05,  1.00343110e-04,
   1.95112578e-05, -4.11077742e-05, -4.71521524e-05, -7.18554837e-05,
   4.75155241e-05,  2.91110308e-07,  9.87146367e-04],
 [-2.41740656e-05,  1.94139218e-03, -8.20495564e-04,  1.95112578e-05,
   3.86970052e-03, -1.19374006e-03, -1.71762177e-04,  4.78702262e-05,
  -9.16235194e-04,  8.73518696e-06,  2.61716591e-03],
 [ 5.29373947e-05, -4.44695925e-04, -2.24746322e-04, -4.11077742e-05,
  -1.19374006e-03,  1.02620206e-03,  2.64899080e-04,  4.39398505e-05,
   2.97670823e-05,  9.82291605e-07,  2.21257039e-03],
 [ 4.44552010e-05,  5.28545672e-05, -2.75458204e-04, -4.71521524e-05,
  -1.71762177e-04,  2.64899080e-04,  2.91149919e-04,  5.14788856e-05,
  -2.63374442e-04, -7.77157070e-06, -1.96523147e-03],
 [ 2.82413933e-05,  1.54186993e-05, -1.04055737e-04, -7.18554837e-05,
   4.78702262e-05,  4.39398505e-05,  5.14788856e-05,  1.85670321e-04,
  -2.27831855e-05,  2.03967108e-06, -1.56549773e-03],
 [-1.58495540e-05, -6.09869928e-04,  4.45382672e-04,  4.75155241e-05,
  -9.16235194e-04,  2.97670823e-05, -2.63374442e-04, -2.27831855e-05,
   1.36494347e-03, -6.25799289e-07,  3.96961696e-04],
 [-6.20125421e-06,  1.14301658e-06, -1.60806092e-06,  2.91110308e-07,
   8.73518696e-06,  9.82291605e-07, -7.77157070e-06,  2.03967108e-06,
  -6.25799289e-07,  6.41081302e-06,  7.13145243e-04],
 [-2.35677888e-03,  4.82693555e-04, -1.10710867e-03,  9.87146367e-04,
   2.61716591e-03,  2.21257039e-03, -1.96523147e-03, -1.56549773e-03,
   3.96961696e-04,  7.13145243e-04,  5.55474134e-01]])


An = np.array(([[4.,-1.,2.], [-1.,6.,0.], [2.,0.,5.]]),dtype=np.float64)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    print('rank==0 flag correct')
else:
    print('unused rank core')

#tes = np.array(([[[[ 0.00435044 , 0.  ,        0.        ], [ 0.,         -0.,00148289 , 0.        ],[ 0.        ,  0.        , -0.,00078422]],[[ 0.        ,  0.00291667 , 0.        ],[ 0.00291667 , 0.         , 0.        ],[ 0.          ,0.         , 0.        ]],[[ 0.          ,0.,          0.00281804],[ 0.          ,0.,          0.        ],[ 0.00281804  ,0.,          0.        ]]],[[[ 0.         , 0.00291667 , 0.        ],[ 0.00291667 , 0.        ,  0.        ],[ 0.         , 0.        ,  0.        ]],[[-0.,00148289 , 0.         , 0.        ],[ 0.          ,0.00435044,  0.        ],[ 0.        ,  0.        , -0.,00078422]],[[ 0.       ,   0.         , 0.        ],[ 0.        ,  0.         , 0.00281804],[ 0.       ,   0.00281804 , 0.        ]]],[[[ 0.         , 0.        ,  0.00281804],[ 0.         , 0.        ,  0.        ],[ 0.00281804 , 0.        ,  0.        ]],[[ 0.       ,  0.         , 0.        ],[ 0.       ,   0.        ,  0.00281804],[ 0.       ,   0.00281804 , 0.        ]],[[-0.,00078422,  0.         , 0.        ],[ 0.        , -0.,00078422 , 0.        ], [ 0.        ,  0.         , 0.00365177]]]]),dtype=np.float64)

print(len(rus.out9param()))
print(len(rus.out21param()))
print(len(rus.out21paramhex()))
uno = rus.out9param()
two = rus.out21paramhex()
#x = 27:
print(uno[27:])
print(two[63:])

c1, c2 = sp.symbols('c1 c2',real=True)
p_1 = rus.p1()

print(sp.Matrix(1/np.array([3.,4.])))
p_2 = rus.p2(p_1,rus.identitymat())
pr = rus.p0(c1,c2,p_1,p_2)
print(rus.isomat_fromeig(np.array([300,100]),p_1,p_2))
a = pr[1,1,1,1]

c11 = 243.3
c12 = 156.7
c44 = 117.8
csnvn = np.array([[c11, c12, c12, 0., 0., 0.],[c12,c11,c12, 0.,0.,0.],[c12,c12,c11,0.,0.,0.],\
        [0.,0.,0.,2.*c44,0.,0.],[0.,0.,0.,0.,2.*c44,0.],[0.,0.,0.,0.,0.,2.*c44]])
cs3s = rus.recovernormalized4th(csnvn)
ret = rus.csci_iso(cs3s)
print(ret)
p1 = rus.p1()

print(rus.scalarprod(cs3s,p1))
print(sum(np.multiply(cs3s,p1).flatten()))
#print(tes)
#out = rus.hd(tes)
#print("array")
#params = {'v1123': -0.,018686909473444268, 'v1122': -0.,019064408608724638, 'v2223': -0.,064764266823702454, 'v2222': 0.01521595716716675, 'v1112': -0.,07243459249499723, 'v1113': 0.098625284245557901, 'v1222': 0.054142146179420231, 'v1111': -0.,063950770901801568, 'v222': -0.,10427319180846173, 'v223': 0.1860372853057907, 'v213': 0.17121754697191655, 'v212': 0.13156746465923602, 'v211': 0.1311752331486688,  'v1223': -0.,059018924171768558}


vtoc = rus.mvtoc_hex(params)
print(vtoc)

fourthcubv = rus.gen_4th_varr(params)
ret = rus.frobnormtens(fourthcubv)
print(ret)


sc = {'sc11': 169.0, 'sc12': 89.0, 'sc13': 62.0, 'sc33': 196.0,'sc44': 43.0}
one, two, three = rus.calc_zero_bounds(sc)
fourth = rus.recovernormalized4th(three)
print(fourth)
print(rus.frobnormtens(fourth))

#print(An)
name = 'inverse_e56_sectioned_s2_12_55_4500_40_10_norm_dist_param'
testprint = 'results/' + name + '_pairwise_plot'
print(testprint)
#steplist = SMCSampler.load_step_list('current_step.h5')


s.plot_pairwise_weights(save=True, show=True,prefix='results/' + name + '_pairwise_plot')
#print(np.array([np.zeros((3,3),dtype=np.float64)]))
#An.shape[]
print('END solve')
#B = np.array([[-2.4098315935342874e-14,0.,0.],[0.,4.819663187068575e-14,0.],[0.,0.,-2.4098315935342874e-14]])
#print(rus.p1())
cdic = {'c22': 290.03574539774331, 'c23': 137.39746621672992, 'c55': 71.878684885178558, 'c36': 0.064455877384584997, 'c26': 6.8662591095616587, 'c24': -23.661695221616526, 'c56': -13.842115783409696, 'c66': 84.876372804468872, 'c25': -3.1007777871992115, 'c46': 0.39099853841112664, 'c45': 6.8569712597338919, 'c44': 84.709788631645196, 'c33': 300.49703456979853, 'c12': 140.10170910768724, 'c35': -1.0135518981295155, 'c34': -13.255612005869583, 'c11': 304.40479557098956, 'c13': 124.79078535192433, 'c16': -8.1650082303405433, 'c15': -0.59969272984944877, 'c14': -28.748122686611303}
cstd = {'c22': 2.7614661679475083, 'c23': 2.72012217652785, 'c55': 1.5652904265932785, 'c26': 6.470876514914238, 'c24': 7.6696429294161632, 'c56': 3.0054907879598396, 'c66': 1.5240030216835911, 'c25': 7.4702225268131244, 'c46': 2.8376638222539854, 'c45': 4.4393910471572919, 'c44': 2.9324346489714967, 'c13': 1.8047356383443702, 'c15': 7.694524962777181, 'c34': 3.4126540322838239, 'c35': 1.6526412573920299, 'c12': 2.4894314636564054, 'c11': 3.5472118670464234, 'c36': 2.1624168416407268, 'c16': 3.4976262991023739, 'c33': 3.0569756416955793, 'c14': 6.4891110860230592}




#c_ten = {'c4-40': -0.,05 + 0.001j,'c4-30': 0.0001 + 0.001j,'c4-20': -0.,08+0.001j,'c4-10': 0.0001 + 0.001j,'c400':0.6 + 0.001j,'c410':0.0001 + 0.001j,'c420':-0.,08+0.001j,'c430':0.0001 + 0.001j,'c440':-0.,05 + 0.001j}

# varr = rus.mctov(c_ten)
# vtens = rus.gen_4th_varr(varr.real)
# print(varr)
# print('c trans to v 4th')
# print(vtens)

vdic = {'v1123': -0.014696357018388589, 'v1122': -0.007631598511492135, 'v2223': -0.035480564914675841, 'v2222': 0.038251367193729409, 'v1112': -0.00071567395582977514, 'v1113': -0.014197116510742193, 'v1222': 0.0023534287354442006, 'v1223': 0.022616009056595714, 'std_dev': 2.8988518725841548, 'v1111': 0.041616948270973772}
varr = rus.vdicttov(vdic)
vtens2 = rus.gen_4th_varr(varr)

vvoight = rus.voigt(vtens2)
print(vvoight)


#test addition of complex vals
varr = rus.vdicttov(vdic)
vtens2 = rus.gen_4th_varr(varr)
print(vtens2)
vvoight = rus.voigt(vtens2)
print(vvoight)

varrc = rus.vdicttov_complex(vdic)
vtens3 = rus.gen_4th_varr_comp(varrc)
print(vtens3)
#print(res)
#print((np.zeros((3,3))).flatten())
#print(rus.isomat_fromeig([556.7, 86.6],rus.p1(),rus.p2(rus.p1(),rus.identitymat())))
#print(rus.oafull(cs3s,rus.sgcub(),rus.sgtric(),vtens2))

#print(rus.CHS(rus.voigtcalc(cs3s,rus.sgcub(),rus.sgtric(),vtens2)))

#print(rus.recovernormalizedvoigtcomp(rus.reusscalc(cs3s,rus.sgcub(),rus.sgtric(),vtens2)))
# chol = la.cholesky(A, lower=True)
# print("cholesky test")
# print(chol)

# cholfact = la.cho_factor(A, lower=True)
# print("cholesky test factor")
# print(cholfact)

#print(rus.J62())
