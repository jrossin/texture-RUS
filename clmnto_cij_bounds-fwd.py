import numpy as np
import rus_tools as rus

cdict = {'c4-40': -0.264265810933726 + 0.0722269237317015j, 'c4-30':0.0113526585042093 + 0.0589993638623195j, 'c4-20':-0.0747215930561519 - 0.265735726214126j, 'c4-10':-0.375102099186336 - 0.317700816255007j, 'c400':0.728980432247045 - 1.13853863141151e-16j, 'c410':-0.375102099186336 + 0.317700816255007j, 'c420':-0.0747215930561519 + 0.265735726214126j,'c430':0.0113526585042093 - 0.0589993638623195j, 'c440':-0.264265810933726 - 0.0722269237317015j}
texvaldict = rus.mctov(cdict)
scdict = {'sc11': 243.3,'sc12': 156.7, 'sc44': 117.8}
zeroboundlow, zeroboundhigh, csci_e, csnvn = rus.calc_zero_bounds(scdict)
cmat, upper_HS, lower_HS = rus.texture_to_c_cub(texvaldict,zeroboundlow, zeroboundhigh,csci_e, csnvn)
np.set_printoptions(suppress=True)
print('Self consistent solution')
print(cmat.tolist())
print('Upper HS Bound')
print(upper_HS.tolist())
print('Lower HS Bound')
print(lower_HS.tolist())
