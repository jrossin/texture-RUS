
from smcpy.smc.propagator_pick import Propagator_pick
from smcpy.utils.plotter import plot_pairwise
import rus.tools as rus
from scipy import stats
import os
import time
import pickle
from smcpy import ImproperUniform
import numpy as np


file = "dump.p"
if os.path.getsize(file) > 0:
  with open(file,'rb') as pf:
    newlist = pickle.load(pf)
steplast = newlist[-1]
print("Total particle Number being unpacked from dump.p:" + str(steplast.params.shape[0]))

if 'v1111' in steplast.param_names:
    from rus.rus_propagator import rus_propagator
else:
  break
#replot this number of particles out of the total

particlenumplot = 2000
#particlenumplot = steplast.params.shape[0]

steplast2 = steplast

t = time.strftime("%Y%m%d-%H%M%S")
fname = f"results/plot_pickl/replot_{t}/"
filename = fname  +'results.txt'
if not os.path.exists(os.path.dirname(filename)):

        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

output_file = open(filename, 'a+')
output_file.truncate(0)
output_file.writelines([f'\n First {particlenumplot} Particles used for Replotting\n'])
output_file.writelines([f'\n Filename from which dump.p was grabbed {file}\n'])
mean = steplast2.compute_mean()
var = steplast2.compute_variance()
std = steplast2.compute_std_dev()
covar = steplast2.compute_covariance()
print('Calculated Variable Means Prerotation:')
print(mean)
output_file.writelines([f'\n Calculated Variable Mean Prerotation {mean}\n'])
print('Calculated Variable Std-Dev Prerotation')
print(std)
output_file.writelines([f'\n Calculated Variable Std Dev Prerotation {std}\n'])
output_file.writelines([f'\n Calculated Variable Covar Prerotation {covar}\n'])
outputpngnameprop = fname + "pairwise_param_" + "Prerotation"
plot_pairwise(steplast2.params[:particlenumplot,:],weights=None,save=True, show=False,
                                                  param_names=list(steplast2.param_names),label_size = 25, tick_size=16,
                                                  prefix=outputpngnameprop)
### propagate (pre rotations)
nfreq = 100
inputs = {'order': 14, 'd1': 0.0069123, 'd2': 0.008214  , 'd3':0.008673 , 'density': 8580.06,'num_moduli':21,'sc11': 236.4, 'sc12': 150.8, 'sc44': 134.1,'nfreq':nfreq}
output_names = rus.out21param()
resfreqname = []
for i in range(nfreq):
    resfreqname.append('fr_'+ str(i+1))
resfreqname = np.array(resfreqname)
output_names = np.append(output_names,resfreqname)

rusprop = rus_propagator(inputs,None,list(steplast2.param_names))
smc_prop = Propagator_pick()


smcstepprop = smc_prop.propagate(rusprop.evaluate,steplast2,particlenumplot, output_names=output_names)
smcstepcijmean = smcstepprop.compute_mean()
smcstepcijstddev = smcstepprop.compute_std_dev()
smcstepcijcovar = smcstepprop.compute_covariance()

output_file.writelines([f'\n \n All Propagated Means: \n {smcstepcijmean}'])

output_file.writelines([f'\n All Propagted Std Devs: \n {smcstepcijstddev}'])
output_file.writelines([f'\n Ordered Covariance Matrix (by input parameter order) \n {smcstepcijcovar}'])

clmnslice = slice(63,72)
rus.outputmatlabclmnrawdata(smcstepprop.params[:particlenumplot,clmnslice],smcstepprop.weights,fname)
clmnmean = rus.ctodictimagcub(smcstepcijmean)
clmnstd = rus.ctodictimagcub(smcstepcijstddev)

output_file.writelines([f'\n \n Mean Propagated Clmn: \n {clmnmean}'])
output_file.writelines([f'\n Std Dev Propagated Clmn: \n{clmnstd}'])

cijindex = slice(0,21)
cijkeys = output_names[cijindex]
clmnkeys = output_names[clmnslice]
outputpngnamepropcij = fname +"propagatedcij_finalreplotnors"
plot_pairwise(smcstepprop.params[:particlenumplot,cijindex],weights=None,save=True, show=False,
                                                  param_names=cijkeys,label_size = 25, tick_size=18,
                                                  prefix=outputpngnamepropcij)
outputpngnameprop_clmn = fname + '_propagated_clmn'
clmnparamslice = smcstepprop.params[:particlenumplot,clmnslice]
clmnimagcub = np.concatenate((np.stack(((clmnparamslice[:,8]).imag,(clmnparamslice[:,7]).imag,(clmnparamslice[:,6]).imag,(clmnparamslice[:,5]).imag),axis=1),clmnparamslice[:,4:9]),axis=1)
clmnkeysalt = np.array(['c440imag','c430imag','c420imag','c410imag','c400','c410','c420','c430','c440'])
plot_pairwise(clmnimagcub,weights=None,save=True, show=False,
                                                  param_names=clmnkeysalt,label_size = 25, tick_size=18,
                                                  prefix=outputpngnameprop_clmn)
#########################################################

#most well defined mode #14,16,34,35 work here
a = 'c14'
#second most well defined
b = 'c15'

# ##
inda = np.where(output_names == a)
indb = np.where(output_names == b)

areg = stats.mode(np.sign(smcstepprop.params[:particlenumplot,inda]))[0]
temp = list()
for i in range(particlenumplot):
    if np.sign(smcstepprop.params[i,inda]) == areg:
        temp.append(smcstepprop.params[i,indb])
breg = stats.mode(np.sign(temp), axis=None)[0]


normalparticles = 0
xrotcount = 0
yrotcount = 0
zrotcount  = 0
smcsteprot = steplast2
for i in range(particlenumplot):
    if np.sign(smcstepprop.params[i,inda]) == areg:
        if np.sign(smcstepprop.params[i,indb]) == breg:
            normalparticles += 1
        elif np.sign(smcstepprop.params[i,indb]) != breg:
            smcsteprot.params[i,0:9] = rus.v4_to_arr(rus.rotarbit(rus.varralttodict(steplast2.params[i,:]),'z'))
            xrotcount +=1
    elif np.sign(smcstepprop.params[i,inda]) != areg:
        if np.sign(smcstepprop.params[i,indb]) != breg:
            smcsteprot.params[i,0:9] = rus.v4_to_arr(rus.rotarbit(rus.varralttodict(steplast2.params[i,:]),'y'))
            zrotcount +=1
        elif np.sign(smcstepprop.params[i,indb]) == breg:
            smcsteprot.params[i,0:9] = rus.v4_to_arr(rus.rotarbit(rus.varralttodict(steplast2.params[i,:]),'x'))
            yrotcount +=1

output_file.writelines([f'\n Number of particles unrotated: \n {normalparticles} out of total particles {particlenumplot}'])
output_file.writelines([f'\n Number of particles rotated X: \n {xrotcount}'])
output_file.writelines([f'\n Number of particles rotated Y: \n {yrotcount}'])
output_file.writelines([f'\n Number of particles rotated Z: \n {zrotcount}'])


##Propagate and rename outputs
meanrot = smcsteprot.compute_mean()
varrot = smcsteprot.compute_variance()
stdrot = smcsteprot.compute_std_dev()
covarrot = smcsteprot.compute_covariance()
print('Calculated Variable Means rotated:')
print(meanrot)
output_file.writelines([f'\n Calculated Variable Mean rotated {meanrot}\n'])
print('Calculated Variable Std-Dev rotated')
print(stdrot)
output_file.writelines([f'\n Calculated Variable Std Dev rotated {stdrot}\n'])
output_file.writelines([f'\n Calculated Variable Covar rotated {covarrot}\n'])
outputpngnameproprot = fname + "pairwise_param_" + "rotated"
plot_pairwise(smcsteprot.params[:particlenumplot,:],weights=None,save=True, show=False,
                                                  param_names=list(smcsteprot.param_names),label_size = 25, tick_size=16,
                                                  prefix=outputpngnameproprot)

rusproprot = rus_propagator(inputs,None,list(smcsteprot.param_names))
smc_prop_rot = Propagator_pick()
#index 17 for c36
smcstepproprot = smc_prop_rot.propagate(rusproprot.evaluate,smcsteprot,particlenumplot, output_names=output_names)
smcstepcijmeanrot = smcstepproprot.compute_mean()
smcstepcijstddevrot = smcstepproprot.compute_std_dev()
smcstepcijcovarrot = smcstepproprot.compute_covariance()

output_file.writelines([f'\n \n All Rotated Propagated Means: \n {smcstepcijmeanrot}'])

output_file.writelines([f'\n All Rotated Propagted Std Devs: \n {smcstepcijstddevrot}'])
output_file.writelines([f'\n Ordered Rotated Covariance Matrix (by input parameter order) \n {smcstepcijcovarrot}'])

rus.outputmatlabclmnrawdata(smcstepproprot.params[:particlenumplot,clmnslice],smcstepproprot.weights,fname)
clmnmeanrot = rus.ctodictimagcub(smcstepcijmeanrot)
clmnstdrot = rus.ctodictimagcub(smcstepcijstddevrot)
output_file.writelines([f'\n \n Mean Rotated Propagated Clmn: \n {clmnmeanrot}'])
output_file.writelines([f'\n Std Dev Rotated Propagated Clmn: \n{clmnstdrot}'])

outputpngnamepropcijrot = fname +"propagatedcij_finalreplotnors_rotated"
plot_pairwise(smcstepproprot.params[:particlenumplot,cijindex],weights=None,save=True, show=False,
                                                  param_names=cijkeys,label_size = 25, tick_size=18,
                                                  prefix=outputpngnamepropcijrot)
outputpngnameprop_clmnrot = fname + '_propagated_clmn_rotated'
clmnparamslice_rot = smcstepproprot.params[:particlenumplot,clmnslice]
clmnimagcubrot = np.concatenate((np.stack(((clmnparamslice_rot[:,8]).imag,(clmnparamslice_rot[:,7]).imag,(clmnparamslice_rot[:,6]).imag,(clmnparamslice_rot[:,5]).imag),axis=1),clmnparamslice_rot[:,4:9]),axis=1)

plot_pairwise(clmnimagcubrot,weights=None,save=True, show=False,
                                                  param_names=clmnkeysalt,label_size = 25, tick_size=18,
                                                  prefix=outputpngnameprop_clmnrot)
