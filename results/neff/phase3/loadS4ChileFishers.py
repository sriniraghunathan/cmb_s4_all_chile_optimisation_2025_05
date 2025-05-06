import matplotlib.pyplot as plt
import numpy as np
import plotTools as pt
import pickle

pythonFlag = 3
nExps = 10 # years
sTypes = ["lensed","delensed","unlensed"]
gTypes = ["Gaussian"]

debug = False
addSO = True

if debug:
    fskywide = np.ones(4)#*0.62
    fskydelensing = np.ones(4)#*0.62
else:
    #apod = True
    fskywide = np.array([0.064, 0.31, 0.016, 0.027])
    fskydelensing = np.array([0.043, 0.021, 0.010, 0.015])
widelensingoverlap = 0.0 #0.0067
fskywidesansoverlap = fskywide - widelensingoverlap

#https://arxiv.org/abs/2503.00636
fskySOFull = 0.4
sigmaNeffSOFull = 0.045
fskySOLeftover = 0.128
sigmaNeffSOLeftover = sigmaNeffSOFull * np.sqrt(fskySOFull/fskySOLeftover)

fishers = dict()
fishers['wide'] = dict()
fishers['delensing'] = dict()
for j in range(4):
    patch = j+1
    fishers['wide'][patch] = dict()
    fishers['delensing'][patch] = dict()
fishers['wide']['all'] = dict()
fishers['delensing']['all'] = dict()
fishers['S4'] = dict()

whichgt = 'Gaussian'
whichst = 'delensed'

for j in range(4):
    patch = j+1
    wideName = '/u/ctrendaf/scratch/results/'+\
        'fisher_CMB-S4_Chile_Wide_Patch'+str(patch)+'_lmaxT5000_lmax5000_lmin30_PlanckNoise'
    delensingName = '/u/ctrendaf/scratch/results/'+\
        'fisher_CMB-S4_Chile_Delensing_Patch'+str(patch)+'_lmaxT5000_lmax5000_lmin30_PlanckNoise'
    
    fishersw, cosmoParams = pt.loadGaussianNG(jobName = wideName, pythonFlag = pythonFlag,\
                                              gTypes = gTypes, returnCosmoParams = True)
    fishersd = pt.loadGaussianNG(jobName = delensingName, pythonFlag = pythonFlag,\
                                gTypes = gTypes, returnCosmoParams = False)
    
    paramsToFix = ['mnu']

    # fisher + fix + fsky + tau -> invert
    for i in range(nExps):
        for gt, gaussianType in enumerate(gTypes):
            for st, spectrumType in enumerate(sTypes):
                fisher = fishersw[gaussianType][spectrumType][i]
                fishersw[gaussianType][spectrumType][i], fixedParams = pt.fixParameters(fisher = fisher,\
                                                                                        cosmoParams = cosmoParams,\
                                                                                        paramsToFix = paramsToFix,\
                                                                                        returnFixedParamList=True)
                fisher = fishersd[gaussianType][spectrumType][i]
                fishersd[gaussianType][spectrumType][i] = pt.fixParameters(fisher=fisher,\
                                                                           cosmoParams = cosmoParams,\
                                                                            paramsToFix = paramsToFix,\
                                                                            returnFixedParamList=False)
    fisherswSky = pt.addfsky(fishersw, fskywidesansoverlap[j], gTypes=gTypes)
    fishersdSky = pt.addfsky(fishersd, fskydelensing[j],gTypes=gTypes)
    if patch==1:
        fisherswTau = pt.addTau(fisherswSky, fixedParams, gTypes=gTypes)
    else:
        fisherswTau = fisherswSky
    #fishersdTau = pt.addTau(fishersdSky, fixedParams, gTypes=gTypes)
    fishersdTau = fishersdSky
    covsw = pt.invertFishers(fisherswTau, gTypes=gTypes)
    covsd = pt.invertFishers(fishersdTau, gTypes=gTypes)
    sigmasw = pt.getSigmas(covsw, gTypes=gTypes)
    sigmasd = pt.getSigmas(covsd, gTypes=gTypes)
    indexNeff = fixedParams.index('N_eff')
    for i in range(nExps):
        year = i+1
        print('Patch '+str(patch)+', '+str(year)+' years:')
        print('Wide sigma = '+str(sigmasw[whichgt][whichst][i][indexNeff]))
        print('Delensing sigma = '+str(sigmasd[whichgt][whichst][i][indexNeff]))
        fishers['wide'][patch][year] = fisherswTau[whichgt][whichst][i]
        fishers['delensing'][patch][year] = fishersdTau[whichgt][whichst][i]
print('----------')
nParams = len(fixedParams)
# #debug
# print(fishers)
# #debug
print('sigma Neff per year, wide + delensing: ')
for i in range(nExps):
    year = i+1
    totalFisher = np.zeros((nParams,nParams))
    wideFisher = np.zeros((nParams,nParams))
    delensingFisher = np.zeros((nParams,nParams))
    for j in range(4):
        patch = j+1
        totalFisher += fishers['wide'][patch][year]
        totalFisher += fishers['delensing'][patch][year]
        wideFisher += fishers['wide'][patch][year]
        delensingFisher += fishers['delensing'][patch][year]
    fishers['wide']['all'][year] = wideFisher
    fishers['delensing']['all'][year] = delensingFisher
    fishers['S4'][year] = totalFisher
    sigmaNeff = np.sqrt(np.diag(np.linalg.inv(totalFisher)))[indexNeff]
    sigmaNeffwide = np.sqrt(np.diag(np.linalg.inv(wideFisher)))[indexNeff]
    sigmaNeffdelensing = np.sqrt(np.diag(np.linalg.inv(delensingFisher)))[indexNeff]
    print(str(year)+' years, total = ' + str(sigmaNeff))
    print(str(year)+' years, wide only = ' + str(sigmaNeffwide))
    print(str(year)+' years, delensing only = ' + str(sigmaNeffdelensing))
    if addSO:
        sigmaNeffS4andSO = 1/np.sqrt(1/sigmaNeff**2 + 1/sigmaNeffSOLeftover**2)
        print(str(year)+' years, total plus SO = '+str(sigmaNeffS4andSO))
print('fsky wide: '+str(fskywidesansoverlap))
print('fsky deep: '+str(fskydelensing))
print('total fsky from S4: '+str(sum(fskywidesansoverlap)+sum(fskydelensing)))
if addSO:
    print('leftover fsky from SO: '+str(fskySOLeftover))
    print('sigmaNeff from SO: '+str(sigmaNeffSOLeftover))

print('debugging '+str(debug))
print('SO '+str(addSO))

print('writing Fisher matrices')
fisherOutput = open('saveChileS4Fishers'+'_fskyVary'+'.pkl', 'wb')
pickle.dump(fishers, fisherOutput, -1)
fisherOutput.close()

# Plotting
import matplotlib
matplotlib.use('PDF')

years = np.arange(1,11,1)
sigmas = np.zeros(nExps)
sigmasSO = np.zeros(nExps)
for i in range(nExps):
    year = i+1
    sigmas[i] = np.sqrt(np.diag(np.linalg.inv(fishers['S4'][year])))[indexNeff]
    sigmasSO[i] =  1/np.sqrt(1/sigmas[i]**2 + 1/sigmaNeffSOLeftover**2)
plt.plot(years,sigmas,label='S4 only')
plt.plot(years,sigmasSO,label='S4 + ASO')
plt.xlabel('S4 Observation Years')
plt.ylabel('\sigma N_eff')
plt.legend()
plt.savefig('saveChileS4Figure.pdf')

print(fixedParams)
