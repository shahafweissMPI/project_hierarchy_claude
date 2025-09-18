"""
https://github.com/davidespalla/code_ahv_speed_cells

https://github.com/davidespalla/code_ahv_speed_cells/blob/main/.ipynb_checkpoints/fit_GLM-checkpoint.ipynb

"""
#%pylab inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats 
#import seaborn as sns
from sklearn.model_selection import train_test_split
#sns.set_style("white")
def StateVector(correlate,minbin,maxbin,nbins):
    stateVec=np.zeros((len(correlate),nbins))
    indexVec=np.digitize(correlate,np.linspace(minbin,maxbin,nbins))
    for i in range(len(correlate)):
        if indexVec[i] < nbins:
            stateVec[i][indexVec[i]]=1
        else :
            stateVec[i][:]=np.nan
    return stateVec
## Angular Velocity
#trial data
AngularVelocity=pd.read_csv("../TrialData/filteredAngularVelocity.csv")
trialMetadata=pd.read_csv("../TrialData/metadata.csv",index_col=0)
#cells data
rate=pd.read_csv("../CellData/firingRate.csv")
cellMetadata=pd.read_csv("../CellData/metadata.csv",index_col=0)
AngularVelocity.columns=trialMetadata["trialIndex"].values
## Model fitting
fittedModels=[]
pvalues=[]
llfs=[]
llnulls=[]
significance=[]
tstats=[]
for i in range(len(rate.T)):
    trialId=cellMetadata["trialIndex"].values[i]
    if cellMetadata["include"].values[i]==1:
        x=StateVector(AngularVelocity[trialId].values,-3,3,10)
        y=rate.T.values[i]
        #10-fold bootstrapping
        logls=[]
        null_logls=[]
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
            #TRAINING
            bad = ~np.logical_or(np.isnan(X_train).any(axis=1), np.isnan(y_train))
            a=np.compress(bad, X_train,axis=0)  
            b=np.compress(bad, y_train) 
            poisson_model=sm.GLM(b,a, family=sm.families.Poisson())
            fitted_model=poisson_model.fit_regularized(alpha=0.04,refit=True)
            logls.append(fitted_model.llf)
            null_logls.append(fitted_model.llnull)
        #pvalue=(stats.wilcoxon(logls,null_logls)[1]/2) #to have a 1-sided test
        llfs.append(logls)
        llnulls.append(null_logls)
        tstat=stats.ttest_ind(logls,null_logls)[0]
        pvalue=stats.ttest_ind(logls,null_logls)[1]
        tstats.append(tstat)
        pvalues.append(pvalue)
        if pvalue < 0.05 and tstat>0:
            significance.append(1)
        else:
            significance.append(0)
        #fit all data for tuning curve plotting    
        bad = ~np.logical_or(np.isnan(x).any(axis=1), np.isnan(y))
        a=np.compress(bad,x,axis=0)  
        b=np.compress(bad,y) 
        tot_model=sm.GLM(b,a, family=sm.families.Poisson())
        tot_fitted_model=tot_model.fit_regularized(alpha=0.04,refit=True)
        fittedModels.append(fitted_model)
    else:
        fittedModels.append(np.nan)
        tstats.append(np.nan)
        pvalues.append(np.nan)
        significance.append(np.nan)
        llfs.append(np.nan)
        llnulls.append(np.nan)
    if i %1==0:
        print("fitted model on cell:"+str(i+1))
for i in range(len(fittedModels)):
    if cellMetadata["include"].values[i]==1:
        fittedModels[i].save("FittedModels/angularVelocityBin3Reg04/modelCell"+str(i))
np.save("GLMangularVelocitySignificance.npy",significance)
np.save("GLMangularVelocityPvalues.npy",pvalues)
np.save("GLMangularVelocityTstats.npy",tstats)
## Speed
#trial data
speed=pd.read_csv("../TrialData/filteredSpeed.csv")
trialMetadata=pd.read_csv("../TrialData/metadata.csv",index_col=0)
#cells data
rate=pd.read_csv("../CellData/firingRate.csv")
cellMetadata=pd.read_csv("../CellData/metadata.csv",index_col=0)
speed.columns=trialMetadata["trialIndex"].values
## Model fitting
fittedModels=[]
pvalues=[]
llfs=[]
llnulls=[]
significance=[]
tstats=[]
for i in range(len(rate.T)):
    trialId=cellMetadata["trialIndex"].values[i]
    if cellMetadata["include"].values[i]==1:
        x=StateVector(speed[trialId].values,2,50,10)
        y=rate.T.values[i]
        #10-fold bootstrapping
        logls=[]
        null_logls=[]
        for j in range(10):
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
            #TRAINING
            bad = ~np.logical_or(np.isnan(X_train).any(axis=1), np.isnan(y_train))
            a=np.compress(bad, X_train,axis=0)  
            b=np.compress(bad, y_train) 
            poisson_model=sm.GLM(b,a, family=sm.families.Poisson())
            fitted_model=poisson_model.fit_regularized(alpha=0.04,refit=True)
            logls.append(fitted_model.llf)
            null_logls.append(fitted_model.llnull)
        #pvalue=(stats.wilcoxon(logls,null_logls)[1]/2) #to have a 1-sided test
        llfs.append(logls)
        llnulls.append(null_logls)
        tstat=stats.ttest_ind(logls,null_logls)[0]
        pvalue=stats.ttest_ind(logls,null_logls)[1]
        tstats.append(tstat)
        pvalues.append(pvalue)
        if pvalue < 0.05 and tstat>0:
            significance.append(1)
        else:
            significance.append(0)
        #fit all data for tuning curve plotting    
        bad = ~np.logical_or(np.isnan(x).any(axis=1), np.isnan(y))
        a=np.compress(bad,x,axis=0)  
        b=np.compress(bad,y) 
        tot_model=sm.GLM(b,a, family=sm.families.Poisson())
        tot_fitted_model=tot_model.fit_regularized(alpha=0.04,refit=True)
        fittedModels.append(fitted_model)
    else:
        fittedModels.append(np.nan)
        pvalues.append(np.nan)
        tstats.append(np.nan)
        significance.append(np.nan)
        llfs.append(np.nan)
        llnulls.append(np.nan)
    if i %1==0:
        print("fitted model on cell:"+str(i+1))
for i in range(len(fittedModels)):
    if cellMetadata["include"].values[i]==1:
        fittedModels[i].save("FittedModels/speed/modelCell"+str(i))
np.save("GLMspeedSignificance.npy",significance)
np.save("GLMspeedPvalues.npy",pvalues)
np.save("GLMspeedTstats.npy",tstats)