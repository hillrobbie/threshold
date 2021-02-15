# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:25:46 2020

@author: hillr

-5 january: This script and the functions are just showing that the non boostrap distirbutions are fitting the right chi 2
- ie: the data generating process is working as it is supposed to 
-the IVX can be put in, and test for b1=b2=0 to get the chi2 w/ 2 dof
-the b1=b2 can be tested, regardless of intercept and without ivx and gives a chi2 w/ 1 dof
-also switching from local to unit root to I(1) doesnt have much effect on null 
"""


import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from bigwald_fns_ivx2_simple import *
import datetime
import scipy.stats as sstats
import time 
import statsmodels.tsa.stattools as ts
start_time = time.time()
#np.random.seed(1)
Ts=[200] #1,0.025,0.05,0.075,0.1]#,0.05,0.075,0.1]# ,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
Ns=[1]
arr1=['size']#,'power b1']#,
betas_power=[0.0]
# T=300#panel depth
# C=10#panle width
b=0#burn in  
# Ts=[600] #1,0.025,0.05,0.075,0.1]#,0.05,0.075,0.1]# ,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
# Ns=[1]
# arr1=['size']#,'power b1']#,
# betas_power=[0.05]

threshold=0 #threshold

mc_reps=500#simulation reps


arrs=[Ts,arr1]
col=pd.MultiIndex.from_product(arrs)
result_frame=pd.DataFrame(np.ones((len(Ns),len(betas_power)*len(Ts))),index=Ns,columns=col) #dataframe to fill for results


for T in Ts:
      
      for N in Ns:
            C=N 
            #loop 1 for MC reps
            betas_results_store=[]
            betas_power_store=[]
            # real_data_test_stats=[]
            for beta in betas_power:
                  print('beta coefficent of regime 1:',beta,' Panel width and length:',N,T)
                  #loop 2 for mc repitions
                  beta1=[0 for i in range(0,C)]
                  beta2=[0 for i in range(0,C)]
                  alpha1=[0 for i in range(0,C)]
                  alpha2=[0 for i in range(0,C)]
                  
                  if C==1:
                        beta1[0]=beta
                        alpha1[0]=0.10
                        beta2[0]=0.0
                  else:
                        beta1[0:int(C/2)]=[beta for i in range(0,int(C/2))]


                  mc_results_store=[]
                  for mc_rep in range(0,mc_reps,1):
                        
                        # start intance of innovations
                        inovs=innovations(T=T+b,C=C)
                        
                        #generate threshold
            #            sam=gen_thresh(T+b)
                        sam=inovs.threshvar()
                        
                        
                        #generate the panel data into a dataframe, high persistence 
                        #generate rw, and then IVX it to simulate hihgly persitent but not RW
            #            X=gen_rw(T+b,C,0.5)   #needs to be fixed with some autocorrelation, X isnt necessarily white noise??               
                        X=inovs.rw_predictor()
                        # X=[]
                        # X.append(0)
                        # for i in range(1,T+b):
            
                        #       X.append(0.975*X[i-1]+np.random.normal())
                        # X=np.array(X)
                        # X=pd.DataFrame(X)
            
            
                        
                        #generate the y noise
            #            u=gen_errors(C,T+b,0.0,ma=None ,MA=False).transpose()
                        u=inovs.y_noise()
                        #u=pd.DataFrame(np.random.normal(size=(T+b,C)))
                        
                        #add the threshold varable so that a theshold can be modeled
                        X['gamma']=sam
                        #generate the y, at figure out how to make one or more 
                        #individual inthe panel a threshold model
                        ydf=pd.DataFrame(columns=[str(x) for x in range(0,C)])
                        count=0
                        for col in range(0,len(X.columns)-1):
                              x_s=X.iloc[:,col] #make list for speed?
                              u_s=u.iloc[:,col]
                              y=np.zeros_like(x_s)
                              for i in range(0,len(x_s)): #probably a faster way
                                    if X['gamma'].iloc[i]  <= threshold: # the u_s are from AR(1) factors: built in autocorr
                                          y[i]=alpha1[count]+beta1[count]*x_s.iloc[i]+u_s[i]#+sc[count]*u_s[i-2] 
                                    elif X['gamma'].iloc[i]  > threshold:      
                                          y[i]=alpha2[count]+beta2[count]*x_s.iloc[i]+u_s[i]#+sc[count]*u_s[i-2]
                                 
                              ydf[str(count)]=y     
                              count=count+1     
                        
                        #we do IVX , as per gonzalo and pitarakis
                        # XVX=ivx(X.iloc[:,:-1],delta=0.95,c=1,adfresults=False,plot=False)
                        # XVX=XVX.drop(0)
                        
                        paneldata=pd.merge(ydf,X.iloc[:,:-1], left_index=True,right_index=True)     
                        #fix the col names
                   
                        # paneldata=pd.merge(paneldata, XVX,left_index=True,right_index=True)  
                        #add threshold to dataframe so it can be sorted by gamma before walds
                        
                        paneldata.columns=map(str,(range(0,2*C)))
                        paneldata['gamma']=    X['gamma'] 
                         #cut out the burn in 
                        paneldata=paneldata.iloc[b:,:]
            
                        #Get betas and residuals of null regression for the bootstrap.
                        # store them in lists
                        resids=pd.DataFrame()
                        null_betas=list()
                        thresh=np.array(paneldata['gamma'],ndmin=2).T
                        max_gamma_index=list()
                        maxwalds=list()
                        for i in range(0,C):
                              y=np.array(paneldata[str(i)],ndmin=2).T
                              x=np.array(paneldata[str(i+C)],ndmin=2).T
                              # iv=np.array(paneldata[str(i+2*C)] ,ndmin=2).T
                              #residuals and null betas for bootstrap
                              # e_4boot,null_beta=beta_null(y, x,iv )
                              # resids[str(i)]=e_4boot.reshape((T-1))
                              # null_betas.append(null_beta)
                              #min SSR index number for gamma
                              max_gamma_index.append(sumsqe(y, thresh,p=0.15))

                              # mgi,drop=supf(y,x,thresh)
                              # mgi=np.random.randint(0.15*T,0.85*T)
                              # max_gamma_index.append(mgi)
                        
                        # max_gamma_index=list()     
                        # for i in range(0,C):
                        #       max_gamma_index.append(np.random.uniform(int(T*0.2),int(T*0.8)))
                        # sup wald stat from the sorted data.       
                        
                                   
                        #sort once so it doesnt have to be done once per column?
                        paneldata_sorted=paneldata.sort_values('gamma')
                        paneldata_sorted=paneldata_sorted.reset_index(drop=True)
                                 
                        thresh=np.array(paneldata_sorted['gamma'])
                        for i in range(0,C):
                              y =np.array(paneldata_sorted[str(i)])
                              x =np.array(paneldata_sorted[str(i+C)])
                              # iv=np.array(paneldata_sorted[str(i+2*C)] )
                              # wald=supf_ivx(y, x, iv,thresh, max_gamma_index[i])
                              mgi=max_gamma_index[i]
                              wald=waldsplit(y,x,thresh, mgi)
                              maxwalds.append(wald)
                                               
                        #get average of max wald stats per country
                        mc_results_store.append(np.average(maxwalds))
                        #real_data_test_stats.append(np.max(maxwalds))
                        print('percent mc done: ', 100*mc_rep/mc_reps)
                        
                  #z=np.abs(np.sqrt(C)*(np.array(mc_results_store)-2)/np.sqrt(2*2))
                  z=np.array(mc_results_store)
                  test_stat_dist=sp.stats.norm.cdf(z,0,1)
                  
                  if beta==0:
                         #sstats.chi2.ppf(0.95,1)
                         z.sort()
                         adjcrit=z[int(0.95*len(z))]
                         betas_results_store.append( adjcrit)

                  else:
                        one=sum(1 for item in z if item>=(adjcrit))/len(z)
                        print("percent of z stats greater than adjcrit " , one)
                        betas_results_store.append(one)
                  
                                                      
            result_frame.loc[C,T]=betas_results_store
            #results_array=pd.DataFrame(betas_results_store,index=betas_power)
            
            
           
      
##size adjusted power 


#print how much time
print("--- %s seconds ---" % (time.time() - start_time))
totaltime=time.time() - start_time
#save file with results     
dt=datetime.datetime.now()
titlestring='results_ivx_CLT'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
with open(titlestring, 'w') as filehandle:
    filehandle.write(result_frame.to_string())     
    filehandle.write(str('\n time '+str(totaltime)))

      
# import statsmodels.api as sm
# from scipy import stats
# z=np.array(mc_results_store)
# z=np.sqrt(C/(2*2))*(z-2)
# kde = sm.nonparametric.KDEUnivariate(z)
# kde.fit()
# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(111)
# ax.hist(z, bins=20, density=True, label='Histogram from samples',
#         zorder=5, edgecolor='k', alpha=0.5)

# # Plot the KDE as fitted using the default arguments
# ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)

# # Plot the true distribution

# true_values = stats.norm.pdf(kde.support,0,1)      
# ax.plot(kde.support, true_values, lw=3, label='True distribution', zorder=15)

# # Plot the samples
# ax.scatter(z, np.abs(np.random.randn(len(z)))/100,
#             marker='x', color='red', zorder=20, label='Samples', alpha=0.5)

# ax.legend(loc='best')
# ax.grid(True, zorder=-5)


import scipy as sy
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF

a=np.array(mc_results_store)
#a=np.sqrt(5/4)*(a-2)
ecdf=ECDF(a)

fig, ax = plt.subplots(1, 1)

aa=min(a)
bb=max(a)
x = np.linspace(0,   bb, 100)
ax.plot(ecdf.x,ecdf.y)
ax.plot(x, sstats.chi2.cdf(x, 1), label='chi2 pdf')

