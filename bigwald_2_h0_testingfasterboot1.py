# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:25:46 2020

@author: hillr

Feb 15 2021
-This script runs that bootstrap version of the sup F test for a threshold in a predicitive regression
-The DGP uses the class 'innovations' from the bigwald_fns file
-The distribution of the Bootstrap supremum wald test are plotted at the end, note this is not a chi2 distribution
since they are maximumal squared browian bridge distributed. This is a subtle difference from the max wald statistic
where the limiting distribution is chi2

The bootstrap uses a cumulative sum to create the x local-to-unit root predictor variable. Details are in the rbb function in 
bigwald_fns
-Important to check the block size for the RBB, when adding serial correlation through the innovations class the block size makes
a big difference to test size


"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from bigwald_fns_ivx2_simple import *
import datetime

import time 
start_time = time.time()
# np.random.seed(3)
#set the nimber of observations per panel unit (can make a list)
Ts=[400] #[200,400,800] 
#set the nimber of panel units
Ns=[2] #[6,10,26]
#set predictor coefficent (beta1 on the predictor in regime when threshodl variables is less than threshold)
betas_power=[0.025] #[0.0,0.025,0.05]
# set where on the threshold variable to divide the sample
threshold=0 #threshold

mc_reps=10#simulation reps

bootreps=100#bootstrap repetition
# set so the results data frame knows what to name the columns, 
arr1=['size'] #['size','power b1', 'power b2']

arrs=[Ts,arr1]
col=pd.MultiIndex.from_product(arrs)
result_frame=pd.DataFrame(np.ones((len(Ns),len(betas_power)*len(Ts))),index=Ns,columns=col) #dataframe to fill for results

counts=mc_reps*len(Ns)*len(betas_power)*len(Ts)
counter=1
for T in Ts:

      for N in Ns:
            C=N 
            betas_results_store=[]
            betas_power_store=[]
            for beta in betas_power:
                  print('beta coefficent of regime 1',beta)
                  #loop 2 for mc repitions
                  beta1=[0.0 for i in range(0,C)]
                  beta2=[0.0 for i in range(0,C)]
                  alpha1=[0.0 for i in range(0,C)]
                  alpha2=[0.0 for i in range(0,C)] 
                  if C==1:
                        beta1[0] =beta
                        alpha1[0]=0.0
                        beta2[0] =0.0
                  else:
                        beta1[0:int(C/2)]=[beta for i in range(0,int(C/2))]
                  mc_results_store=[]
                  for mc_rep in range(0,mc_reps,1):
                        
                        # start intance of innovations
                        inovs=innovations(T=T,C=C)
                        
                        #generate threshold
            #            sam=gen_thresh(T+b)
                        sam=inovs.threshvar()
                        
                        
                        #generate the panel data into a dataframe, high persistence 
                        #generate rw, and then IVX it to simulate hihgly persitent but not RW
            #            X=gen_rw(T+b,C,0.5)   #needs to be fixed with some autocorrelation, X isnt necessarily white noise??               
                        X=inovs.rw_predictor()
            
                        
                        #generate the y noise
            #            u=gen_errors(C,T+b,0.0,ma=None ,MA=False).transpose()
                        u=inovs.y_noise()
                        
                        
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
                        
                        # #we do IVX , as per gonzalo and pitarakis
                        # XVX=ivx(X.iloc[:,:-1],delta=0.95,c=10,adfresults=False,plot=False)
                        # XVX=XVX.drop(0)
                        
                        paneldata=pd.merge(ydf,X.iloc[:,:-1], left_index=True,right_index=True)     
                        #fix the col names
                   
                        # paneldata=pd.merge(paneldata, XVX,left_index=True,right_index=True)  
                        # #add threshold to dataframe so it can be sorted by gamma before walds
                        # change 2 to 3 for IV
                        paneldata.columns=map(str,(range(0,2*C)))
                        paneldata['gamma']=    X['gamma'] 
                         #cut out the burn in 
                        paneldata=paneldata.iloc[0:,:]
                        
                        #before sorting, we need starting values for the RBB bootstrap since 
                        #x is I(1), and differences in X in order to block bootstrap
                        startingx=paneldata.iloc[0:1,C:2*C]
                        x_disturb=paneldata.iloc[:,C:2*C].diff().fillna(0).reset_index(drop=True)
                        
                        #Get betas and residuals of null regression for the bootstrap.
                        #store them in lists
                        resids=pd.DataFrame()
                        null_betas=list()
                        thresh=np.array(paneldata['gamma'],ndmin=2).T
                        max_gamma_index=list()
                        maxwalds=list()
                        xbetas=list()
                        for i in range(0,C):
                              y=np.array(paneldata[str(i)],ndmin=2).T
                              x=np.array(paneldata[str(i+C)],ndmin=2).T
                              # iv=np.array(paneldata[str(i+2*C)] ,ndmin=2).T
                              # #residuals and null betas for bootstrap
                              e_4boot,null_beta=beta_null(y, x,x )
                              resids[str(i)]=e_4boot.reshape((T))
                              null_betas.append(null_beta)
                              # drop,xbeta=ar1(x.T)
                              # xbetas.append(xbeta)
                              # #min SSR index number for gamma
                              mgi,drop=supf(y,x,thresh)
                              max_gamma_index.append(mgi)
                              maxwalds.append(max(drop))
                              # max_gamma_index.append(sumsqe(y, thresh,p=0.15))
                        
                        # # sup wald stat from the sorted data.                      
                        # #sort once so it doesnt have to be done once per column?
                        # paneldata_sorted=paneldata.sort_values('gamma')
                                 
                        # thresh=np.array(paneldata_sorted['gamma'])
                        # for i in range(0,C):
                        #       y =np.array(paneldata_sorted[str(i)])
                        #       x =np.array(paneldata_sorted[str(i+C)])
                        #       iv=np.array(paneldata_sorted[str(i+2*C)] )
                        #       wald=supf_ivx(y, x, iv,thresh, max_gamma_index[i])
                        #       maxwalds.append(wald)
                                               
                        #get average of max wald stats per country
                        real_data_test_stat=np.average(maxwalds)
                        
            
                        t1=time.time()
                        ###############################################PART 2 Bootstrap
                        bootrep=0
                        bootstrap_stats=[]
                        print('starting bootstrap')
                        while bootrep<bootreps:
                                #   # generate boot u                
                              # e_boot,x_boot=rbb(resids,x_disturb,startingx,4)  
                              
                              # #generate boot thresh, this is just white noise 
                              # boot_gamma=gen_thresh(T)
                                
                              
                              # #generate the y, (and figure out how to make one or more 
                              # #individual inthe panel a threshold model) under ull of no threshold
                              # ydf_boot=pd.DataFrame(columns=[str(x) for x in range(0,C)])
                              # count=0
                              # for col in range(0,len(x_boot.columns)):
                              #   x_s=x_boot.iloc[:,col] #
                              #   u_s=e_boot.iloc[:,col]
                              #   y=null_betas[col][0]*np.array(x_s)+np.array(u_s)
                              #   # y=np.zeros_like(x_s)
                              #   # for i in range(0,len(u_s)):#lag y doesnt makes sense
                              #   #        y[i]=null_betas[count][0]*x_s.iloc[i]+u_s.iloc[i]
                              #   ydf_boot[str(count)] =y     
                              #   count=count+1     
                              
                            
                              # # IVX for bootstrapped X's
                              # x_boot=x_boot.reset_index(drop=True)
                              # # XVX=ivx(x_boot,delta=0.95,c=10,adfresults=False,plot=False)
                                                   
                              # paneldata_boot=pd.merge(ydf_boot, x_boot, left_index=True,right_index=True)
                              # # paneldata_boot=pd.merge(paneldata_boot, XVX, left_index=True,right_index=True) 
                              # paneldata_boot.columns=map(str,(range(0,2*C)))
                              # paneldata_boot['gamma']=boot_gamma
                              # paneldata_boot_sorted=paneldata_boot.sort_values('gamma')
                              # maxwalds_boot=list()
                            
                              # thresh=np.array(paneldata_boot['gamma'] ,ndmin=2).T
                              # #sorted thresh for wald split with mgi
                              # thresh_sort=np.array(paneldata_boot_sorted['gamma'] ,ndmin=2).T
                              # for i in range(0,C):
                              #         ## delete
                              #   y=np.array(paneldata_boot.iloc[:,i],ndmin=2).T
                              #   x=np.array(paneldata_boot.iloc[:,i+C],ndmin=2).T
                              #   # iv=np.array(paneldata_boot.iloc[:,i+2*C],ndmin=2).T
                              #   mgi,wald_boot=supf(y, x, thresh)
                              #   wald_boot=max(wald_boot)
                              
                              #   # mgi=max_gamma_index[i]
                              #   # wald=waldsplit(y,x,thresh_sort, mgi)
                              #   # maxwalds_boot.append(wald)
                              
                              #   # maxwalds_boot.append(wald_boot)
                              #   # mgi=max_gamma_index[i]
                              #   # mgi=np.random.randint(0.2*T,0.8*T)
                              #   # y =np.array(paneldata_boot_sorted.iloc[:,i])
                              #   # x =np.array(paneldata_boot_sorted.iloc[:,i+C])
                              #   # iv=np.array(paneldata_boot_sorted.iloc[:,i+2*C] )
                              #   # wald_boot=supf_ivx(y, x, iv ,thresh_sort,mgi)
                              
                              #   #maxwalds_boot.append(wald_boot)
                              # bootstrap_stats.append(np.average(wald_boot))
                              # bootrep+=1

                                # ############## fast bootstrap
                                e_boot,x_boot=rbb(resids,x_disturb,startingx,4)  
                              
                      
                                                            
                                #generate the y, (and figure out how to make one or more 
                                #individual inthe panel a threshold model) under ull of no threshold
                              
                                count=0
                                #generate boot thresh, this is just white noise                             
                                thresh=np.array(gen_thresh(T) ,ndmin=2).T
                                for col in range(0,len(x_boot.columns)):
                                      x_s=np.array(x_boot.iloc[:,col],ndmin=2).T#
                                      u_s=np.array(e_boot.iloc[:,col],ndmin=2).T
                                      y=null_betas[col][0]*np.array(x_s)+np.array(u_s)
                                      # y=np.zeros_like(x_s)
                                      # for i in range(0,len(u_s)):#lag y doesnt makes sense
                                      #        y[i]=null_betas[count][0]*x_s.iloc[i]+u_s.iloc[i]
                                      # ydf_boot[str(count)] =y     
                                      mgi,wald_boot=supf(y, x_s, thresh)
                                      wald_boot=max(wald_boot)
                                    
                                    
                                      count=count+1     
                                
                    
                                    
                                #maxs= [max(p) for p in maxwalds] 
                                bootstrap_stats.append(np.average(wald_boot))
                                bootrep+=1
                                # ############## 
                                
                              #end bootstrap
                        t2=time.time()
                        print('time of bootstrap',t2-t1 )
                        
                        a = np.array(bootstrap_stats)     
                        quantile=(a<real_data_test_stat).astype(int).mean()
                        mc_results_store.append(quantile)
                        print('quantile from bootstrap ',quantile)
                        print('percent mc done: ', 100*counter/counts)
                        counter=counter+1
                        #end MC loop
                  #power from the round of MC reps
                  power=len([1 for i in mc_results_store if i >= 0.95])/mc_reps
                  print('power of test for ', beta, ' is: ', power )
                  betas_results_store.append(mc_results_store)
                  betas_power_store.append(power)
                  #end betas loop

            
                 
            result_frame.loc[C,T]=betas_power_store

            
            
#results_array=pd.DataFrame({'beta of unit 1 ': betas_power, 'power of test': betas_power_store})
results_array=pd.DataFrame(np.array(betas_results_store).T,columns=betas_power,index=range(1,mc_reps+1))
#results_out=np.mean(results_array,axis=1)
#print('mean bootstrap results over MC reps ',np.mean(results_array,axis=1))


#print the individual monte carlo results (each iteration of the simulation)
import datetime
dt=datetime.datetime.now()
titlestring='results_out_jit_'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
# with open(titlestring, 'w') as filehandle:
#     filehandle.write(results_array.to_string())   
#     filehandle.write(str('\n panel size NT '+str(C)+' '+str(T)))
#     filehandle.write(str('\n power '+str(betas_power_store)))
          


#print the overall monte carlo results (average from each iteration)
print("--- %s seconds ---" % (time.time() - start_time))
totaltime=time.time() - start_time
#save file with results     
dt=datetime.datetime.now()
titlestring='results_ivx_CLT'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
# with open(titlestring, 'w') as filehandle:
#     filehandle.write(result_frame.to_string())     
#     filehandle.write(str('\n time '+str(totaltime)))


#plot the distribution of the bootstrap statistics
import scipy as sy
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as sstats
# a=np.array(mc_results_store)
#a=np.sqrt(5/4)*(a-2)
ecdf=ECDF(a)

fig, ax = plt.subplots(1, 1)

aa=min(a)
bb=max(a)
x = np.linspace(0,   bb, 100)
ax.plot(x, sstats.chi2.cdf(x, 1), label='chi2 pdf')
ax.plot(ecdf.x,ecdf.y)
