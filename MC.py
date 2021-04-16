
############################################################################
############################################################################
### FUNCTIONS 
############################################################################
############################################################################
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from numba import jit
import statsmodels.api as sm



def gen_thresh(T,mean = 0,std = 1 ):
      samples = np.random.normal(mean, std, size=T)
      return samples


            
            
#


@jit()
def supf(y, x, thresh , p=0.15 ):
      T = y.shape[0]
      r = np.floor(np.array([T * p, T * (1-p)]))
      r = np.arange(r[0], r[1] + 1, dtype=np.int32)
      ones=np.ones((T,1))
      x_=np.hstack((ones, x))
      mi=np.linalg.inv(x_.T@x_)
      e=y-x_@mi@(x_.T@y)
      ee=e.T@e/(T-2)
      #null_betas=mi@(x_.T@y)
      
      #testing a hypothesis of either b1=b2 or b2=0 depending on how the 
      #threshold equation looks.
      sortedframe=np.hstack((y, x, thresh))
      sortedframe=sortedframe[sortedframe[:,2].argsort()]
      y_=sortedframe[:,0:1] 
      x_=sortedframe[:,1:2]
      x_=np.hstack((ones,x_))
      
      # mi=np.linalg.inv(x_.T@x_)
      # e=y_-x_@mi@x_.T@y_
      # ee=e.T@e/(T-2)
      # xe=x_*e
      # cxe0=xe[:,0].cumsum()
      # cxe1=xe[:,1].cumsum()
      # cxe=np.vstack((cxe0,cxe1))
      # R = np.array([[1,0],[0,1]])
      walds = np.zeros((T,1))
      Xb=np.zeros((T, 2))
      Xt=np.zeros((T, 2))
      invX=np.linalg.inv(x_.T@x_)
      for t in r:  
            # Xb=x_[0:t,:]
            # yb=y_[0:t,:]
            # bb=np.linalg.inv(Xb.T@Xb)@Xb.T@yb
            
            # Xt=x_[t+1:,:]
            # yt=y_[t+1:,:]
            # bt=np.linalg.inv(Xt.T@Xt)@Xt.T@yt
            
            # walds[t]=(bb-bt).T@(Xb.T@Xb-Xb.T@Xb@invX@Xb.T@Xb)@(bb-bt)/ee
            Xb=np.zeros((T, 2))
            Xt=np.zeros((T, 2))
            Xb[:t,:]=x_[:t,:]
            Xt[t:,:]=x_[t:,:]
            bb=np.linalg.inv(Xb.T@Xb)@Xb.T@y_
            bt=np.linalg.inv(Xt.T@Xt)@Xt.T@y_
            sig=y_- Xb@bb - Xt@bt
            sig2=sig.T@sig/(T)
            walds[t,:]=(bb-bt).T@(Xb.T@Xb-Xb.T@Xb@invX@Xb.T@Xb)@(bb-bt)/sig2
            
            Z=np.hstack((Xb,Xt))
            R=np.array([[1.0,0.0,-1.0,0.0],[0.0,1.0,0.0,-1.0]])
            R=np.array([[0.0,1.0,0.0,-1.0]])
            theta=np.vstack((bb,bt))
            walds[t,:]=theta.T@R.T@np.linalg.inv(R@np.linalg.inv(Z.T@Z)@R.T)@R@theta/ sig2
            

            # X1[:t,:]=x_[:t,:]
            # left=e.T@X1-e.T@x_@ invX@X1.T@X1
            # right=left.T
            # middle=np.linalg.inv(X1.T@X1-X1.T@X1@invX@X1.T@X1)/ee
            # wald=left@middle@right
            # walds[t]=wald
   
            # S=cxe.T[t,:]
            # Minv = np.linalg.inv(x_[:t,:].T@x_[:t,:])
            # vv=xe[:t,:].T@xe[:t,:]
            # Vstar=Minv@vv@Minv
            # RVRinv=np.linalg.inv(R.T@Minv@vv@Minv@R)
            # walds.append(S.T@Minv@R@RVRinv@R.T@Minv@S)
            # Parameters and errors before the break
      max_gamma_index=walds.argmax() #+r[0]
      
      return  max_gamma_index, walds
#


@jit()
def supf_nosort(y, x,p=0.15 ):
      T = y.shape[0]
      r = np.floor(np.array([T * p, T * (1-p)]))
      r = np.arange(r[0], r[1] + 1, dtype=np.int32)
      ones=np.ones((T,1))
      x_=np.hstack((ones, x))
      mi=np.linalg.inv(x_.T@x_)
      e=y-x_@mi@(x_.T@y)
      ee=e.T@e/(T-2)
      #null_betas=mi@(x_.T@y)
      
      #testing a hypothesis of either b1=b2 or b2=0 depending on how the 
      #threshold equation looks.
      sortedframe=np.hstack((y, x))
      y_=sortedframe[:,0:1] 
      x_=sortedframe[:,1:2]
      x_=np.hstack((ones,x_))
      
      # mi=np.linalg.inv(x_.T@x_)
      # e=y_-x_@mi@x_.T@y_
      # ee=e.T@e/(T-2)
      # xe=x_*e
      # cxe0=xe[:,0].cumsum()
      # cxe1=xe[:,1].cumsum()
      # cxe=np.vstack((cxe0,cxe1))
      # R = np.array([[1,0],[0,1]])
      walds = np.zeros((T,1))
      Xb=np.zeros((T, 2))
      Xt=np.zeros((T, 2))
      invX=np.linalg.inv(x_.T@x_)
      for t in r:  
            # Xb=x_[0:t,:]
            # yb=y_[0:t,:]
            # bb=np.linalg.inv(Xb.T@Xb)@Xb.T@yb
            
            # Xt=x_[t+1:,:]
            # yt=y_[t+1:,:]
            # bt=np.linalg.inv(Xt.T@Xt)@Xt.T@yt
            
            # walds[t]=(bb-bt).T@(Xb.T@Xb-Xb.T@Xb@invX@Xb.T@Xb)@(bb-bt)/ee
            Xb=np.zeros((T, 2))
            Xt=np.zeros((T, 2))
            Xb[:t,:]=x_[:t,:]
            Xt[t:,:]=x_[t:,:]
            bb=np.linalg.inv(Xb.T@Xb)@Xb.T@y_
            bt=np.linalg.inv(Xt.T@Xt)@Xt.T@y_
            sig=y_- Xb@bb - Xt@bt
            sig2=sig.T@sig/(T)
            # walds[t,:]=(bb-bt).T@(Xb.T@Xb-Xb.T@Xb@invX@Xb.T@Xb)@(bb-bt)/sig2
            
            Z=np.hstack((Xb,Xt))
            R=np.array([[1.0,0.0,-1.0,0.0],[0.0,1.0,0.0,-1.0]])
            R=np.array([[0.0,1.0,0.0,-1.0]])
            theta=np.vstack((bb,bt))
            walds[t,:]=theta.T@R.T@np.linalg.inv(R@np.linalg.inv(Z.T@Z)@R.T)@R@theta/ sig2
            
      max_gamma_index=walds.argmax() #+r[0]
      
      return  max_gamma_index, walds

   
@jit()
def waldsplit(y,x,thresh,mgi):
      T = y.shape[0]
     
      Q1=np.ones((T,1))
      Q1[mgi :]=0
      Q2=np.ones((T,1))
      Q2[: mgi]=0
      X1=Q1*x.reshape((T,1))
      X2=Q2*x.reshape((T,1))
      X=x.reshape((T,1))
      Q=np.hstack((Q1,Q2))
  
      X=np.hstack((X1,X2))

      M=np.identity(Q.shape[0])-Q@np.linalg.inv(Q.T@Q)@Q.T
     
      ystar=M@y
      Xstar=M@X
      
      b=np.linalg.inv(Xstar.T@Xstar)
      b=b@Xstar.T@ystar
     
      pred=Xstar@b
     
      resid=ystar-pred
      sigma2=np.sum(resid**2)/(resid.shape[0]-1)
       
      # forcorr=sm.OLS(x[1:],sm.add_constant(x[0:-1]),missing='drop')
      # u=forcorr.fit().resid
      u,drop=ar1(x)
      rho=np.corrcoef(resid[1:],u,False)[1,0] 
     
      R=np.array([[1.0,0.0],[0.0,1.0]])
      R=np.array([[1.0,-1.0]])
      
    
      varbhat=np.linalg.inv(Xstar.T@Xstar)*sigma2
      max_waldivc=b.T@R.T@np.linalg.inv(R@varbhat@R.T)@R@b
      return max_waldivc
   
@jit()
def ar1(x):
      x=x.reshape(x.shape[0],1)
      xl=x[:-1,:]
      xf=x[1:,:]
      X=np.hstack((np.ones(xl.shape),xl))
      b=np.linalg.inv(xl.T@xl)@xl.T@xf
      res=xf-xl@b.T
      return res,b


@jit()
def beta_null(y, x,iv): 
      #create stared (demeaned variables as per GP)

       T = y.shape[0]
       Q=np.ones((T,1))
       inv=np.linalg.inv(Q.T@Q)@Q.T
       M=np.identity(Q.shape[0])-np.dot(Q,inv)
       ystar=y
       Hstar=x
       Xstar=x
       #get B_iv
       b_iv=np.linalg.inv(Hstar.T@Xstar)
       b_iv=b_iv@Hstar.T@ystar   
       #resids for bootstrap
       pred=Xstar@b_iv
       resids_iv_null=ystar-pred
       
       return  resids_iv_null,b_iv[0][0]
      
      


def null_resids(y, x ): 
      #this tests for predictability in the sense of GP 2017
      #find lowest SSR 
      T = y.shape[0]
      y_=np.array(y)
      x_=np.array(x)
      x_=np.vstack([np.ones_like(x), np.array(x)])
      mi=np.linalg.inv(x_@x_.T)
      null_betas=mi@(x_@y_)
      e_4boot=y_-x_.T@null_betas      
      return e_4boot,null_betas

def rbb(resids,x_disturb,startingx,block_size,xbetas):
#      x1=startingx[c] 
      #e is residual list from supwald fn null regression
      e=resids
      num_items=len(e)
      
      #have to block demean properly 
      if block_size==1:
            brange=list(range(1,num_items-block_size+2))
            mainmean=np.mean(e)
            max_index = num_items
            num_blocks = (num_items )// block_size
      else:
            brange=list(range(1,num_items-block_size+1))
            mainmean=np.mean([np.mean(e.iloc[list(range(i,i+block_size)),:]) for i in brange],0)
            max_index = num_items - block_size
            num_blocks = (num_items-1 )// block_size
            
            
      e_=e-mainmean #demeaning residulas
      
      indices = np.random.randint(max_index, size=num_blocks+1)
      indices = indices[:, None] + np.arange(block_size)
      indices = indices.flatten()
      
      booted_e=e_.iloc[indices,:]
      booted_x=x_disturb.iloc[indices[1:],:]
      startingx.columns=booted_x.columns
      booted_x=pd.concat([startingx,booted_x])
      ####I(1) predictors
      booted_x=np.cumsum(booted_x)
      #### AR(1) estimated preditor 
      # for t in range(1,T):
      #        booted_x.iloc[t,:]=booted_x.iloc[t-1,:]*xbetas+ booted_x.iloc[t,:]
      return booted_e,booted_x



class innovations:
      
      def __init__(self, T=1000,
                         C=5,
                         factor_means = np.array([0.0, 0.0, 0.0]),
                         factor_covar = np.array([
                                            [ 1,  0.4,  0.4],
                                            [ 0.4,  1,  0.4],
                                            [ 0.4,  0.4,  1]
                                        ]),
                         factor_phi=0.6):
                         # lax=, #[-0.0,-0.0,0.0,0.0,0.0],
                         # lay= #[0.1,0.2,0.15,0.25,0.5]):
            self.T, self.C, self.factor_means = T,C,factor_means
            self.factor_covar, self.actor_phi = factor_covar, factor_phi
            self.lax = 0.2*np.ones(C)  #np.random.uniform(-0.25,0.25,C)# 0.0*np.ones(C)  #0.8*np.ones(C)  
            self.lay = 0.2*np.ones(C)  # np.random.uniform(-0.25,0.25,C)# 0.0*np.ones(C)  #0.8*np.ones(C)
            self.non_cen_p=1-1/T
            

            num_samples =self.T #num_samples =T
            
            # The desired mean values of the sample.
            mu = self.factor_means
            
            # The desired covariance matrix.
            r = self.factor_covar
            
            # Generate the random samples.
            y = np.random.multivariate_normal(mu, r, size=num_samples)
            
            #plt.plot(y)
            #plt.show()
            
            #turn fctors into AR 1 series            
            fac=np.zeros_like(y)
            fac[0,:]=y[0,:]
            for i in range(1,T):
                fac[i]=y[i,:]+factor_phi*fac[i-1,:]  

            fac=pd.DataFrame(fac) # gives 3 vectors of factors 
      #     fac.plot()
            self.fac=fac
      
      def threshvar(self):
      #gen thresh  ## thresh variable can be just the first factor
            thresh=self.fac.iloc[:,0]
            return thresh
      
      #gen randomwalk x for predictors  
      def rw_predictor(self):
            y2=self.fac.iloc[:,1] #second factor  y2=fac.iloc[:,1] 
            lax=self.lax #the coefficents for each x ie: x_i=lax_i fac
            non_cen_p=self.non_cen_p
            T,C=self.T,self.C
            x =np.zeros((T,C))
            u_x=np.random.normal(0,1,(T,C))
            for j in range(0,C):
                  for i in range(1,T):
                        x[i,j]=non_cen_p*x[i-1,j] +u_x[i,j] +lax[j]*y2[i]
            X=pd.DataFrame(x)
            return X

      
      #############gen y noise.
      def y_noise(self):
            y3=self.fac.iloc[:,2]
            ## get factor and factor loadings 
            lay=self.lay
            
            cols=[np.full((self.T, 1), i) for i in lay] 
            loadings=pd.DataFrame(np.concatenate(cols,1),index=y3.index)
            u=loadings.multiply(y3,0)
            u = 1*pd.DataFrame(np.random.normal(size=(self.T, self.C)))
            u=u.add(loadings,axis=0)
            return u 


############################################################################
############################################################################
### SCRIPT
############################################################################
############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time 
start_time = time.time()
# np.random.seed(3)
Ts=[200]
Ns=[1]
betas_power=[0.05]

b=0#burn in  

threshold=0 #threshold

mc_reps=50#simulation reps

bootreps=250#bootstrap repetition

arr1=['power b1']

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
                  beta1=[0.5 for i in range(0,C)]
                  beta2=[0.5 for i in range(0,C)]
                  alpha1=[0.5 for i in range(0,C)]
                  alpha2=[0.5 for i in range(0,C)] 
                  if C==1:
                        beta1[0]=0.5+beta
                        alpha1[0]=0.5
                        beta2[0]=0.5
                  else:
                        beta1[0:int(C/2)]=[0.5+beta for i in range(0,int(C/2))]
                        alpha1[0:int(C/2)]=[0.5+beta for i in range(0,int(C/2))]
                  mc_results_store=[]
                  for mc_rep in range(0,mc_reps,1):
                        
                        # start intance of innovations
                        inovs=innovations(T=T+b,C=C)
                        
                        #generate threshold   
                        sam=inovs.threshvar()
                        
                        
                        #generate the panel data into a dataframe, high persistence 
                        #generate rw, and then IVX it to simulate hihgly persitent but not RW            
                        X=inovs.rw_predictor()
            
                        
                        #generate the y noise
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
                        # x_disturb=paneldata.iloc[:,C:2*C].diff().fillna(0).reset_index(drop=True)
                        x_disturb=pd.DataFrame()
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
                              x_res,xbeta=ar1(x)
                              x_disturb[str(i)]=x.squeeze()
                              xbetas.append(xbeta[0][0])
                              # #min SSR index number for gamma
                              mgi,drop=supf(y,x,thresh)
                              max_gamma_index.append(mgi)
                              maxwalds.append(max(drop))

                              
                  
                                               
                        #get average of max wald stats per country
                        real_data_test_stat=np.average(maxwalds)
                        
            
                        t1=time.time()
                        ###############################################PART 2 Bootstrap
                        bootrep=0
                        bootstrap_stats=[]
                        print('starting bootstrap')
                        while bootrep<bootreps:

                                ############# 
                                e_boot,x_boot=rbb(resids,x_disturb,startingx,4,xbetas)  
                              
                                
                                
                                x_boot=x_boot.reset_index(drop=True)
                                e_boot=e_boot.reset_index(drop=True)
                                x_boot.columns=e_boot.columns
                                y_boot=x_boot*null_betas+e_boot
                                boot_df=y_boot.merge( x_boot,left_index=True, right_index=True)
                                boot_df['thresh']=gen_thresh(T)
                                boot_df= boot_df.sort_values(by=['thresh'])
                                maxwald_boot=[]
                                for col in range(0,len(x_boot.columns)):
                                      y_s=np.array(boot_df.iloc[:,col],ndmin=2).T#
                                      x_s=np.array(boot_df.iloc[:,col+C],ndmin=2).T
                                       
                                      mgi,wald_boot=supf_nosort(y_s, x_s)
                                      maxwald_boot.append(max(wald_boot))
                            

                                    
                                bootstrap_stats.append(np.average(maxwald_boot))
                                bootrep+=1
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
    
# ######Print which quantile of bootstrap is real_data_test_stat
# ##### for all MC iterations
# import datetime
# dt=datetime.datetime.now()
# titlestring='results_out_jit_'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
# with open(titlestring, 'w') as filehandle:
#     filehandle.write(results_array.to_string())   
#     filehandle.write(str('\n panel size NT '+str(C)+' '+str(T)))
#     filehandle.write(str('\n power '+str(betas_power_store)))
          

#print how much time
print("--- %s seconds ---" % (time.time() - start_time))
totaltime=time.time() - start_time

###### Print results of what how often real test stat is above 95th 
###### quantile of bootstrap test stats (indicating rejection of null)
dt=datetime.datetime.now()
titlestring='results_ivx_CLT'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
with open(titlestring, 'w') as filehandle:
    filehandle.write(result_frame.to_string())     
    filehandle.write(str('\n time '+str(totaltime)))


print(result_frame,
'   time ',Ts,
'size ',Ns,
'beta ',betas_power,
'non cen p ', inovs.non_cen_p,
'qunatile ', quantile )

plt.hist(a,bins=50)
plt.xlim(0.0,14)
