# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:25:46 2020

@author: hillr
"""


from scipy.stats import multivariate_normal

import statsmodels.api as sm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime
from multiprocessing import Pool, Value, current_process
from multiprocessing import Process
from multiprocessing import Manager
import time 
import itertools            


    
def bootfun(resids,x_disturb,startingx,T,C,null_betas):
    #generate boot u                
    e_boot,x_boot=rbb(resids,x_disturb,startingx,4)  
    
    #generate boot thresh, this is just white noise 
    boot_gamma=gen_thresh(T)
                              
    #generate the y, (and figure out how to make one or more 
    #individual inthe panel a threshold model) under ull of no threshold
    ydf_boot=pd.DataFrame(columns=[str(x) for x in range(0,C)])
    count=0
    for col in range(0,len(x_boot.columns)):
        x_s=x_boot.iloc[:,col] #
        u_s=e_boot.iloc[:,col]
        y=null_betas[col][0]*np.array(x_s)+np.array(u_s)
        # y=np.zeros_like(x_s)
        # for i in range(0,len(u_s)):#lag y doesnt makes sense
        #        y[i]=null_betas[count][0]*x_s.iloc[i]+u_s.iloc[i]
        ydf_boot[str(count)] =y     
        count=count+1     
      
    #find wald statistic of generated null data      
     
    
    #IVX for bootstrapped X's
    #XVX=ivx(x_boot,delta=0.9,c=10,adfresults=False,plot=False)
    
    paneldata_boot=pd.merge(ydf_boot,  x_boot.reset_index(), left_index=True,right_index=True).drop(['index'],1) 
    #paneldata_boot=pd.merge(paneldata_boot, XVX, left_index=True,right_index=True) 
    # paneldata_boot.columns=map(str,(range(0,3*C)))
    paneldata_boot['gamma']=boot_gamma
    paneldata_boot_sorted=paneldata_boot.sort_values('gamma')
    maxwalds_boot=list()
    
    thresh=np.array(paneldata_boot['gamma'] ,ndmin=2).T
    thresh_sort=np.array(paneldata_boot_sorted['gamma'] ,ndmin=2).T
    for i in range(0,C):
        y=np.array(paneldata_boot.iloc[:,i],ndmin=2).T
        x=np.array(paneldata_boot.iloc[:,i+C],ndmin=2).T
        mgi,wald_boot=supf(y, x, thresh)
        
        wald_boot=max(wald_boot)
        maxwalds_boot.append(wald_boot)

    #maxs= [max(p) for p in maxwalds] 
    return np.average(maxwalds_boot)

def bootfun_wrapper(args):
    # Convenience wrapper for use with map
    return bootfun(args[0], args[1], args[2],args[3],args[4],args[5])




def gen_thresh(T,mean = 0,std = 1 ):
      samples = np.random.normal(mean, std, size=T)
      return samples



def gen_rw(T,C,rho):
      df = pd.DataFrame(np.random.normal(size=(T, C)))
      x=[]
      for t in range(1,T+1):
            vec_ =np.arange(0,t)
            x.append(np.sum(np.reshape(np.power(rho, vec_), (t, 1)) * df.iloc[ np.flip(vec_),:]))
            
            
      X=pd.DataFrame(x)
      df=X.cumsum()
#      df.plot()
      return df

def gen_errors(T,C,ar,ma=None ,MA=False):
      if MA==False:
            df = pd.DataFrame(np.random.np.random.normal(size=(T, C)))
            x=[]
            for t in range(1,T+1):
                  vec_ =np.arange(0,t)
                  x.append(np.sum(np.reshape(np.power(ar, vec_), (t, 1)) * df.iloc[ np.flip(vec_),:]))
            df=pd.DataFrame(x)                 
      else:
            df = pd.DataFrame(np.random.np.random.normal(size=(T, C)))
            df=df+ma*df.shift(1)

      return df
            
            


def ivx(X,delta,c,adfresults=True,plot=True):#xvar=X,delta=0.99,c=10,adfresults=False,plot=True
      T=np.shape(X)[0]
      phi=1-c/np.power(T,delta)
      x=X.diff().iloc[1:,:]
#     ivx=[sum([np.power(phi,t-j)*x.iloc[j,:] for j in range(0,t)]) for t in range(1,T)]     
#      ivx=pd.DataFrame([item.tolist() for item in ivx])
#      ivx.index=range(1,T)
      ivx=[]
      for t in range(1, T):
            vec_ = np.arange(0, t)
            ivx.append(np.sum( x.iloc[vec_].multiply(np.power(phi, t-vec_-1),0)))
            
      ivx=pd.DataFrame( ivx)
      ivx.loc[-1] =0 
      ivx.index = ivx.index + 1 
      ivx = ivx.sort_index()  
#      ivx.loc[len(ivx)+1] = 0
      if adfresults==True:
            adfresults=adfuller(ivx)
            if adfresults[0]-adfresults[4]['5%']>0:
               print('do not reject ho of unit root')
            else:
               print('reject ho of unit root')   
      if plot==True:  
            fig,ax=plt.subplots(2,1)
            ax[0].plot(xvar,label='x variable')
            ax[1].plot(ivx,label='ivx')
            ax[0].legend()
            ax[1].legend()
            plt.show()
      return ivx

#



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
            walds[t,:]=theta.T@R.T@np.linalg.inv(R@np.linalg.inv(Z.T@Z)@R.T)@R@theta/  sig2
            

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



def sumsqe(y, thresh,p=0.15):
      #reutnr sum squred residuals from a 
       T = y.shape[0]
       r = np.floor(np.array([T * p, T * (1-p)]))
       r = np.arange(r[0], r[1] + 1, dtype=np.int32)
       sortedframe=np.hstack((y, thresh))
       sortedframe=sortedframe[sortedframe[:,1].argsort()]
       y_=sortedframe[:,0:1] 
       ones1=np.zeros((T,1))
       ones2=np.zeros((T,1))
       ones1[:r[0]]=1.0
       ones2[r[1]:]=1.0
       alphas=np.hstack((ones1,ones2))
       ssr=np.zeros((T,1))
       for t in r:
             alphas[t,:]=[1,0]
             bb=np.linalg.inv(alphas.T@alphas)@alphas.T@y_
             u=y_-alphas@bb
             u2=u.T@u
             ssr[t]=u2
       ssr=ssr[ssr!=0]     
       max_gamma_index=ssr.argmin()+r[0]-1
       # plt.plot(ssr[ssr!=0])
       return  max_gamma_index
   

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
     
      R=np.array([[1.0,-0.0],[0.0,1.0]])
      R=np.array([[1.0,-1.0]])
      
    
      varbhat=np.linalg.inv(Xstar.T@Xstar)*sigma2
      max_waldivc=b.T@R.T@np.linalg.inv(R@varbhat@R.T)@R@b
      return max_waldivc
       
  

def ar1(x):
      X=np.vstack((np.ones(x.shape),x))
      xl=X[:,:-1]
      xf=X[1,1:]
      b=np.linalg.inv(xl@xl.T)@xl@xf.T
      res=xf-b.T@xl
      return res,b



def beta_null(y, x,iv): 
      #create stared (demeaned variables as per GP)

       T = y.shape[0]
       Q=np.ones((T,1))
       inv=np.linalg.inv(Q.T@Q)@Q.T
       M=np.identity(Q.shape[0])-np.dot(Q,inv)
       ystar=y
       Hstar=iv
       Xstar=x
       #get B_iv
       b_iv=np.linalg.inv(Hstar.T@Xstar)
       b_iv_=b_iv@Hstar.T@ystar   
       #resids for bootstrap
       pred=Xstar@b_iv
       resids_iv_null=ystar-pred
       
       return  resids_iv_null,b_iv
      
      
      


def supf_ivx(y, x,iv, thresh,mgi,p=0.15): 
      #this tests for predictability in the sense of GP 2017
      #set up variables
      T = y.shape[0]
      #y_=np.array(y)
      #x_=np.array(x)
      r = np.floor(np.array([T * p, T * (1-p)]))
      r = np.arange(r[0], r[1] + 1, dtype=np.int32)
      #data is sorted but it should be ok to get null betas and errors for the bootstrap 
      #procedure later. 

      #data is already sorted - thought it might be faster
       
      # walds=supf(y,x,thresh,p=0.15)      
      # #maxindex=pd.Series(walds[0]).idxmax()+r[0]
      # maxindex=np.array(walds[0]).argmax()+r[0]
      
           
      #now that the threshold under null of now beta predictability is found, IV wald test at that thresh
      #ivxdf=np.vstack((y, x,iv,thresh)).transpose()
      #Q1=(ivxdf.index<=maxindex).astype(int)
      #Q2=(ivxdf.index>maxindex).astype(int)
      Q1=np.ones((T,1))
      Q1[mgi :]=0
      Q2=np.ones((T,1))
      Q2[: mgi]=0
      X1=Q1*x.reshape((T,1))
      X2=Q2*x.reshape((T,1))
      H1=Q1*iv.reshape((T,1))
      H2=Q2*iv.reshape((T,1))
      
      Q=np.hstack((Q1,Q2))
      H=np.hstack((H1,H2))
      X=np.hstack((X1,X2))

      M=np.identity(Q.shape[0])-Q@np.linalg.inv(Q.T@Q)@Q.T
     
      ystar=M@y
      Hstar=M@H
      Xstar=M@X
      
      b_iv=np.linalg.inv(Hstar.T@Xstar)
      b_iv=b_iv@Hstar.T@ystar
     
      pred=Xstar@b_iv
     
      resid=ystar-pred
      sigma2=np.sum(resid**2)/(resid.shape[0]-1)
       
      # forcorr=sm.OLS(x[1:],sm.add_constant(x[0:-1]),missing='drop')
      # u=forcorr.fit().resid
      u,drop=ar1(x)
      rho=np.corrcoef(resid[1:],u,False)[1,0] 
     
      R=np.array([[1.0,0.0],[0.0,1.0]])
      R=np.array([[1.0,-1.0]])
      
      G=sigma2*((Hstar.T@Hstar)+np.power(rho,2)*np.dot(H.T,Q)@np.linalg.inv(Q.T@Q)@np.dot(Q.T,H))
      varbhat=np.linalg.inv(Hstar.T@Xstar)*G*np.linalg.inv(Hstar.T@Xstar)
      max_waldivc=b_iv.T@R.T@np.linalg.inv(R@varbhat@R.T)@R@b_iv
      return max_waldivc
     

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
#      


#def rbb(data,blocks):
#      
#      
#      return bootindex
#      
#
def rbb(resids,x_disturb,startingx,block_size):
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
      x1=startingx
      booted_x=pd.concat([x1,booted_x])
      
      booted_x=np.cumsum(booted_x)
      # x_phi=0.95
      # fac=[]
      # fac.append(np.array(x_phi*booted_x.iloc[0,:]))
      # for t in range(1,num_items+1):
      #       fac.append(np.array(x_phi*fac[t-1]+np.array(booted_x.iloc[t-1,:] )))           
      # booted_x=pd.DataFrame(fac)
      return booted_e,booted_x

def rbbx(x_disturb,startingx,block_size=2):
      x1=startingx
      e=x_disturb
      num_items=len(e)
      #what am I thinking here? 
      #have to block demean properly 
      num_blocks = (num_items-1 )// block_size
      max_index = num_items - block_size
      indices = np.random.randint(max_index, size=num_blocks)
      indices = indices[:, None] + np.arange(block_size)
      indices = indices.flatten()
      booted_e= e.iloc[indices,:]
#     booted_x=e[indices]
#     booted_e=booted_e(booted_e.insert(0,[5:9],x1))
#     x1=pd.DataFrame(x1)
      booted_e=pd.concat([x1,booted_e])
#      booted_e=booted_e.drop('index',1)
      booted_e=np.cumsum(booted_e)
      return  booted_e



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
            self.lax = 0.2*np.ones(C)  #np.random.uniform(-0.25,0.25,C)
            self.lay = 0.2*np.ones(C) #np.random.uniform(-0.25,0.25,C)
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
            fac=[]
            for t in range(1,T+1):
                  vec_ =np.arange(0,t)
                  fac.append(np.sum(np.transpose([np.power(factor_phi, vec_)]) * y[ np.flip(vec_),:],0))

            fac=pd.DataFrame(fac) # gives 3 vectors of factors 
      #      fac.plot()
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
                        x[i,j]=non_cen_p*x[i-1,j]+ u_x[i,j] #+lax[j]*y2[i]#
            X=pd.DataFrame(x)
            return X

      
      #############gen y noise.
      def y_noise(self):
            y3=self.fac.iloc[:,2]
            ## get factor and factor loadings 
            lay=self.lay
            
            cols=[np.full((self.T, 1), i) for i in lay]
            loadings=pd.DataFrame(np.concatenate(cols,1),index=y3.index)
            loadings=loadings.multiply(y3,0)
            u = 1*pd.DataFrame(np.random.normal(size=(self.T, self.C)))
            u=u.add(loadings,axis=0)
            return u 


def main(T,C,beta):
          # start intance of innovations

    print('time ',T,'   C ',C,'   beta ', beta)      
    mc_results_store=[]
    b=0
    threshold=0
    counter=0
    if C==1:
        beta1=[beta]
        alpha1=[0.0]
        alpha2=[0.0]
        beta2=[0.0]
    else:
        beta1=[0 for i in range(0,int(C))]
        beta1[0:int(C/2)]=[beta for i in range(0,int(C/2))]
        beta2=[0 for i in range(0,int(C))]
        alpha1=[0 for i in range(0,int(C))]
        alpha2=[0 for i in range(0,int(C))]
    betas_results_store=[]
    betas_power_store=[]
    inovs=innovations(T=T+b,C=C)
    sam=inovs.threshvar()
    X=inovs.rw_predictor()
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

    paneldata.columns=map(str,(range(0,2*C)))
    paneldata['gamma']=    X['gamma'] 
     #cut out the burn in 
    paneldata=paneldata.iloc[1:,:]
    
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
          #residuals and null betas for bootstrap
          e_4boot,null_beta=beta_null(y, x,x )
          resids[str(i)]=e_4boot.reshape((T-1))
          null_betas.append(null_beta)
          drop,xbeta=ar1(x.T)
          xbetas.append(xbeta)
          #min SSR index number for gamma
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

    # bootinlist=[resids,x_disturb,startingx,T,C,null_betas]
    # setup=[bootinlist for i in range(bootreps)]    
    # po = Pool()
    # res = po.map(bootfun_wrapper, setup)
    # bootstrap_stats=list(res)


    bootinlist=[resids,x_disturb,startingx,T,C,null_betas]
    setup=[bootinlist for i in range(bootreps)]  
    bootstrap_stats=list(map(bootfun_wrapper, setup))


    #end bootstrap
    t2=time.time()
    print('time of bootstrap',t2-t1,' boot reps ', bootreps )
    
    a = np.array(bootstrap_stats)     
    quantile=(a<real_data_test_stat).astype(int).mean()
    mc_results_store.append(quantile)
    print('process id  ',  print ,current_process())

    return mc_results_store #,bootstrap_stats
      
def main_wrapper(args):
    # Convenience wrapper for use with map
    return main(args[0], args[1], args[2])    
                




start_time = time.time()
#np.random.seed(1)
Ts=[400] #[800,400,200] #1,0.025,0.05,0.075,0.1]#,0.05,0.075,0.1]# ,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
Ns=[26] #[1,6,10,26]
arr1=['power b1']#['size','power b1','power b2']#,'power b1']#,
betas_power=[0.025]#[0.0,0.025,0.05]
b=0#burn in  


mc_reps=2#simulation reps
bootreps=100#bootstrap repetition


arrs=[Ts,arr1]
col=pd.MultiIndex.from_product(arrs)
result_frame=pd.DataFrame(np.ones((len(Ns),len(betas_power)*len(Ts))),index=Ns,columns=col) #dataframe to fill for results


if __name__ == '__main__':
 
    # mainlist=list(itertools.product(Ts,Ns,betas_power))
    # setup=[mainlist for i in range(bootreps)]  
    # setup=list(itertools.chain.from_iterable(setup))
    # bootstrap_stats=list(map(main_wrapper, setup))
    
    mainlist=list(itertools.product(Ts,Ns,betas_power))
    setup=list(itertools.chain.from_iterable(itertools.repeat(x, mc_reps) for x in mainlist))
    
    #for multiprocessing
    # po = Pool(processes=(8))
    # mc_results_store=list(po.map(main_wrapper, setup))
    #for runnning on my cpu
    mc_results_store=list(map(main_wrapper, setup))
    
    
    
    mc_results_store = [val for sublist in mc_results_store for val in sublist]
    chunks = [mc_results_store[x:x+mc_reps] for x in range(0, len(mc_results_store), mc_reps)]
    
    powerfun = lambda x: len([1 for i in x if i >= 0.95])/mc_reps



    power = list(map(powerfun, chunks))
    count1=0
    for i in Ts:
        for j in Ns:
            for k in arr1:
    
                result_frame.loc[j,(i,k)]=power[count1]
                count1=count1+1
    
        #print how much time
    print("--- %s seconds ---" % (time.time() - start_time))
    totaltime=time.time() - start_time
    #save file with results     
    dt=datetime.datetime.now()
    titlestring='results_ivx_CLT'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
    # with open(titlestring, 'w') as filehandle:
    #     filehandle.write(result_frame.to_string())     
    #     filehandle.write(str('\n time '+str(totaltime)))
    
    
    
    
    # print('percent mc done: ', 100*counter/counts)
 
    # #end MC loop
    # #power from the round of MC reps
    # power=len([1 for i in mc_results_store if i >= 0.95])/mc_reps
    # print('power of test for ', beta, ' is: ', power )
    
    # betas_power_store.append(power)
    # #end betas loop

    
    


    # results_array=pd.DataFrame({'beta of unit 1 ': betas_power, 'power of test': betas_power_store})
    # results_array=pd.DataFrame(np.array(betas_results_store).T,columns=betas_power,index=range(1,mc_reps+1))
    # #results_out=np.mean(results_array,axis=1)
    # #print('mean bootstrap results over MC reps ',np.mean(results_array,axis=1))
    
    # import datetime
    # dt=datetime.datetime.now()
    # titlestring='results_out_jit_'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
    # with open(titlestring, 'w') as filehandle:
    #     filehandle.write(results_array.to_string())   
    #     filehandle.write(str('\n panel size NT '+str(C)+' '+str(T)))
    #     filehandle.write(str('\n power '+str(betas_power_store)))
          
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    
    # #print how much time
    # print("--- %s seconds ---" % (time.time() - start_time))
    # totaltime=time.time() - start_time
    # #save file with results     
    # dt=datetime.datetime.now()
    # titlestring='results_ivx_CLT'+dt.strftime('%d_%m_%Y_%H_%M_%S')+'.txt'
    # with open(titlestring, 'w') as filehandle:
    #     filehandle.write(result_frame.to_string())     
    #     filehandle.write(str('\n time '+str(totaltime)))
    
    
    
    # import scipy as sy
    # import statsmodels as sm
    # from statsmodels.distributions.empirical_distribution import ECDF
    # import scipy.stats as sstats
    # a=np.array(bootstrap_stats)
    # # a=np.sqrt(5/4)*(a-2)
    # ecdf=ECDF(a)
    
    # fig, ax = plt.subplots(1, 1)
    
    # aa=min(a)
    # bb=max(a)
    # x = np.linspace(0,   bb, 100)
    # ax.plot(x, sstats.chi2.cdf(x, 1), label='chi2 pdf')
    # ax.plot(ecdf.x,ecdf.y)
