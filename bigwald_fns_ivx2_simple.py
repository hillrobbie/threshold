# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 21:40:35 2020

@author: hillr
"""
# -*- coding: utf-8 -*-
"""
15 feb 2021
This file contains the funcitons used in table1_Ha_alpha.py
-Important funcitons are sumsq and waldsplit and the DGP itself.
-The DGP is governed by the 'innovations' class at the bottom. It is set to no CSD and no serial corr
by default
"""
import numpy  as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
from numba import jit
import statsmodels.api as sm


            
def gen_thresh(T,mean = 0,std = 1 ):
      samples = np.random.normal(mean, std, size=T)
      return samples


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
       
       return  resids_iv_null,b_iv
      
      
      

@jit()
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
      
           
      #now that the threshold under null of no beta predictability is found, IV wald test at that thresh
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
      u,drop=ar1(x.T)
      rho=np.corrcoef(resid[1:],u,False)[1,0] 
      
      R=np.array([[1.0,0.0],[0.0,1.0]])
      # R=np.array([[1.0,-1.0]])
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





def rbb(resids,x_disturb,startingx,block_size):

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





class innovations:
      
      def __init__(self, T=1000,
                         C=5,
                         factor_means = np.array([0.0, 0.0, 0.0]),
                         factor_covar = np.array([
                                            [ 1,  0.0,  0.0],
                                            [ 0.0,  1,  0.0],
                                            [ 0.0,  0.0,  1]
                                        ]),
                         factor_phi=0.0):
                         # lax=, #[-0.0,-0.0,0.0,0.0,0.0],
                         # lay= #[0.1,0.2,0.15,0.25,0.5]):
            self.T, self.C, self.factor_means = T,C,factor_means
            self.factor_covar, self.actor_phi = factor_covar, factor_phi
            self.lax = 0.0*np.ones(C)  #np.random.uniform(-0.25,0.25,C)
            self.lay = 0.0*np.ones(C) #np.random.uniform(-0.25,0.25,C)
            self.non_cen_p=1#-1/T
            

            num_samples =self.T #num_samples =T
            
            # The desired mean values of the sample.
            mu = self.factor_means
            
            # The desired covariance matrix.
            r = self.factor_covar
            
            # Generate the random samples.
            y = np.random.multivariate_normal(mu, r, size=num_samples)

            #turn fctors into AR 1 series            
            fac=[]
            for t in range(1,T+1):
                  vec_ =np.arange(0,t)
                  fac.append(np.sum(np.transpose([np.power(factor_phi, vec_)]) * y[ np.flip(vec_),:],0))

            fac=pd.DataFrame(fac) # gives 3 vectors of factors 

            self.fac=fac
      
      def threshvar(self):
      #############gen thresh  ## thresh variable can be just the first factor
            thresh=self.fac.iloc[:,0]
            return thresh
      
      #############gen randomwalk x for predictors  
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
            # u=u.add(loadings,axis=0)
            return u 
