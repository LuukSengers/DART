import numpy as np
import pandas as pd
import itertools
import scipy.optimize as opt

from .utils import load_crosssections


def multi1Dgaus(z,y,u): 
    """
    Computes key wake steering parameters from cross-sections.
    Applies the Multiple 1D Gaussian model

    Args:
        z: vertical coordinates, normalized by rotor diameter and centered around hub height
        y: lateral coordinates, normalized by rotor diameter and centered around turbine
        u: wake data, can be wake deficit or normalized wake deficit
    """


    def gaus1d(y, du, mu, sigma): ## returns du, mu, sigma
        return du*np.exp(-((y-mu)**2)/(2*sigma**2))
    
    def tilt_curl(y_ori,z_ori):  ## determination curl and tilt parameters
        ut,lt = 0.5,-0.5 ## hardcoded      
        y_ori = [y_ori[k] for k in range(len(y_ori)) if lt<=z_ori[k]<=ut]
        z_ori = [z_ori[k] for k in range(len(z_ori)) if lt<=z_ori[k]<=ut]    
        curl,tilt,c = np.polyfit(z_ori, y_ori, 2) 
        return tilt,curl    
    
    def stds_poly(y_ori,z_ori): ## determination width parameters
        ut,lt = 0.5,-0.5 ## hardcoded      
        y_ori = [y_ori[k] for k in range(len(y_ori)) if lt<=z_ori[k]<=ut]
        z_ori = [z_ori[k] for k in range(len(z_ori)) if lt<=z_ori[k]<=ut] 
        a,b,c = np.polyfit(z_ori, y_ori, 2) 
        return a,b
    
    ## Prepare wind speed
    u = np.asarray(u)
    u[np.isnan(u)] = 0
    u[u>0] = 0
    
    ## Prepare dataframe
    df = pd.DataFrame(np.nan,index=['exp + D'],columns=['A_z','mu_y','mu_z','sigma_y','sigma_z','c','t','s_a','s_b'])
    A_z,mu_y,mu_z,sigma_y,sigma_z,c,t,s_a,s_b = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
    
    ## Prepare limits
    z = np.asarray(z)
    idx_z = np.interp(0,z,range(len(z)))
    lim_dn = 0.05 * np.nanmin(u) ## minimum deficit in horizontal; first guess
    lim_up = 2 *  np.nanmin(u) ## maximum deficit in horizontal
    
    ## fit Gaussian at every vertical level
    popt_y = np.zeros([len(z),3])  
    for i in [int(idx_z)]+list(range(len(z))): ## for each vertical level (first hub height)
        init_y = [np.nanmin(u[i]),y[int(np.nanargmin(u[i]))],0.5] ## initial guess
        try:
            popt, _ = opt.curve_fit(gaus1d, y, u[i], init_y) ## fit
            if popt[2] < 0: 
                popt=[0,0,0]
        except RuntimeError: 
            ## if no clear deficit, fit will be unsuccessfull
            popt=[0,0,0]

        if i == int(idx_z) and np.nanmax(popt_y) == 0.0: ## first loop, set lim_dn
            lim_dn = gaus1d(popt[1]+1.96*popt[2],popt[0],popt[1],popt[2])
            continue                                
        if lim_up < popt[0]< lim_dn and y[0]<popt[1]<y[-1]: ## only keep reasonable values
            popt_y[i,:] = popt ## append
            
    ### Additional security checks for robustness
    ## Delete cells that are smaller than 5 grid cells vertically
    ymin = popt_y[:,0]
    a = [1 if k!=0 else 0 for k in ymin]
    starts = [k for k in range(len(a)) if a[k] == 1 and (a[k-1]==0 or k==0)] ## start new cell
    n = [sum(1 for _ in g) for k, g in itertools.groupby(a) if k == 1]
    for k in range(len(starts)): ## starts and n same length
        if n[k]<5: ## at least 5 grid cells
            popt_y[starts[k]:starts[k]+n[k]] = [0,0,0]
    ## If multiple cells, only keep one with largest deficit
    ymin = popt_y[:,0]
    a = [1 if k!=0 else 0 for k in ymin]
    starts = [k for k in range(len(a)) if a[k] == 1 and (a[k-1]==0 or k==0)]
    n = [sum(1 for _ in g) for k, g in itertools.groupby(a) if k == 1]
    for k in range(len(starts)): ## starts and n same length
        if np.nanmin(ymin[starts[k]:starts[k]+n[k]]) != np.nanmin(ymin): ## largest deficit
            popt_y[starts[k]:starts[k]+n[k]] = [0,0,0]  
    ## If no information on hub height: skip
    if popt_y[int(idx_z),1] == 0.: ## if no information at hub height, skip
        return df
    ## Delete outliers (non-contineous cell)
    y1,y2 = popt_y[:int(idx_z)],popt_y[int(idx_z):] 
    y1 = y1[::-1]
    maxdiff = 0.5 ## max displacement center two consecutive heights
    for i in range(len(y1)-1):
        if np.abs(y1[i,1]-y1[i+1,1]) > maxdiff: ## if larger than max allowed
            y1[i+1:,:] = [0,0,0] 
            break
    y1 = y1[::-1]        
    for i in range(len(y2)-1):
        if np.abs(y2[i,1]-y2[i+1,1]) > maxdiff: 
            y2[i+1:,:] = [0,0,0]
            break
    popt_y = np.concatenate((y1, y2)) 
    
    ### Determination key wake steering variables
    ## center
    y_A = np.copy(popt_y[:,0]) ## list with max deficit at every vertical level
    z2 = z[y_A!=0] ## only keep deficit values
    y_A2 = y_A[y_A!=0]    
    init_z = (np.nanmin(u),0,0.5)
    try:
        popt_z, _ = opt.curve_fit(gaus1d, z2, y_A2, init_z) ## fit
        if popt_z[1]<-0.5 or popt_z[1]>0.5: ## if center outside of rotor area
            popt_z = [np.nan,np.nan,np.nan]
        A_z,mu_z,sigma_z = popt_z
        ## With mu_z, now determine mu_y and sigma_y
        wc_z = np.interp(mu_z,z,range(len(z))) ## index of wake center
        hor_gaus = popt_y[int(np.floor(wc_z)):int(np.ceil(wc_z)+1)] ## cells surrounding center
        hor_gaus = [np.interp(wc_z%1,range(2),hor_gaus[:,k]) for k in range(len(hor_gaus[0]))] ## interpolated  
        A_y,mu_y,sigma_y = hor_gaus
    except:
        return df
    ## tilt and curve
    y_mu = np.copy(popt_y[:,1])
    z_mu = [z[k] for k in range(len(z)) if y_mu[k] != 0]
    y_mu = [k for k in y_mu if k != 0]
    t,c = tilt_curl(y_mu,z_mu)  
    ## width parameters
    y_std = np.copy(popt_y[:,2])
    z_std = [z[k] for k in range(len(z)) if y_std[k] != 0]
    y_std = [k for k in y_std if k != 0]
    y_hh = np.interp(0,z_std,y_std)
    y_std = [k/y_hh for k in y_std] ## normalize by width at hub heigth
    s_a,s_b = stds_poly(y_std,z_std) ## fit 2nd degree polynomial

    ## add to dataframe
    df.iloc[0,:] = [A_z,mu_y,mu_z,sigma_y,sigma_z,c,t,s_a,s_b]
    return df
