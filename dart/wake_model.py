import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import MultiTaskLassoCV

def wake_model(df_inflow,dict_LUT):
    """
    Core of the model, calculations (matrix multiplication) takes place here

    Args:
        df_inflow: inflow parameters
        dict_LUT: lookup table with coefficients
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    targets = ['A_z','mu_y','mu_z','sigma_y','sigma_z','c','t','s_a','s_b']
    predictors = dict_LUT['predictors'] 
    transformations = dict_LUT['transformations']
    
    ## transformations
    for i in range(len(transformations)):
        if transformations[i] == 'log': 
            df_inflow.iloc[:,i] = np.log(df_inflow.iloc[:,i])
        elif transformations[i] == 'exp':
            df_inflow.iloc[:,i] = np.exp(df_inflow.iloc[:,i])
        elif transformations[i] == 'rec':
            df_inflow.iloc[:,i] = 1/df_inflow.iloc[:,i]
        elif transformations[i] == 'sqrt':
            df_inflow.iloc[:,i] = np.sqrt(df_inflow.iloc[:,i])  
    input_val = np.squeeze(df_inflow.values)    
    
    ## polynomial features
    polynomial_features = PolynomialFeatures(degree=2) 
    input_val = np.asarray(input_val)
    input_val = input_val.reshape(1, -1)
    input_val = np.squeeze(polynomial_features.fit_transform(input_val))
    
    ## Actual calculations
    pred = np.empty((len(targets),len(dict_LUT['x/D'])))
    for i,target in enumerate(targets):
        coef,inter = dict_LUT[target], dict_LUT[target+'_i'] ## coefficients and intercept
        pred[i,:] = coef @ input_val + inter ## matrix multiplication + intercept
        
    return pred

def wake_composition(pred,Ds,hh=90,rD=126,dx=5):
    """
    Composition method. Generate cross-section from key wake steering parameters

    Args:
        pred: prediction of key wake steering parameters
        Ds: Distances downstream
        hh: hub height of turbine (same as turbine the model was trained for)
        rD: rotor diameter of turbine (same as turbine the model was trained for)
        dx: resolution of composed image
    """

    def gaus1d(y, du, mu, sigma): 
        return du*np.exp(-((y-mu)**2)/(2*sigma**2))

    def ellipse(ywidth,zwidth,z):
        if ywidth > zwidth: ## ellips: y**2/a**2 + z**2/b**2 = 1
            a,b = ywidth,zwidth
            y = np.sqrt(a**2*(1-(z**2/b**2)))
        elif ywidth < zwidth: ## ellips: y**2/b**2 + z**2/a**2 = 1
            a,b = zwidth,ywidth
            y = np.sqrt(b**2*(1-(z**2/a**2)))
        return y

    ### prep work  
    y = np.arange(-2*rD,2*rD,dx)
    y = [k/126 for k in y]
    z = np.arange(-hh,rD,dx)
    z = [k/126 for k in z]
    
    ### Step 1: Get wake parameters
    A_z,mu_y,mu_z,sigma_y,sigma_z,c,t,s_a,s_b =  pred
    
    composition = np.zeros((len(Ds),len(z),len(y))) ## new empty field
    for D in range(len(Ds)):
    ### Step 2: Local wake center deficits
        As = [gaus1d(k,A_z[D],mu_z[D],sigma_z[D]) for k in z] 
        if np.nanmin(As) == 0: continue

    ### Step 3: Local wake center positions
        top = np.nanmax([z[i] for i in range(len(z)) if As[i]!=0])
        z_base = [k for k in z if k<top]
        y_base = [c[D]*k**2+t[D]*k for k in z_base] ## corresponding y
        shift = np.interp(mu_z[D],z_base,y_base) ## wake center location should be zero
        y_base = [k-shift for k in y_base] ## relative to wake center
        mus = mu_y[D]+y_base

    ### Step 4: Local wake widths
        ## determine sections
        z_base = [k for k in z if k<top]
        z_base1 = [k for k in z_base if k < -0.5]
        z_base2 = [k for k in z_base if -0.5 < k < 0.5]
        z_base3 = [k for k in z_base if k > 0.5]
        ## rotor area
        y_base2 = [s_a[D]*k**2+s_b[D]*k for k in z_base2] ## corresponding y
        stds_hh = np.interp(0,z_base2,y_base2) 
        y_base2 = y_base2 + (1-stds_hh) ## hub height should be one
        norm = np.interp(mu_z[D],z_base2,y_base2)
        y_base2 = list(y_base2/norm) ## normalize
        ## outside of rotor area
        y_base1 = [ellipse(y_base2[0],z_base1[-1]-z_base1[0],k-z_base1[-1]) for k in z_base1]
        y_base3 = [ellipse(y_base2[-1],z_base3[-1]-z_base3[0],k-z_base3[0]) for k in z_base3]
        ## concat and give right value
        stds = y_base1 + y_base2 + y_base3
        stds = np.asarray(stds)*sigma_y[D] ## multiply by std value at wake center height

    ### Step 5: Reversed Multiple 1D Gaussian
        As = As[:len(z_base)]
        for zi in range(len(composition[D])): ## vertical
            for yi in range(len(composition[D,zi])): ## horizontal
                if zi < len(z_base):
                    composition[D,zi,yi] = gaus1d(y[yi], As[zi], mus[zi], stds[zi])
                else:
                    composition[D,zi,yi] = 0
    composition = np.swapaxes(composition,0,1) 
    composition = np.swapaxes(composition,1,2) ## swap axis such that (z,y,x)
    return y,z,composition


def make_LUT_coefs(df_inflow,df_params,transformations,Ds):
    """
    Training function. 
    Transforms input parameters as desired and applies Multi-task Lasso to obtain coefficients

    Args:
        df_inflow: input parameters
        df_params: key wake steering parameters
        transformations: transformations to be applied to input parameters
        Ds: downstream distances to be used
    """

    ## due to cross validation in MultiTaskLassoCV, at least 5 samples are needed
    ## code below is a workaround, only to be used for testing purposes
    ## quality of the DART can be low for small sample sizes
    if len(df_inflow) < 5:
        for i in range(3):
            df_inflow = pd.concat([df_inflow,df_inflow],axis=0)
            df_params = pd.concat([df_params,df_params],axis=0)

    ## Transformations
    for i in range(len(transformations)):
        if transformations[i] == 'log': 
            df_inflow.iloc[:,i] = np.log(df_inflow.iloc[:,i])
        elif transformations[i] == 'exp':
            df_inflow.iloc[:,i] = np.exp(df_inflow.iloc[:,i])
        elif transformations[i] == 'rec':
            df_inflow.iloc[:,i] = 1/df_inflow.iloc[:,i]
        elif transformations[i] == 'sqrt':
            df_inflow.iloc[:,i] = np.sqrt(df_inflow.iloc[:,i])
            
    targets = ['A_z','mu_y','mu_z','sigma_y','sigma_z','c','t','s_a','s_b']
    dict_LUT = {}
    dict_LUT['predictors'] = df_inflow.columns
    dict_LUT['transformations'] = transformations
    dict_LUT['x/D'] = list(Ds)
    for target in targets:
        ydy = df_params[[target + str(i) for i in Ds]]
        Xdy = df_inflow
        polynomial_features = PolynomialFeatures(degree=2) ## polynomial features
        Xdy = polynomial_features.fit_transform(Xdy)
        reg_dy = MultiTaskLassoCV(cv=5, random_state=0, normalize=True, max_iter=5e3).fit(Xdy, ydy)
        dict_LUT[target] = reg_dy.coef_
        dict_LUT[target+'_i'] = reg_dy.intercept_
    return dict_LUT

