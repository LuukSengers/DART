import pandas as pd
import numpy as np

from .utils import load_crosssections, plot_wake
from .multi1Dgaus import multi1Dgaus
from .wake_model import wake_model, wake_composition, make_LUT_coefs


class DART:
    """
    This is the main API for using the Data-driven wAke steeRing surrogaTe model (DART)
    """

    def determine_key_parameters(self,sims=['yaw_00','yaw_30','yaw_-30'],f_path="data/"):
        """
        Wrapper that loops over simulation and downstream distances. 
        Saves dataframe with key wake steering parameters

        Args:
            sims: names of simulations 
        """    
        print("Determining key wake steering parameters")
        for i in range(len(sims)):
            z,y,x,u_all = load_crosssections(f_path + sims[i]+'.nc')
            if i == 0:
                variables = ['A_z','mu_y','mu_z','sigma_y','sigma_z','c','t','s_a','s_b'] ## variables to be determined
                cols = np.asarray([[k+str(D) for k in variables] for D in x]).flatten()
                df_all = pd.DataFrame(np.nan,index=sims,columns=cols) 
            for D in range(len(x)):
                u = u_all[:,:,D] ## select correct D
                df = multi1Dgaus(z,y,u) ## actual determination
                df.columns = [k+str(x[D]) for k in df.columns] 
                for col in df.columns: ## add to big dataframe
                    df_all.loc[sims[i],col]=df[col].values
        print("Done")
        self.key_parameters=df_all

    def train_coeffs(self,df_input,transformations,Ds):
        """
        Training function. 

        Args:
        df_inflow: input parameters
        transformations: transformations to be applied to input parameters
        Ds: downstream distances to be used
        """
        print("Determining coefficients")
        self.dict_LUT=make_LUT_coefs(df_input,self.key_parameters,transformations,Ds) ## execute
        print("Done")
        

    def run_wake_model(self,input_values):
        """
        Wrapper that collects all necessary data and calls the wake model

        Args:
        input_values: New input parameters
        """
        df_input = pd.DataFrame(np.nan,index=['dummie'],columns=self.dict_LUT['predictors'])
        df_input.iloc[0,:] = input_values ## create df of input values
        if not hasattr(self, "dict_LUT"):
            print("model needs to be trained with train_coeffs before running wake model")
        Ds = self.dict_LUT['x/D'] ## Define downstream distances (between 4 and 10)

        print('Predicting wake for following settings: \n Input parameters:',self.dict_LUT['predictors'].tolist(),'\n',
            'Input values', input_values,'\n Transformations: ',self.dict_LUT['transformations'],'\n Distances downstream:',np.asarray(Ds))

        pred = wake_model(df_input,self.dict_LUT)
        y,z,u_normdef = wake_composition(pred,Ds)
        self.wake_yzu = {"y":y,"z":z, "u_normdef":u_normdef}
        print('\nSucess: The wake parameters(y,z,u_normdef) are saved in .wake_yzu')

    def plot(self,D=5):
        """
        Call simple plotting function
        """
        plot_wake(self.wake_yzu, D=D)
