import pytest
import dart
from dart import DART
import pandas as pd

def test_init():
    wake_model = DART()
    
    
def test_determine_key_parameters():
    wake_model = DART()
    wake_model.determine_key_parameters(['yaw_00','yaw_30','yaw_-30'])
    assert hasattr(wake_model, "key_parameters")
    assert round(wake_model.key_parameters.loc["yaw_00", "t1"],5) == -0.02075

def test_train_coeffs():
    wake_model = DART()
    wake_model.determine_key_parameters(['yaw_00','yaw_30','yaw_-30'])
    Ds = range(4,11) ## Distances downstream
    df_input = pd.read_csv('data/input.csv',index_col=0)
    transformations = ['','','']
    wake_model.train_coeffs(df_input,transformations,Ds)

    for pred in ['yaw', 'shear', 'Ct']:
        assert pred in wake_model.dict_LUT["predictors"]


def test_run_wake_model():
    wake_model = DART()
    wake_model.determine_key_parameters(['yaw_00','yaw_30','yaw_-30'])
    Ds = range(4,11) ## Distances downstream
    df_input = pd.read_csv('data/input.csv',index_col=0)
    transformations = ['','','']
    wake_model.train_coeffs(df_input,transformations,Ds)

    input_values = [15,0.165,0.65]
    wake_model.run_wake_model(input_values)
    assert round(wake_model.wake_yzu["y"][10],5) == -1.60317
    assert round(wake_model.wake_yzu["z"][11],5) == -0.27778
    assert round(wake_model.wake_yzu["u_normdef"][25][50][3],5) == -0.19731
