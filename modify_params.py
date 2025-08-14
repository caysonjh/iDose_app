import streamlit as st
import pprint
import streamlit_antd_components as sac
from storage_interaction import write_user_environment

def update_parameters(new_params): 
    with open('model_parameters.py', 'w') as f: 
        f.write("XGB_PARAMS = ")
        f.write(pprint.pformat(new_params))
    st.success('Successfully updated parameters')
    
    st.session_state['model_parameters'] = new_params 
    write_user_environment()

def modify_model_parameters():
    st.header('Modify the parameters used to train the XGBoost Model')
    st.markdown('Available parameters and descriptions can be found [here](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)')
    
    from model_parameters import XGB_PARAMS
    
    new_learning_rate = st.slider(
        label='Learning Rate',
        min_value=0.0,
        max_value=1.0,
        value=XGB_PARAMS['learning_rate'],
        step=0.005
    )
    subsample = st.slider(
        label='Subsample',
        min_value=0.0,
        max_value=1.0,
        value=XGB_PARAMS['subsample'],
        step=0.05
    )
    new_max_depth = st.slider(
        label='Max Depth',
        min_value=0,
        max_value=12,
        value=XGB_PARAMS['max_depth'],
        step=1
    )
    new_n_estimators = st.slider(
        label='Number of Estimators',
        min_value=25,
        max_value=500,
        step=25, 
        value=XGB_PARAMS['n_estimators']
    )
    
    new_params = {
        'learning_rate':new_learning_rate,
        'subsample':subsample,
        'max_depth':new_max_depth,
        'n_estimators':new_n_estimators
    }
        
    if st.button('Save New Parameters', icon=':material/build:', width='stretch'): 
        update_parameters(new_params)
        
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='load_end')  
