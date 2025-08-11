import streamlit as st
import pprint

def modify_model_parameters():
    st.header('Modify the parameters used to train the XGBoost Model')
    st.markdown('Available parameters and descriptions can be found [here](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn)')
    
    from model_parameters import XGB_PARAMS
    
    new_params = {}
    for param, value in XGB_PARAMS.items(): 
        new_val = st.number_input(
            label=f'{param}',
            value=f'{value}'
        )
        new_params[param] = new_val
        
    if st.button('Save New Parameters'): 
        with open('model_parameters.py', 'w') as f: 
            f.write("XGB_PARAMS = ")
            f.write(pprint.pformat(new_params))
        st.success('Successfully updated parameters')