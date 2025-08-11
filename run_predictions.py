import streamlit as st 
from streamlit_option_menu import option_menu
from save_models import update_saved_models
import joblib
import random
import numpy as np 
from load_data import check_data_loaded

def run_prediction(): 
    st.header('Test Various Model Configurations and View Metrics/Feature Insights')

    selected = option_menu(None, ['Use Saved Model', 'Upload Model From File', 'Train New Model'], 
                           icons=['life-preserver', 'paperclip', 'airplane-engines'], orientation='horizontal')
    
    model = None 
    
    if selected == 'Use Saved Model': 
        model_file = None
        if st.session_state.get('saved_classifiers', []): 
            st.markdown('#### Previously Saved Models -- Click a file to use for prediction')
            macs = np.unique([clf[0] for clf in st.session_state['saved_classifiers']])
            cols = st.columns(len(macs))
            for i, mac in enumerate(macs): 
                cols[i].markdown(f'###### {mac}')
            
            for mac, clf in st.session_state['saved_classifiers']:
                idx = np.where(macs == mac)[0][0]
                col = cols[idx]
                if col.button(clf[:-4], key=f'predict_{mac}_{clf}_{random.randint(0,10000)}'): 
                    st.session_state['predict_model'] = clf
            
            #TODO not working
            if st.session_state.get('predict_model', []):
                model = joblib.load(st.session_state['predict_model'])
                st.text(f'Selected Model is {st.session_state["predict_model"]}')
            
        else: 
            st.error('No Saved Models')
                
                
    if selected == 'Upload Model From File': 
        model_file = st.file_uploader("Upload Classifier File", type=['pkl'])
        filename = model_file.name.lower()
        model = joblib.load(filename)
        
    
    if selected == 'Train New Model': 
        if check_data_loaded(): 
            st.subheader('Train and save a new model')
            
            data = st.session_state['generated_df']
            st.success('Data Loaded -- Ready for Training')
            st.dataframe(data)
            
        
    
    