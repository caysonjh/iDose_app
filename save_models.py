import streamlit as st
import numpy as np 
from load_data import check_data_loaded
import random 
from model_backend import train_model
from model_exploration import prep_run_data, feature_selection
import os
import pandas as pd

def save_model():  
    st.header('Save a model for future prediction')
    
    saved_models_placeholder = st.empty()
    
    def render_saved_models(): 
        with saved_models_placeholder.container(): 
            update_saved_models()
            
    render_saved_models()
                
    if check_data_loaded(): 
        st.subheader('Train and save a new model')
        
        data = st.session_state['generated_df']
        st.success('Data Loaded -- Ready for Training')
        st.dataframe(data)
        
        with st.expander('Train and Save model for specific MAC'): 
            train_mac_split(data)
            render_saved_models()
        with st.expander('Train and Save model for all MACs'):
            train_all_macs(data)
            render_saved_models()
        
    else: 
        st.warning('Data must be loaded before training a model')

def train_all_macs(data):
    beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, use_mac = feature_selection('mac_split_save', all_macs=False)
    model_name = st.text_input(label='Model Name?', value='idose_prediction', placeholder='eg. glaucoma_surgery_codes', key='train_macs')
    
    run_data = prep_run_data(data, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options)
    if use_mac: 
        run_data['MAC'] = data['MAC']
        df_dummies = pd.get_dummies(run_data['MAC'], dummy_na=False, drop_first=False).astype(int)
        run_data_onehot = pd.concat([run_data.drop('MAC', axis=1), df_dummies], axis=1)
        
    y = data[st.session_state['idose_col_name']]
    
    if st.button('Train Model', key='train_all'):
        clf_file_name = train_model(run_data_onehot, y, balance_classes, model_name, 'ALL_MACS')
                    
        if not st.session_state.get('saved_classifiers', []): 
            st.session_state['saved_classifiers'] = []
        for mac, saved_clf in st.session_state['saved_classifiers']: 
            if clf_file_name == f'{saved_clf}_{mac}': 
                backup_file = f'{saved_clf}_overwritten-old'
                if os.path.exists(backup_file): 
                    os.remove(backup_file)
                if os.path.exists(saved_clf):
                    os.rename(saved_clf, backup_file)
                st.session_state['saved_classifiers'].remove((mac,saved_clf))
                st.session_state['saved_classifiers'].append((mac, f'{saved_clf}_overwritten-old'))
        st.session_state['saved_classifiers'].append(('ALL_MACS', clf_file_name))

        st.success(f'Model Saved as {clf_file_name}')

def train_mac_split(data): 
    beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, _ = feature_selection('mac_split_explore', all_macs=False)
    run_data = prep_run_data(data, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options)
    run_data['MAC'] = data['MAC']
    y = data[st.session_state['idose_col_name']]
    
    with st.container(): 
        mac_options = np.unique(run_data['MAC'])
        macs = st.multiselect('Select one or more MACs', mac_options)
        model_name = st.text_input(label='Model Name?', value='idose_prediction', placeholder='eg. glaucoma_surgery_codes', key='train_splits')
    
    run_data = run_data[run_data['MAC'].isin(macs)]
    run_data = run_data.drop('MAC', axis=1)
    y = y.loc[run_data.index]
    
    if st.button('Train Model', key='train_spl'):
        clf_file_name = train_model(run_data, y, balance_classes, model_name, '_'.join(macs))
        
        if not st.session_state.get('saved_classifiers', []): 
            st.session_state['saved_classifiers'] = []
        for mac, saved_clf in st.session_state['saved_classifiers']: 
            if clf_file_name == f'{saved_clf}_{mac}': 
                backup_file = f'{saved_clf}_overwritten-old'
                if os.file.exists(backup_file): 
                    os.remove(backup_file)
                os.rename(saved_clf, f'{saved_clf}_overwritten-old')
                st.session_state['saved_classifiers'].remove((mac,saved_clf))
                st.session_state['saved_classifiers'].append((mac, f'{saved_clf}_overwritten-old'))
        st.session_state['saved_classifiers'].append(('_'.join(macs), clf_file_name))
        
        st.success(f'Model Saved as {clf_file_name}')

  
def update_saved_models(): 
    if st.session_state.get('saved_classifiers', []) != []: 
        st.markdown('#### Previously Saved Models -- Click to Download .pkl File')
        macs = np.unique([clf[0] for clf in st.session_state['saved_classifiers']])
        cols = st.columns(len(macs))
        for i, mac in enumerate(macs): 
            cols[i].markdown(f'###### {mac}')
        
        for mac, clf in st.session_state['saved_classifiers']:
            idx = np.where(macs == mac)[0][0]
            col = cols[idx]
            with open(clf, 'rb') as f: 
                col.download_button(
                    label=clf[:-4],
                    data=f.read(),
                    file_name=clf, 
                    key=f'{mac}_{clf}_{random.randint(0,10000)}'
                )