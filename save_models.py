import streamlit as st
import numpy as np 
from load_data import check_data_loaded
import random 
from model_backend import train_model
from model_exploration import prep_run_data, feature_selection
import os
import pandas as pd
import glob
from streamlit_option_menu import option_menu
import streamlit_antd_components as sac
import random

MODEL_ICONS = [':material/network_intel_node:', ':material/automation:', ':material/network_node:', ':material/graph_2:',
               ':material/schema:', ':material/graph_5:', ':material/flowsheet:', ':material/batch_prediction:']

MAIN_COLOR = '#4682b4'
ACCENT_COLOR = '#f28c8c'
BACKGROUND = '#DCECFA'
SAGE = '#8cae9c'

def save_model():  
    st.header('Save a model for future prediction')
    
    saved_models_placeholder = st.empty()
    
    def render_saved_models(): 
        with saved_models_placeholder.container(): 
            update_saved_models()
            
    render_saved_models()
    
    if len(st.session_state.get('saved_classifiers', [])) > 0:
        if st.button('CLEAR SAVED MODELS', icon=':material/delete_forever:', width='stretch'): 
            for file in glob.glob('*pkl*'):
                os.remove(file)
            if st.session_state.get('saved_classifiers', []): 
                st.session_state['saved_classifiers'] = []
                render_saved_models()
            st.rerun()
    else: 
        st.markdown('#### No Saved Models Yet...')
                
    sac.divider(label='save a new model', icon='diagram-3', align='center', color='gray')
    if check_data_loaded(): 
        st.subheader('Train and save a new model')
        
        data = st.session_state['generated_df']
        st.success('Data Loaded -- Ready for Training')
        st.dataframe(data)
        
        mac_selection = option_menu(None, ['Train and Save model for specific MAC', 'Train and Save model for all MACs'], 
                                    icons=['geo', 'globe-americas'], orientation='horizontal',
                                    styles={
                                        'container': {'background-color': BACKGROUND},
                                        'nav-link-selected': {'background-color': SAGE, 'color':'#FFFFFF'},
                                        'nav-link': {'color': MAIN_COLOR}
                                    })
        
        if mac_selection == 'Train and Save model for specific MAC': 
            train_mac_split(data)
            render_saved_models()
            
        elif mac_selection == 'Train and Save model for all MACs':
            train_all_macs(data)
            render_saved_models()
        
    else: 
        st.error('Data must be loaded before training a model')
    
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='save_end')

def train_all_macs(data):
    clf_file_name = None
    new_shap = None 
    new_lime = None
    beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, use_mac = feature_selection('mac_split_save', all_macs=True)
    
    sac.divider(label='train all macs', icon='play-btn', align='center', color='gray', key=f'train_all_divider')
    with st.container(border=True):
        model_name = st.text_input(label='Model Name?', value='idose_prediction', placeholder='eg. glaucoma_surgery_codes', key='train_macs')
        
        run_data = prep_run_data(data, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options)
        if use_mac: 
            run_data['MAC'] = data['MAC']
            df_dummies = pd.get_dummies(run_data['MAC'], dummy_na=False, drop_first=False).astype(int)
            run_data_onehot = pd.concat([run_data.drop('MAC', axis=1), df_dummies], axis=1)
        else: 
            run_data_onehot = run_data
        y = data[st.session_state['idose_col_name']]
        
        # st.text(feat_settings)
        feat_settings = {
            'beneficiaries':beneficiaries, 'services':services, 'proportions':proportions, 'totals':totals, 'no_time':no_time,
            'balance_classes':balance_classes, 'selected_options':selected_options, 'ex_options':ex_options, 'use_mac':use_mac, 
            'start_year':st.session_state.get('start_year', None), 'feature_means':run_data_onehot.mean().to_dict(), 'feature_stds':run_data_onehot.std().to_dict()
        }
        
        if st.button('Train and Save Model', key='train_all', width='stretch', icon=':material/train:'):
            #st.dataframe(run_data_onehot)        
            clf_file_name, new_shap = train_model(run_data_onehot, y, balance_classes, model_name, 'ALL_MACS', feat_settings)
                        
            # total_dupes = 1
            if not st.session_state.get('saved_classifiers', []): 
                st.session_state['saved_classifiers'] = []
            # else:
            #     for _, clf_name, _, _ in st.session_state['saved_classifiers']: 
            #         if f'{clf_file_name}_overwritten' in clf_name: 
            #             total_dupes += 1
                        
            for mac, saved_clf, saved_feat_settings, shap in st.session_state['saved_classifiers']: 
                if clf_file_name == saved_clf: 
                    backup_file = f'{saved_clf}_overwritten{random.randint(0,10000)}'
                    st.session_state['saved_classifiers'] = [vals for vals in st.session_state['saved_classifiers'] if vals[1] != saved_clf]
                    st.session_state['saved_classifiers'].append((mac, backup_file, saved_feat_settings, shap))
            st.session_state['saved_classifiers'].append(('ALL_MACS', clf_file_name, feat_settings, new_shap))

            st.success(f'Model Saved as {clf_file_name}')
    
    return clf_file_name, feat_settings, new_shap


def train_mac_split(data): 
    clf_file_name = None
    new_shap = None
    beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, _ = feature_selection('mac_split_explore', all_macs=False)
    run_data = prep_run_data(data, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options)
    run_data['MAC'] = data['MAC']
    y = data[st.session_state['idose_col_name']]
    
    sac.divider(label='train mac split', icon='play-btn', align='center', color='gray', key=f'train_split_divider')
    with st.container(border=True): 
        mac_options = np.unique(run_data['MAC'])
        macs = st.multiselect('Select one or more MACs', mac_options)
        model_name = st.text_input(label='Model Name?', value='idose_prediction', placeholder='eg. glaucoma_surgery_codes', key='train_splits')
    
        run_data = run_data[run_data['MAC'].isin(macs)]
        run_data = run_data.drop('MAC', axis=1)
        y = y.loc[run_data.index]
        
        feat_settings = {
            'beneficiaries':beneficiaries, 'services':services, 'proportions':proportions, 'totals':totals, 'no_time':no_time,
            'balance_classes':balance_classes, 'selected_options':selected_options, 'ex_options':ex_options, 'use_mac':False, 
            'start_year':st.session_state.get('start_year', None), 'feature_means':run_data.mean().to_dict(), 'feature_stds':run_data.std().to_dict()
        }
        
        if st.button('Train and Save Model', key='train_spl', width='stretch', icon=':material/train:'):
            clf_file_name, new_shap = train_model(run_data, y, balance_classes, model_name, '_'.join(macs), feat_settings)
            
            # total_dupes = 1
            if not st.session_state.get('saved_classifiers', []): 
                st.session_state['saved_classifiers'] = []
            # else:
            #     for _, clf_name, _, _ in st.session_state['saved_classifiers']: 
            #         if f'{clf_file_name}_overwritten' in clf_name: 
            #             total_dupes += 1
                        
            for mac, saved_clf, saved_feat_settings, shap in st.session_state['saved_classifiers']: 
                if clf_file_name == f'{saved_clf}_{mac}': 
                    backup_file = f'{saved_clf}_overwritten{random.randint(1,10000)}'
                    st.session_state['saved_classifiers'] = [vals for vals in st.session_state['saved_classifiers'] if vals[1] != f'{saved_clf}_{mac}']
                    st.session_state['saved_classifiers'].append((mac, backup_file, saved_feat_settings, shap))
            st.session_state['saved_classifiers'].append(('_'.join(macs), clf_file_name, feat_settings, new_shap))
            
            st.success(f'Model Saved as {clf_file_name}')

    return clf_file_name, feat_settings, new_shap

  
def update_saved_models(): 
    #st.session_state['saved_classifiers'] = []
    if st.session_state.get('saved_classifiers', []) != []: 
        st.markdown('#### Previously Saved Models -- Click to Download .pkl File')
        macs = np.unique([clf[0] for clf in st.session_state['saved_classifiers']])
        cols = st.columns(len(macs))
        for i, mac in enumerate(macs): 
            cols[i].markdown(f'###### {mac}')
        
        for mac, clf, feat_settings, shap in st.session_state['saved_classifiers']:
            idx = np.where(macs == mac)[0][0]
            col = cols[idx]
            with open(clf, 'rb') as f: 
                col.download_button(
                    label=clf[:-4],
                    data=f.read(),
                    file_name=clf, 
                    key=f'{mac}_{clf}_{random.randint(0,10000)}',
                    icon=MODEL_ICONS[0]
                )