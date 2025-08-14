import streamlit as st 
from streamlit_option_menu import option_menu
from save_models import update_saved_models, train_all_macs, train_mac_split
import joblib
import random
import numpy as np 
from load_data import check_data_loaded, update_info, make_progress_updater
import pandas as pd
from model_backend import get_code_data_from_cms, get_drug_data_from_cms, get_nppes_info, format_cms_data, format_uploaded_data
from model_exploration import feature_selection, prep_run_data
from model_backend import IDOS_VAL_COLUMN, MOST_UP_TO_DATE_CMS_YEAR
from modify_npis import parse_npi_list
import re
from datetime import datetime
import streamlit_antd_components as sac
from miscellaneous import set_cancel_button, set_norm_button

MODEL_ICONS = [':material/network_intel_node:', ':material/automation:', ':material/network_node:', ':material/graph_2:',
               ':material/schema:', ':material/graph_5:', ':material/flowsheet:', ':material/batch_prediction:']

MAIN_COLOR = '#4682b4'
ACCENT_COLOR = '#f28c8c'
BACKGROUND = '#DCECFA'
SAGE = '#8cae9c'

def load_prediction(): 
    st.header('Test Various Model Configurations and View Metrics/Feature Insights')

    selected = option_menu(None, ['Use Saved Model', 'Upload Model From File', 'Train New Model'], 
                           icons=['life-preserver', 'paperclip', 'airplane-engines'], orientation='horizontal',
                           styles={
                                'container': {'background-color': BACKGROUND},
                                'nav-link-selected': {'background-color': SAGE, 'color':'#FFFFFF'},
                                'nav-link': {'color': MAIN_COLOR}
                            })
        
    if selected == 'Use Saved Model': 
        model_file = None
        if st.session_state.get('saved_classifiers', []): 
            st.markdown('#### Previously Saved Models -- Click a file to use for prediction')
            macs = np.unique([clf[0] for clf in st.session_state['saved_classifiers']])
            cols = st.columns(len(macs))
            for i, mac in enumerate(macs): 
                cols[i].markdown(f'###### {mac}')
            
            for mac, clf, feat_settings in st.session_state['saved_classifiers']:
                idx = np.where(macs == mac)[0][0]
                col = cols[idx]
                if col.button(clf[:-4], key=f'predict_{mac}_{clf}', icon=MODEL_ICONS[0]): 
                    st.session_state['predict_model'] = (clf, joblib.load(clf), feat_settings)
        
            if st.session_state.get('predict_model'): 
                sac.divider(label='generate prediction data', icon='boxes', align='center', color='gray', key='magic2')
                run_prediction()
        
        else: 
            st.error('No Saved Models')
                
                
    if selected == 'Upload Model From File': 
        filename = None
        model_file = st.file_uploader("Upload Classifier File", type=['pkl'])
        if model_file: 
            filename = model_file.name.lower()
            feat_settings = None
            st.session_state['predict_model'] = (filename, joblib.load(filename), joblib.load(filename)['feat_settings'])
            st.success('Model loaded successfully!')
            sac.divider(label='generate prediction data', icon='boxes', align='center', color='gray', key='magic3')
            run_prediction()
    
    if selected == 'Train New Model': 
        if check_data_loaded(): 
            st.subheader('Train and save a new model')
            
            data = st.session_state['generated_df']
            st.success('Data Loaded -- Ready for Training')
            st.dataframe(data)
            
            mode = option_menu(None, ['Train and Save model for specific MAC(s)', 'Train and Save model for all MACs'], 
                                icons=['geo', 'globe-americas'], orientation='horizontal', 
                                styles={
                                'container': {'background-color': BACKGROUND},
                                'nav-link-selected': {'background-color': SAGE, 'color':'#FFFFFF'},
                                'nav-link': {'color': MAIN_COLOR}
                                })        

            if mode == 'Train and Save model for specific MAC(s)':
                clf, feat_settings = train_mac_split(data)
            elif mode == 'Train and Save model for all MACs':
                clf, feat_settings = train_all_macs(data)
            else: 
                clf, feat_settings = None, None

            if clf is not None:
                st.session_state['predict_model'] = (clf, joblib.load(clf), feat_settings)
                sac.divider(label='generate prediction data', icon='boxes', align='center', color='gray', key='magic4')
                run_prediction()
        else: 
            st.error('Data must be loaded first')
            
    #sac.divider(label='run prediction', icon='magic', align='center', color='gray', key='magig1')
    if 'pred_data' not in st.session_state:
        st.session_state['pred_data'] = None
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='split_end')


def run_prediction(): 
    if st.session_state.get('predict_model'): 
        st.markdown(f'### Selected Model: {st.session_state['predict_model'][0]}')
        model_name = st.session_state['predict_model'][0]
        model = st.session_state['predict_model'][1]['model']
        feat_settings = st.session_state['predict_model'][2]
        pred_cols = st.columns([3,0.2,2])
        with pred_cols[0]: 
            st.subheader('Upload NPI list and get CMS data')
            new_npis = st.text_area("NPIs to predict on (one per line):", value=st.session_state.get('pred_npis', '') , height=400, key='pred_npi_text')
            
            if st.button('Generate prediction data', icon=':material/assignment_add:', width='stretch'):
                npi_list = parse_npi_list(new_npis)
                st.session_state['pred_npis'] = '\n'.join(npi_list)
                
                set_cancel_button()
                cancel_button = st.empty()
                if cancel_button.button('Cancel', icon=':material/cancel:', width='stretch'):
                    st.stop()
                    
                train_list, cpt_codes, drug_list, idose_npis = update_info()
                
                progress_updater, progress_cleaner = make_progress_updater(len(npi_list)*3)
                
                df1, cpt_missing = get_code_data_from_cms(npi_list, cpt_codes, str(feat_settings['start_year']), progress_updater, 0)
                cpt_status = st.empty()
                cpt_status.success('Completed fetching CPT code data')
                df2, drug_missing = get_drug_data_from_cms(npi_list, drug_list, str(feat_settings['start_year']), progress_updater, len(npi_list))
                drug_status = st.empty()
                drug_status.success('Completed fetching prescription data')
                miss_warn = st.empty() 
                cms_missing = list(dict.fromkeys(cpt_missing+drug_missing)) 
                if len(cpt_missing) > 0: 
                    miss_warn.warning(f'NPIS: {cms_missing} were not found in CMS for selected years, and will be ignored')
                
                df3, nppes_missing = get_nppes_info(npi_list, progress_updater, len(npi_list)*2)
                nppes_status = st.empty()
                nppes_status.success('Completed fetching NPPES data')
                missings = st.empty()
                if len(nppes_missing) > 0:
                    missings.warning(f'NPIS: {nppes_missing} were not found in NPPES and will be ignored')
                    
                progress_cleaner()
                npi_list = [npi for npi in npi_list if npi not in cms_missing and npi not in nppes_missing]
                
                df_temp = pd.merge(df1, df2, on=['NPI'])
                all_data = pd.merge(df_temp, df3, on=['NPI'])
                all_data = all_data[all_data['NPI'].isin(npi_list)]
                
                cpt_status.empty()
                drug_status.empty()
                nppes_status.empty() 
                
                idose_zips = pd.read_csv('idose_zips.csv', dtype={'ZIP':str})
                prep_data_text = st.empty() 
                prep_data_text.text('Formatting Data...')
                all_data = format_cms_data(all_data, feat_settings['start_year'], MOST_UP_TO_DATE_CMS_YEAR, idose_zips['ZIP'])
                        
                prep_data_text.empty()
            
                st.session_state['pred_data'] = all_data
                cancel_button.empty()
                set_norm_button()
                
        with pred_cols[1]:
            st.markdown("<div style='height: 600px; border-left: 1px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)
        
        with pred_cols[2]:
            st.subheader('Upload file with prediction data')
            sub_cols = st.columns(2)
            with sub_cols[0]:
                st.markdown('##### Upload Medscout Data')
                medscout_file = st.file_uploader('Upload medscout data to predict on', type=['csv','xlsx','xls'])
            with sub_cols[1]: 
                st.markdown('##### Upload CMS Data')
                cms_file = st.file_uploader('Upload CMS data to predict on', type=['csv', 'xlsx', 'xls'])
                
            
            if medscout_file: 
                data_file = medscout_file
                medscout = True 
            elif cms_file: 
                data_file = cms_file 
                medscout = False 
            else: 
                data_file = None
                
            st.markdown("""<style>div.stButton > button {display: block;margin: 0 auto;}</style>""", unsafe_allow_html=True)

            if st.button('Load File', icon=':material/upload_file:', width='stretch'): 
                if data_file is not None: 
                    filename = data_file.name.lower() 
                    try:
                        if filename.endswith('.csv'):
                            df = pd.read_csv(data_file, dtype={'NPI':str})
                        elif filename.endswith(('.xlsx','.xls')):
                            df = pd.read_excel(data_file)
                        else:
                            st.error('Unsupported File Format')       
                        load_success = st.empty()
                        load_success.success(f"Successfully Loaded: {data_file.name}")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                        
                    if medscout:
                        text = st.text('Formatting data, this may take a minute...')
                        upload_updater, upload_cleaner = make_progress_updater(len(df)*2+4)
                        years = [int(MOST_UP_TO_DATE_CMS_YEAR)]
                        for col in df.columns: 
                            match = re.search(r'\b(19|20)\d{2}\b', col)
                            if match: 
                                years.append(int(match.group()))
                        start_year = min(years)
                        
                        if str(start_year) != str(feat_settings('start_year')): 
                            st.error('Loaded data start year does not match model start year')
                            st.stop() 
                        
                        idose_zips = pd.read_csv('idose_zips.csv', dtype={'ZIP':str})
                        all_data = format_uploaded_data(df, start_year, MOST_UP_TO_DATE_CMS_YEAR, idose_zips['ZIP'], upload_updater)
                        st.session_state['pred_data'] = all_data
                        upload_cleaner()
                        text.empty()
                    else:
                        st.session_state['pred_data'] = df
                
                    load_success.empty()
                    
                else: 
                    if medscout_file and cms_file: 
                        st.error('Load one file at a time')
                    else:
                        st.error('Load a file first') 
                       
        if st.session_state.get('pred_data') is not None: 
            data = st.session_state['pred_data']
            st.text('Data and model ready for prediction')
            st.text(f'Selected Model: {model_name}')
            st.text(f'Feature Settings: {feat_settings}')
            st.dataframe(data)
            sac.divider(label='run prediction', icon='magic', align='center', color='gray', key='magig1')
            
            if st.button('Run Prediction', icon=':material/flight_takeoff:', width='stretch'):
                run_data = prep_run_data(data, feat_settings['beneficiaries'], feat_settings['services'], feat_settings['proportions'], feat_settings['totals'],
                                        feat_settings['no_time'], feat_settings['selected_options'], feat_settings['ex_options'])
                #run_data['MAC'] = data['MAC']
                        
                #st.text([col for col in run_data.columns if col not in model.feature_names_in_])
                
                if feat_settings['use_mac']: 
                    run_data['MAC'] = data['MAC']
                    df_dummies = pd.get_dummies(run_data['MAC'], dummy_na=False, drop_first=False).astype(int)
                    run_data = pd.concat([run_data.drop('MAC', axis=1), df_dummies], axis=1)
                
                #dupes = run_data.columns[run_data.columns.duplicated()].tolist()
                #run_data = run_data.loc[:, ~run_data.columns.duplicated()]
                
                for feat in model.feature_names_in_: 
                    if feat not in run_data.columns: 
                        run_data[feat] = 0
                
                st.text('Predicting on this dataset...')
                st.dataframe(run_data)
                
                # Reorder run_data to exactly match the model
                run_data = run_data.loc[:, model.feature_names_in_]
                
                predictions = model.predict(run_data)
                probs = model.predict_proba(run_data)
                pred_class_props = probs.max(axis=1)
                
                run_data['Prediction'] = predictions
                run_data['Probability'] = pred_class_props
                
                out = run_data.sort_values(by=['Prediction', 'Probability'], ascending=[False, False])
                out['NPI'] = out.index
                
                out = out[['NPI', 'Prediction', 'Probability']].reset_index(drop=True)
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

                col1, col2 = st.columns(2)
                with col2:
                    st.markdown('###')
                    out_filename = f'predictions_{formatted_datetime}.csv'
                    out.to_csv(out_filename)
                    with open(out_filename, 'rb') as f: 
                        st.download_button('Download Predictions as CSV', icon=':material/person_celebrate:', 
                                        data=f.read(), file_name=out_filename, width='stretch')
                with col1:
                    st.markdown('##### Predictions:')
                    st.dataframe(out)

    