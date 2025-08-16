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
from miscellaneous import center_header
import shap 
#import lime
#import lime.lime_tabular
import io
import matplotlib.pyplot as plt
from modify_npis import get_nppes_info_for_npis

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
            
            for mac, clf, feat_settings, shap in st.session_state['saved_classifiers']:
                idx = np.where(macs == mac)[0][0]
                col = cols[idx]
                if col.button(clf[:-4], key=f'predict_{mac}_{clf}', icon=MODEL_ICONS[0]): 
                    st.session_state['predict_model'] = (clf, joblib.load(clf), feat_settings, shap)
        
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
            st.session_state['predict_model'] = (filename, joblib.load(filename), joblib.load(filename)['feat_settings'], 
                                                 joblib.load(filename)['shap_explainer'])
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
                clf, feat_settings, shap = train_mac_split(data)
            elif mode == 'Train and Save model for all MACs':
                clf, feat_settings, shap = train_all_macs(data)
            else: 
                clf, feat_settings, shap = None, None, None

            if clf is not None:
                st.session_state['predict_model'] = (clf, joblib.load(clf), feat_settings, shap)
                sac.divider(label='generate prediction data', icon='boxes', align='center', color='gray', key='magic4')
                run_prediction()
        else: 
            st.error('Data must be loaded first')
            
    #sac.divider(label='run prediction', icon='magic', align='center', color='gray', key='magig1')
    if 'pred_data' not in st.session_state:
        st.session_state['pred_data'] = None
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='split_end')


def shap_explainer(explainer, pred_data): 
    pred_shap_images = {npi:None for npi in pred_data.index}
    top_features = {npi:None for npi in pred_data.index}
    top_vals = {npi:None for npi in pred_data.index}
    for npi, npi_df in pred_data.iterrows():
        row_df = npi_df.to_frame().T
        shap_values = explainer(row_df)
        
        shap_for_class = shap_values.values[0]
        top_idx = np.argsort(np.abs(shap_for_class))[::-1]
        N = 10
        top_feats = pred_data.columns[top_idx[:N]].to_list()
        top_vs = row_df[top_feats].iloc[0].to_list()
        top_features[npi] = top_feats 
        top_vals[npi] = top_vs
        
        shap_buf = io.BytesIO()
        shap.plots.waterfall(shap_values[0], show=False)
        plt.savefig(shap_buf, format='png', bbox_inches='tight')
        plt.close()
        shap_buf.seek(0)
        
        pred_shap_images[npi] = shap_buf
    
    return pred_shap_images, top_features, top_vals


# def lime_explainer(model, pred_data, all_data): 
#     explainer = lime.lime_tabular.LimeTabularExplainer(
#         training_data=all_data.values, 
#         feature_names = all_data.columns,
#         class_names=['Non iDose', 'iDose'],
#         mode='classification', 
#     )
#     pred_lime_images = {npi:None for npi in pred_data.index}
#     for npi, npi_df in pred_data.iterrows():
#         lime_exp = explainer.explain_instance(
#             data_row=npi_df.values,
#             predict_fn=model.predict_proba
#         )
        
#         lime_buf = io.BytesIO()
#         lime_exp.as_pyplot_figure()
#         plt.savefig(lime_buf, format='png', bbox_inches='tight')
#         plt.close()
#         lime_buf.seek(0)
        
#         pred_lime_images[npi] = lime_buf
    
#     return pred_lime_images


def generate_pred_data(npi_list, feat_settings):
    st.session_state['pred_npis'] = '\n'.join(npi_list)
    
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


def run_prediction(): 
    if st.session_state.get('predict_model'): 
        center_header(f'Selected Model: {st.session_state['predict_model'][0]}', 3)
        model_name = st.session_state['predict_model'][0]
        model = st.session_state['predict_model'][1]['model']
        feat_settings = st.session_state['predict_model'][2]
        shap = st.session_state['predict_model'][3]
        pred_cols = st.columns([2,0.05,1.5,0.05,1.5])
        with pred_cols[0]: 
            center_header('Upload NPI list and get CMS data', 3)
            new_npis = st.text_area("NPIs to predict on (one per line):", value=st.session_state.get('pred_npis', '') , height=400, key='pred_npi_text')
            npi_list = parse_npi_list(new_npis)
            if st.button('Generate Prediction Data', icon=':material/assignment_add:', width='stretch', key='npilist'):
                set_cancel_button()
                generate_pred_data(npi_list, feat_settings)
        
        with pred_cols[1]:
            st.markdown("<div style='height: 600px; border-left: 1px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)
         
        with pred_cols[2]: 
            center_header('Upload file with list of NPIs', 3)
            filename = st.file_uploader('File with NPIs (one per line)', type='txt')
            npi_list = []
            if filename:
                with open(filename, 'r') as infile: 
                    for i, line in enumerate(infile): 
                        line = line.strip() 
                        if len(line) == 0: 
                            continue
                        elif len(line) != 10: 
                            st.error(f'NPI on line {i} must have 10 characters')
                            st.stop()
                        else:
                            npi_list.append(line)
            if st.button('Generate Prediction Data', icon=':material/assignment_add:', width='stretch', key='npifile'):
                set_cancel_button()
                generate_pred_data(npi_list, feat_settings)
        
        with pred_cols[3]:
            st.markdown("<div style='height: 600px; border-left: 1px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)
        
        with pred_cols[4]:
            center_header('Upload file with prediction data', 3)
            sub_cols = st.columns(2)
            with sub_cols[0]:
                st.markdown('##### Upload Medscout')
                medscout_file = st.file_uploader('Upload medscout data', type=['csv','xlsx','xls'])
            with sub_cols[1]: 
                st.markdown('##### Upload CMS')
                cms_file = st.file_uploader('Upload CMS data', type=['csv', 'xlsx', 'xls'])
                
            
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
            set_norm_button()
            data = st.session_state['pred_data']
            st.text('Data and model ready for prediction')
            st.text(f'Selected Model: {model_name}')
            #st.text(f'Feature Settings: {feat_settings}')
            st.dataframe(data)
            sac.divider(label='run prediction', icon='magic', align='center', color='gray', key='magig1')
            
            explanation = st.toggle('Include Prediction Explanations', value=True)
            if st.button('Run Prediction', icon=':material/flight_takeoff:', width='stretch'):
                set_cancel_button()
                
                run_data = prep_run_data(data, feat_settings['beneficiaries'], feat_settings['services'], feat_settings['proportions'], feat_settings['totals'],
                                        feat_settings['no_time'], feat_settings['selected_options'], feat_settings['ex_options'])
                #run_data['MAC'] = data['MAC']
                        
                #st.text([col for col in run_data.columns if col not in model.feature_names_in_])
                
                if feat_settings['use_mac']: 
                    run_data['MAC'] = data['MAC']
                    df_dummies = pd.get_dummies(run_data['MAC'], dummy_na=False, drop_first=False).astype(int)
                    run_data = pd.concat([run_data.drop('MAC', axis=1), df_dummies], axis=1)
                    #run_data.drop('MAC', axis=1)
                
                #dupes = run_data.columns[run_data.columns.duplicated()].tolist()
                #run_data = run_data.loc[:, ~run_data.columns.duplicated()]
                
                for feat in model.feature_names_in_: 
                    if feat not in run_data.columns: 
                        run_data[feat] = 0
                
                st.text('Predicting on this dataset...')
                st.dataframe(run_data)
                
                cancel_button = st.empty()
                if cancel_button.button('Cancel', icon=':material/cancel:', key='pred_cancel', width='stretch'): 
                    st.stop()
                
                # Reorder run_data to exactly match the model
                run_data = run_data.loc[:, model.feature_names_in_]
                
                predictions = model.predict(run_data)
                probs = model.predict_proba(run_data)
                pred_class_props = probs.max(axis=1)
                
                shaps = {}
                #limes = {}
                if explanation: 
                    shaps, shap_feats, shap_vals = shap_explainer(shap, run_data)
                    if 'generated_df' in st.session_state and all(feat in st.session_state['generated_df'] for feat in run_data.columns): 
                        all_data = st.session_state['generated_df'][all_data.columns.to_list()]                    
                    else: 
                        feat_means = feat_settings['feature_means']
                        feat_stds = feat_settings['feature_stds']
                        all_data = pd.DataFrame() 
                        for feature in run_data.columns: 
                            if feature in feat_means and feature in feat_stds: 
                                all_data[feature] = np.random.normal(loc=feat_means[feature], scale=feat_stds[feature], size=200)
                    #limes = lime_explainer(model, run_data, all_data)            
                    
                run_data['Prediction'] = ['iDose User' if pred == True else 'Non iDose User' for pred in predictions]
                run_data['Confidence'] = [str(round(prop*100, 2))+'%' for prop in pred_class_props]
                
                out = run_data.sort_values(by=['Prediction', 'Confidence'], ascending=[False, False])
                out['NPI'] = out.index
                info_df = get_nppes_info_for_npis(out['NPI'])
                out['Name'] = info_df['Name'].values
                out['City'] = info_df['City'].values
                out['State'] = info_df['State'].values
                out['Zip'] = info_df['Zip'].values
                out['MAC'] = info_df['MAC'].values
                
                out = out[['NPI', 'Name', 'Prediction', 'Confidence', 'MAC', 'City', 'State', 'Zip']].reset_index(drop=True)
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

                col1, col2 = st.columns([3,1])
                with col2:
                    st.markdown('###')
                    out_filename = f'predictions_model-{model_name}_{formatted_datetime}.csv'
                    out.to_csv(out_filename)
                    with open(out_filename, 'rb') as f: 
                        st.download_button('Download Predictions as CSV', icon=':material/person_celebrate:', 
                                        data=f.read(), file_name=out_filename, width='stretch')
                with col1:
                    st.markdown('##### Predictions:')
                    st.dataframe(out, hide_index=True)
                    
                for npi in out['NPI']: 
                    npi_df = get_nppes_info_for_npis([npi])
                    name = npi_df['Name'].to_list()[0]
                    MAC = npi_df['MAC'].to_list()[0]
                    city = npi_df['City'].to_list()[0]
                    state = npi_df['State'].to_list()[0]
                    zipcode = npi_df['Zip'].to_list()[0]
                    prediction = out[out['NPI'] == npi]['Prediction'].iloc[0]
                    with st.expander(f'Prediction Explanation for {npi} -- {name} -- Predicted: {prediction}'): 
                        st.markdown(f'MAC: {MAC}')
                        st.markdown(f'Location: {city}, {state} {zipcode}')
                        #st.markdown(f'Feature Values For {name}')
                        #st.dataframe(pd.DataFrame([feat_settings['feature_means']]))
                        #shapcol, limecol = st.columns(2)
                        dfcol, shapcol = st.columns([4,3])
                        with shapcol:
                            #st.markdown('Mean Values For Most Influential Features')
                            # st.dataframe(pd.DataFrame([feat_settings['feature_means']])[shap_feats[npi]])

                            center_header('SHAP Explainer', 3) 
                            shaps[npi].seek(0)
                            st.image(shaps[npi])
                        with dfcol:
                            shap_feat_mean_df = pd.DataFrame([feat_settings['feature_means']])[shap_feats[npi]]
                            shap_feat_std_df = pd.DataFrame([feat_settings['feature_stds']])[shap_feats[npi]]
                            differences = []
                            for feature in shap_feats[npi]: 
                                feature_value = run_data[feature].loc[npi]
                                mean_value = shap_feat_mean_df[feature].iloc[0]
                                std = shap_feat_std_df[feature].iloc[0]
                                if feature_value < -2*std*mean_value: 
                                    differences.append('Significantly Lower')
                                elif feature_value < -1*std*mean_value: 
                                    differences.append('Moderately Lower')
                                elif feature_value > 2*std*mean_value: 
                                    differences.append('Significantly Higher')
                                elif feature_value > std*mean_value: 
                                    differences.append('Moderately Higher')
                                elif feature_value < mean_value: 
                                    differences.append('Slightly Lower')
                                elif feature_value > mean_value: 
                                    differences.append('Slightly Higher')
                                elif feature_value == mean_value: 
                                    differences.append('Same')
                                else: 
                                    differences.append('Error')
                                
                            
                            comp_df = pd.DataFrame({
                                'Feature':shap_feat_mean_df.columns,
                                f'{name} Values':run_data[shap_feats[npi]].loc[npi],
                                'Average Values':shap_feat_mean_df.iloc[0],
                                'Difference':differences
                            }).set_index('Feature', drop=True)
                            st.dataframe(comp_df)
                            
                        # with limecol: 
                        #     center_header('LIME Explainer', 3)
                        #     limes[npi].seek(0)
                        #     st.image(limes[npi])
                
                set_norm_button()
                cancel_button.empty()

    