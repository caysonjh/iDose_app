import streamlit as st
import base64 
import pandas as pd 
from code_groupings import new_feats
from url_info import MOST_UP_TO_DATE_CMS_YEAR
from model_backend import run_model_all_macs, run_model_mac_split
import numpy as np
from miscellaneous import make_progress_updater
from load_data import check_data_loaded, IDOS_VAL_COLUMN
import os
from streamlit_option_menu import option_menu
import streamlit_antd_components as sac
from storage_interaction import write_user_environment
import random
from streamlit_extras.dataframe_explorer import dataframe_explorer

MAIN_COLOR = '#4682b4'
ACCENT_COLOR = '#f28c8c'
BACKGROUND = '#DCECFA'
SAGE = '#8cae9c'


metric_helps = {
    'METRICS':'This table highlights the performance of the model trained with 80% of the dataset on the 20% that was held out as a test set. For more information ' +
    'on what each of the metrics mean, see [here](https://medium.com/@piyushkashyap045/understanding-precision-recall-and-f1-score-metrics-ea219b908093).', 
    'CM_IMAGES':"This image displays a confusion matrix. The y-axis shows true labels, and the x-axis shows predicted labels. The numbers in the top left and " + 
    "bottom right squares indicate where the model's prediction matched with the true value, so a higher concentration in those quadrants indicates better " + 
    "performance. For more information see [here](https://www.v7labs.com/blog/confusion-matrix-guide).",
    'PAR_DEP':"This image displays partial dependence plots. These images highlight the top 6 features that had the most importance in training the model. " + 
    "The plot shows how the model prediction is pushed (higher towards iDose, lower away from iDose) as the feature value changes. A positive slope line would " +
    "indicate that higher values of that feature leads to a higher liklihood of being iDose. A non-linear line indicates a complex relationship. " + 
    "For more information see [here](https://medium.com/data-science-in-your-pocket/understanding-partial-dependence-plots-pdps-415346b7e7f1)", 
    'SHAP_SUMMARIES':"This image shows the SHAP summary of the features. Each dot on the figure corresponds to a physician in the training set. A red dot " + 
    "corresponds to a high value for that feature, and a blue dot corresponds to a low value for the feature. A high concentration of red dots on the left " + 
    "side of the line indicate that having a high value for that feature makes one less likely to be predicted as an iDose user. Similarly, a high concentration " + 
    "of red dots on the right side of the center line indicates that a higher value for that feature makes one more likely to be an iDose user. " + 
    "For more information see [here](https://www.aidancooper.co.uk/a-non-technical-guide-to-interpreting-shap-analyses/)."
}


def model_explore():
    st.header('Test Various Model Configurations and View Metrics/Feature Insights')
    if check_data_loaded():
        data = st.session_state.generated_df
        st.success('Data Loaded -- Ready for Training')
        filtered_data = dataframe_explorer(data, case=False)
        st.dataframe(data, use_container_width=True)
        
        sac.divider(label='explore models', icon='controller', align='center', color='gray')
        
        
        mode = option_menu(None, ['Run Model and Split by MAC', 'Run Model with MAC as Feature'], 
                            icons=['geo', 'globe-americas'], orientation='horizontal',
                            styles={
                                'container': {'background-color': BACKGROUND},
                                'nav-link-selected': {'background-color': SAGE, 'color':'#FFFFFF'},
                                'nav-link': {'color': MAIN_COLOR}
                            })        

        if mode == 'Run Model and Split by MAC':
            run_mac_split()
        elif mode == 'Run Model with MAC as Feature':
            run_all_macs()
        
    else: 
        st.error('Data must be loaded before training a model')


def prep_run_data(df, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options): 
    run_data = df
    #run_data.set_index('NPI', inplace=True)
    run_data = run_data.replace('NO', 0, regex=False)
    run_data = run_data.replace('YES', 1, regex=False)
    
    # if not beneficiaries and not services and any('Beneficiaries' in col for col in run_data.columns) or any('Services' in col for col in run_data.columns): 
    #     st.error('One of Beneficiaries or Services must be selected')
    #     st.stop()
    if not beneficiaries: 
        run_data = run_data.drop([col for col in run_data.columns if 'Beneficiaries' in col], axis=1)
    elif not services: 
        run_data = run_data.drop([col for col in run_data.columns if 'Services' in col], axis=1)
    
    run_data = run_data.drop([val for val in run_data.columns if not any(feat in val for feat in selected_options) and val not in ex_options.keys()], axis=1)
    
    for ex in ex_options.keys(): 
        if ex == 'Time_Features': 
            if no_time:
                continue
            if not ex_options[ex]: 
                run_data = run_data.drop([col for col in run_data.columns if any(time_feat in col for time_feat in ['Mean', 'Median', 'Standard_Deviation', 'Range', 'Rate_of_Change'])], axis=1)
            continue
        if not ex_options[ex]: 
            run_data = run_data.drop(ex, axis=1)
      
    true_options = [option for option in selected_options if selected_options[option]]  
    feature_cols = [col for col in run_data.columns if any(option in col for option in true_options)] 
    extra_time_cols = [col for col in feature_cols if any(time_feat in col for time_feat in ['Mean','Median','Standard_Deviation','Range', 'Rate_of_Change'])]
    if ex_options['Time_Features']: 
        time_df = run_data[extra_time_cols]
    feature_cols = [col for col in feature_cols if col not in extra_time_cols]
    other_cols = [col for col in ex_options.keys() if ex_options[col] and col != 'Time_Features']
    other_df = run_data[other_cols]
    run_data = run_data[feature_cols]
    
    if proportions: 
        feature_df = run_data[feature_cols].astype(float)
        prop_df = feature_df.div(feature_df.sum(axis=1), axis=0)
        prop_df.columns = [f'{col} Proportion' for col in prop_df.columns]
        if not totals: 
            run_data = pd.merge(prop_df, other_df, left_index=True, right_index=True)
    if totals: 
        if proportions: 
            run_data = pd.merge(run_data[feature_cols].astype(float), prop_df, left_index=True, right_index=True)
            run_data = pd.merge(run_data, other_df, left_index=True, right_index=True)
        else: 
            run_data = pd.merge(run_data[feature_cols], other_df, left_index=True, right_index=True)
    
    if ex_options['Time_Features']: 
        run_data = pd.merge(run_data, time_df, left_index=True, right_index=True)
          
    run_data = run_data.fillna(0)
    run_data = run_data.astype(float)
    
    return run_data


def show_pdf(file_path): 
    st.text('Note, pdf will not generate if it is too large. Download instead to view:')
    with open(file_path, "rb") as f:
        base64_path = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_path}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True) 
    
    with open(file_path, 'rb') as f: 
        st.download_button('Download PDF', f, file_name=file_path, icon=':material/download:')
    
    
def feature_selection(key_header, all_macs): 
    sac.divider(label='feature selection', icon='toggles', align='center', color='gray', key=f'{key_header}_feat_divider')
    df = st.session_state['generated_df']
    ex_options = {
        'Time_Features':False, 
        'Sole_Prop':False,
        'Min_Dist':False,
        'Enum_Time':False
    }    
    ex_descriptions = {
        'Time_Features': 'Mean, median, standard deviation, range, rate of change over selected years',
        'Sole_Prop': 'If the physician is a sole proprietor',
        'Min_Dist': 'The nearest distance to an iDose user',
        'Enum_Time': 'Time since the NPI was created'
    }   
    
    if st.session_state.get('default_settings', []): 
        default_settings = st.session_state['default_settings']
    else:
        default_settings = {'beneficiaries':False, 'services':True, 'totals':False, 'proportions':True, 'balance_classes':True,
                            'selected_options':{feat:True for feat in new_feats.keys() if any(feat in val for val in df.columns)}, 
                            'ex_options':{key:True for key in ex_options.keys()}, 'use_mac':True}
    
    with st.container(border=True):
        if any('Services' in col for col in df.columns) and any('Beneficiaries' in col for col in df.columns):  
            st.markdown('##### Select Data Source (Or Both):')
            beneficiaries = st.checkbox('Beneficiaries', key=f'{key_header}_beneficiaries', value=default_settings['beneficiaries'])
            services = st.checkbox('Services', key=f'{key_header}_services', value=default_settings['services'])
            st.markdown("<br>", unsafe_allow_html=True)
        else: 
            beneficiaries, services = False, False
    
        st.markdown('##### Select Base Features:')
        selected_options = {}
        cols = st.columns(3)
        default_options = default_settings['selected_options']
        options = [feat for feat in new_feats.keys() if any(feat in val for val in df.columns)]
        for feat in options: 
            if feat not in default_options: 
                default_options[feat] = True
        # if len(default_options) != len(options):
        #     default_options = [True] * len(options)
        
        for i, option in enumerate(options): 
            col = cols[i % 3]
            selected_options[option] = col.checkbox(option, value=default_options[option], key=f'{key_header}_{option}')
            # if col.checkbox(option, value=default_options[i], key=f'{key_header}_{option}'): 
            #     selected_options[option] = True
        st.markdown("<br>", unsafe_allow_html=True)
            
        st.markdown('##### Select Data Type (Or Both):')
        proportions = st.checkbox('Proportions (of all selected features)', value=default_settings['proportions'], key=f'{key_header}_proportions')
        totals = st.checkbox('Totals (raw total for each selected feature)', key=f'{key_header}_totals', value=default_settings['totals'])
        st.markdown("<br>", unsafe_allow_html=True)
    
        st.markdown('##### Extra Feature Options:')
        no_time = False
        for i, ex in enumerate(ex_options.keys()): 
            if ex == 'Time_Features': 
                if 'start_year' in st.session_state: 
                    if st.session_state.start_year == MOST_UP_TO_DATE_CMS_YEAR: 
                        no_time = True
                        continue
                
                if not any(str(year) in col for year in range(2013,int(MOST_UP_TO_DATE_CMS_YEAR)) for col in df.columns) and not any('Range' in col for col in df.columns): 
                    no_time = True
                    continue
            
            if st.checkbox(f'{ex} -- {ex_descriptions[ex]}', key=f'{key_header}_{ex}', value=default_settings['ex_options'][ex]): 
                ex_options[ex] = True
        st.markdown("<br>", unsafe_allow_html=True)
    
        st.markdown('##### Model Options:')
        balance_classes = st.toggle('Oversample the under represented class to balance the two classes and get better predictions', value=True, key=f'{key_header}_balance')
        st.markdown("<br>", unsafe_allow_html=True)
        
        if all_macs: 
            st.markdown('##### MAC Options:')
            use_mac = st.toggle('MAC -- Use MACs as features (onehot encoded)', value=default_settings['use_mac'])
        else: 
            use_mac = False
            
    
    new_settings = {'beneficiaries':beneficiaries, 'services':services, 'totals':totals, 'proportions':proportions,
                                            'balance_classes':balance_classes, 'selected_options':selected_options, 'ex_options':ex_options, 'use_mac':use_mac}
    
    st.session_state['default_settings'] = new_settings
    st.session_state['user_environment']['feature_settings'] = new_settings
    write_user_environment()
    
    return beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, use_mac


def run_mac_split():     
    #TODO Make it so that previously selected options will be saved
    
    beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, _ = feature_selection('mac_split_explore', all_macs=False)
    feat_settings = {'beneficiaries': beneficiaries, 'services': services, 'totals':totals, 'proportions':proportions, 'no_time':no_time, 'balance_classes':balance_classes,
                    'selected_options':selected_options, 'ex_options':ex_options, 'use_mac':False, 'start_year':st.session_state['start_year'],
    }
    
    sac.divider(label='run mac split', icon='play-btn', align='center', color='gray', key=f'mac_split_divider')
    
    with st.container(border=True): 
        model_name = st.text_input(label='Model Name?', value='idose_prediction', placeholder='eg. glaucoma_surgery_codes')

        if st.button('Run Model', key='mac_split', icon=':material/sprint:', width='stretch'): 
            df = st.session_state['generated_df']
                
            run_data = prep_run_data(df, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options)
            run_data['MAC'] = df['MAC']
            y = df[st.session_state['idose_col_name']]
            
            feat_settings['feature_means'] = run_data.drop('MAC',axis=1).mean().to_dict()
            feat_settings['feature_stds'] = run_data.drop('MAC',axis=1).std().to_dict()
            
            st.text('Running model with this dataset...')   
            run_data_filtered = dataframe_explorer(run_data, case=False)
            st.dataframe(run_data_filtered, use_container_width=True)
            
            progress_reporter = make_progress_updater(len(np.unique(run_data['MAC'])))
            
            mac_clfs, pdf_report, web_info = run_model_mac_split(run_data, y, balance_classes, progress_reporter, model_name, feat_settings)
            mac_clf_w_feats = [(mac_clf[0], mac_clf[1], {
                'beneficiaries':beneficiaries, 'services':services, 'proportions':proportions, 'totals':totals, 'no_time':no_time, 
                'balance_classes':balance_classes, 'selected_options':selected_options, 'ex_options':ex_options, 'use_mac':False, 'start_year':st.session_state.get('start_year', None)
            }, mac_clfs[2]) for mac_clf in mac_clfs]
                        
            # total_dupes = 1
            # if not st.session_state.get('saved_classifiers', []): 
            #     st.session_state['saved_classifiers'] = []
            # else:
            #     for _, clf_name, _, _ in st.session_state['saved_classifiers']: 
            #         for _, clf_file_name, _ in mac_clfs:
            #             if f'{clf_file_name}_overwritten' in clf_name: 
            #                 total_dupes += 1
                
            for mac, saved_clf, saved_feat_settings, shap in st.session_state['saved_classifiers']: 
                for mac_clf in mac_clfs: 
                    if mac_clf[1] == saved_clf: 
                        backup_file = f'{saved_clf}_overwritten{random.randint(1,10000)}'
                        st.session_state['saved_classifiers'] = [vals for vals in st.session_state['saved_classifiers'] if vals[1] != saved_clf]
                        st.session_state['saved_classifiers'].append((mac, backup_file, saved_feat_settings, shap))
            st.session_state['saved_classifiers'].extend(mac_clf_w_feats)
            
            st.markdown('##### FEATURE INFO')
            feat_dict = web_info[list(web_info.keys())[0]]['FEATURE_INFO']
            st.dataframe(feat_dict, width=2000)
            
            for mac in np.unique(run_data['MAC']): 
                with st.expander(f'{mac} Results'): 
                    st.markdown(f'##### CLASS_SUMMARY')
                    st.markdown(f'###### {web_info[mac]['CLASS_SUMMARY']}')
                    cols = st.columns(3)
                    for i, metric in enumerate(web_info[mac].keys()): 
                        if metric == 'FEATURE_INFO' or metric == 'CLASS_SUMMARY': 
                            continue 
                        elif metric == 'METRICS':
                            col = cols[i%3]
                            with col.popover(f'##### {metric}'):
                                st.markdown(f'{metric_helps[metric]}')
                            col.table(web_info[mac][metric])
                        else:
                            col = cols[i%3]
                            with col.popover(f'##### {metric}'):
                                st.markdown(f'{ metric_helps[metric]}')
                            col.image(web_info[mac][metric])
            
            
            #st.text(pdf_report)
            with st.expander('Full PDF Report'):
                show_pdf(pdf_report)
                
                with open(pdf_report, 'rb') as f: 
                    pdf_bytes = f.read()
                    
                # st.download_button(
                #     label="Download PDF Report",
                #     data=pdf_bytes,
                #     file_name=pdf_report,
                #     mime='application/pdf'
                # )
            
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='split_end')
    
    
def run_all_macs(): 
    beneficiaries, services, proportions, totals, no_time, balance_classes, selected_options, ex_options, use_mac  = feature_selection('mac_feature_explore', all_macs=True)
    feat_settings = {'beneficiaries': beneficiaries, 'services': services, 'totals':totals, 'proportions':proportions, 'no_time':no_time, 'balance_classes':balance_classes,
                     'selected_options':selected_options, 'ex_options':ex_options, 'use_mac':use_mac, 'start_year':st.session_state['start_year'],
    }
    
    sac.divider(label='run all macs', icon='play-btn', align='center', color='gray', key=f'all_mac_divider')
    with st.container(border=True):
        model_name = st.text_input(label='Model Name?', value='idose_prediction', placeholder='eg. glaucoma_surgery_codes', key='feature_mac_name')

        if st.button('Run Model', key='mac_feature', icon=':material/sprint:', width='stretch'):
            #st.session_state['saved_classifiers'] = []
            
            df = st.session_state['generated_df']                 
            run_data = prep_run_data(df, beneficiaries, services, proportions, totals, no_time, selected_options, ex_options)
            if use_mac: 
                run_data['MAC'] = df['MAC']
                df_dummies = pd.get_dummies(run_data['MAC'], dummy_na=False, drop_first=False).astype(int)
                run_data_onehot = pd.concat([run_data.drop('MAC', axis=1), df_dummies], axis=1)
            else:
                run_data_onehot = run_data
            y = df[st.session_state['idose_col_name']]
        
            feat_settings['feature_means'] = run_data_onehot.mean().to_dict()
            feat_settings['feature_stds'] = run_data_onehot.std().to_dict()
                                    
            clf_file_name, pdf_report, web_info, new_shap = run_model_all_macs(run_data_onehot, y, balance_classes, model_name, feat_settings)
            
            st.success('Model training finished!')
            
            # total_dupes = 1
            # if not st.session_state.get('saved_classifiers', []): 
            #     st.session_state['saved_classifiers'] = []
            # else:
            #     for _, clf_name, _, _ in st.session_state['saved_classifiers']: 
            #         if f'{clf_file_name}_overwritten' in clf_name: 
            #             total_dupes += 1
                
            for mac, saved_clf, saved_feat_settings, shap in st.session_state['saved_classifiers']: 
                if clf_file_name == f'{saved_clf}_{mac}': 
                    backup_file = f'{saved_clf}_overwritten{random.randint(1,10000)}'
                    st.session_state['saved_classifiers'] = [vals for vals in st.session_state['saved_classifiers'] if vals[1] != f'{saved_clf}_{mac}']
                    st.session_state['saved_classifiers'].append((mac, backup_file, saved_feat_settings, shap))
            st.session_state['saved_classifiers'].append(('ALL_MACS', clf_file_name, feat_settings, new_shap))
            
            st.markdown('##### CLASS SUMMARIES')
            st.markdown(f'###### {web_info['ALL_MACS']['CLASS_SUMMARY']}')
            st.markdown('##### FEATURE INFO')
            feat_dict = web_info['ALL_MACS']['FEATURE_INFO']
            st.dataframe(feat_dict, width=2000)
            
            with st.expander('ALL MAC Results'): 
                cols = st.columns(3)
                for i, metric in enumerate(web_info['ALL_MACS'].keys()): 
                    if metric == 'FEATURE_INFO' or metric == 'CLASS_SUMMARY': 
                        continue 
                    elif metric == 'METRICS':
                        col = cols[i%3]
                        with col.popover(f'##### {metric}'):
                            st.markdown(f'{metric_helps[metric]}')
                        col.table(web_info['ALL_MACS'][metric])
                    else:
                        col = cols[i%3]
                        with col.popover(f'##### {metric}'):
                            st.markdown(f'{ metric_helps[metric]}')
                        col.image(web_info['ALL_MACS'][metric])
            
            with st.expander('Full PDF Report'): 
                show_pdf(pdf_report)
                
                with open(pdf_report, 'rb') as f: 
                    pdf_bytes = f.read() 
                    
                # st.download_button(
                #     label='Download PDF Report',
                #     data=pdf_bytes,
                #     file_name=pdf_report,
                #     mime='application/pdf'
                # )
    
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray')

