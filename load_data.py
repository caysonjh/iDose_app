import streamlit as st 
from model_backend import get_drug_data_from_cms, get_nppes_info, format_cms_data, get_code_data_from_cms, format_uploaded_data
from url_info import MOST_UP_TO_DATE_CMS_YEAR
import pandas as pd
from model_backend import IDOS_VAL_COLUMN
import re
from datetime import datetime
from itertools import chain
from code_groupings import new_feats
from miscellaneous import make_progress_updater, center_header, center_text, sac_button, set_norm_button, set_cancel_button
import streamlit_antd_components as sac
from miscellaneous import plot_map, IDOSE_FILE, NON_IDOSE_FILE, FEATURE_CODE_FILE
from modify_npis import get_nppes_info_for_npis
from streamlit_pandas_profiling import st_profile_report
from streamlit_folium import st_folium
from ydata_profiling import ProfileReport
from streamlit_extras.dataframe_explorer import dataframe_explorer

def update_info():
    idose_npis = pd.read_csv(IDOSE_FILE)['NPI'].to_list()
    non_idose_npis = pd.read_csv(NON_IDOSE_FILE)['NPI'].to_list()

    train_list = idose_npis + non_idose_npis
    all_codes = list(chain.from_iterable(new_feats.values()))
    cpt_codes = [code for code in all_codes if not code[-1].isalpha()]
    drug_list = [drug for drug in all_codes if drug[-1].isalpha() and drug[0].isalpha()]
    
    return train_list, cpt_codes, drug_list, idose_npis


def load_and_prepare_data():
    st.markdown("""<style>div.stButton > button {display: block;margin: 0 auto;}</style>""", unsafe_allow_html=True)
    center_header('Load and Prepare Data for Model Training', 2)
    col1, col_spacer, col2 = st.columns([15, 0.1, 15])
    
    with col1:
        st.subheader('Auto-generate data directly from CMS')
        start_year = st.selectbox("Starting Year -- including more years will take longer to generate data", list(range(2023, 2012, -1)), key='load_data_startbox')
        
        if st.button('Generate/Regenerate Data', icon=':material/database_upload:', type='primary', width='stretch'):
            st.session_state['idose_col_name'] = IDOS_VAL_COLUMN
            st.session_state['generate_map'] = True
            
            set_cancel_button()
            cancel_button = st.empty()
            if cancel_button.button('Cancel', icon=':material/cancel:', key='cancel_load_data', width='stretch'):
                st.stop()
            
            
            train_list, cpt_codes, drug_list, idose_npis = update_info()
            
            progress_updater, progress_cleaner = make_progress_updater(len(train_list)*3)
            
            df1, cpt_missing = get_code_data_from_cms(train_list, cpt_codes, str(start_year), progress_updater, 0)
            idose_zips = df1[df1['NPI'].isin(idose_npis)][['NPI','ZIP']]
            idose_zips.to_csv('idose_zips.csv')
            cpt_status = st.empty()
            cpt_status.success('Completed fetching CPT code data')
            
            df2, drug_missing = get_drug_data_from_cms(train_list, drug_list, str(start_year), progress_updater, len(train_list))
            drug_status = st.empty()
            drug_status.success('Completed fetching prescription data')
            miss_warn = st.empty() 
            cms_missing = list(dict.fromkeys(cpt_missing+drug_missing)) 
            if len(cpt_missing) > 0: 
                miss_warn.warning(f'NPIS: {cms_missing} were not found in CMS for selected years, and will be ignored')
                            
            df3, nppes_missing = get_nppes_info(train_list, progress_updater, len(train_list)*2)
            nppes_status = st.empty()
            nppes_status.success('Completed fetching NPPES data')
            missings = st.empty()
            if len(nppes_missing) > 0:
                missings.warning(f'NPIS: {nppes_missing} were not found in NPPES and will be ignored')
                        
            progress_cleaner()
            train_list = [npi for npi in train_list if npi not in cms_missing and npi not in nppes_missing]
                
            df_temp = pd.merge(df1, df2, on=['NPI'])
            all_data = pd.merge(df_temp, df3, on=['NPI'])
            all_data = all_data[all_data['NPI'].isin(train_list)]
            
            is_idose = [True if npi in idose_npis else False for npi in all_data['NPI']]
            all_data[IDOS_VAL_COLUMN] = is_idose
            
            cpt_status.empty()
            drug_status.empty()
            nppes_status.empty()
            progress_cleaner()
            
            prep_data_text = st.empty()
            prep_data_text.text('Formatting Data...')
            all_data = format_cms_data(all_data, start_year, MOST_UP_TO_DATE_CMS_YEAR, idose_zips['ZIP'])
            st.session_state.generated_df = all_data.set_index('NPI')
            st.session_state.start_year = start_year
            prep_data_text.empty()
            cancel_button.empty()
            
            # features = [feat for feat in new_feats.keys() if any(feat in col for col in all_data.columns)]
            # for idx, row in all_data.iterrows(): 
            #     for feature in features: 
            #         bene_total = row[f'{feature}_Beneficiaries_in_2021'] + row[f'{feature}_Beneficiaries_in_2022'] + row[f'{feature}_Beneficiaries_in_2023']
            #         if bene_total != row[f'{feature}_Beneficiaries_TOTAL']:
            #             st.error(f'{feature} beneficiaries does not match for {idx}')
            #             st.stop()
            #         serv_total = row[f'{feature}_Services_in_2021'] + row[f'{feature}_Services_in_2022'] + row[f'{feature}_Services_in_2023']
            #         if serv_total != row[f'{feature}_Services_TOTAL']:
            #             st.error(f'{feature} services does not match for {idx}')
            #             st.stop()
            
            set_norm_button()
    
    with col_spacer:
        st.markdown("<div style='height: 600px; border-left: 1px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)
    
    with col2: 
        st.subheader('Upload previously generated data')
        
        new_col1, new_col2 = st.columns(2)
        
        medscout_file = None
        cms_file = None
        medscout = False
        with new_col1:
            st.markdown('##### Medscout Data')
            idose_col_name = st.text_input(label='iDose Value Column', value='Idose (0660T, J7355)')
            medscout_file = st.file_uploader("Upload MedScout Data", type=['csv','xlsx','xls'])
        with new_col2: 
            st.markdown('##### CMS Data')
            start_year = st.selectbox("Starting Year For This Data", list(range(2023, 2012, -1)))
            idose_col_name2 = st.text_input(label='iDose Value Column', value='is_idose')
            cms_file = st.file_uploader("Upload Previously Generated CMS Data", type=['csv','xlsx','xls'])

        
        if medscout_file: 
            data_file = medscout_file 
            medscout = True
            st.session_state['idose_col_name'] = idose_col_name
        elif cms_file: 
            data_file = cms_file
            medscout = False 
            st.session_state['idose_col_name'] = idose_col_name2
        else: 
            data_file = None
        
        if st.button('Load File', icon=':material/attach_file_add:', type='primary', width='stretch'):
            st.session_state['generate_map'] = True
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
                    
                set_cancel_button()
                cancel_button = st.empty()
                if cancel_button.button('Cancel', icon=':material/cancel:', width='stretch', key='cancel-format'):
                    st.stop()
                    
                if medscout:
                    text = st.text('Formatting data, this may take a minute...')
                    upload_updater, upload_cleaner = make_progress_updater(len(df)*2+4)
                    years = [int(MOST_UP_TO_DATE_CMS_YEAR)]
                    for col in df.columns: 
                        match = re.search(r'\b(19|20)\d{2}\b', col)
                        if match: 
                            years.append(int(match.group()))
                    start_year = min(years)
                    idose_zips = pd.read_csv('idose_zips.csv')
                    all_data = format_uploaded_data(df, start_year, MOST_UP_TO_DATE_CMS_YEAR, idose_zips['ZIP'], upload_updater)
                    st.session_state['generated_df'] = all_data.set_index('NPI')
                    st.session_state['start_year'] = start_year
                    upload_cleaner()
                    text.empty()
                else:
                    if idose_col_name2 not in df.columns: 
                        st.warning('iDose value column not found in data')
                    else:
                        st.session_state['generated_df'] = df.set_index('NPI')
                        st.session_state['start_year'] = start_year
            
                load_success.empty()
                cancel_button.empty()
                set_norm_button()
                
            else: 
                if medscout_file and cms_file: 
                    st.error('Load one file at a time')
                else:
                    st.error('Load a file first')
    
    col1, col_spacer, col2 = st.columns([5,0.1,2])
            
    if check_data_loaded(): 
        with col1:
            st.success('Data is loaded and ready!')
            filtered_df = dataframe_explorer(st.session_state.generated_df, case=False)
            st.dataframe(filtered_df, use_container_width=True)
            
        with col_spacer:
            st.markdown("<div style='height: 500px; border-left: 1px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader('You can download the data to avoid future generation (recommended!)')
            st.write('NOTE: Regeneration will need to be run when changes are made to the npi list, or if wanting the most updated CMS versions')
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S") 
            st.download_button('Download Generated Data', st.session_state['generated_df'].to_csv(), f'idose_data_startyear-{st.session_state['start_year']}_{formatted_datetime}.csv', 'text/csv', icon=':material/download:', width='stretch')

        center_header('Data Map', 3)
        
        #pr = ProfileReport(st.session_state.generated_df, title='Data Report')
        #st_profile_report(pr)
        #st.dataframe(st.session_state.npi_info)
        
        with st.spinner('Generating Map...'):
            if 'my_map' not in st.session_state: 
                info_df = get_nppes_info_for_npis(st.session_state.generated_df.index)
            if 'map_npis' not in st.session_state: 
                st.session_state['map_npis'] = info_df['NPI'].to_list()
            if 'map_names' not in st.session_state:
                st.session_state['map_names'] = info_df['Name'].to_list()
            if 'map_zips' not in st.session_state:
                st.session_state['map_zips'] = info_df['Zip'].to_list()
            if 'map_dataset' not in st.session_state:
                st.session_state['map_dataset'] = ['iDose Training Set' if st.session_state.generated_df.loc[npi][st.session_state.idose_col_name] else 'Non-iDose Training Set' for npi in st.session_state['map_npis']]
                
            if st.session_state['generate_map']==True or 'my_map' not in st.session_state: 
                st.session_state['my_map'] = plot_map(st.session_state['map_npis'], st.session_state['map_names'], st.session_state['map_zips'], st.session_state['map_dataset'], show_train=True)
                st.session_state['generate_map'] = False
            
            if st.session_state['my_map'] is not None:  
                st_folium(st.session_state['my_map'], width=1400, height=800, returned_objects=[])         

    
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='load_end')  
            
def check_data_loaded():
    if len(st.session_state.get('generated_df', [])) > 0:
        return True