import streamlit as st 
from model_backend import get_drug_data_from_cms, get_nppes_info, format_cms_data, get_code_data_from_cms, format_uploaded_data
from url_info import MOST_UP_TO_DATE_CMS_YEAR
import pandas as pd
from model_backend import IDOS_VAL_COLUMN
import re
from datetime import datetime
from itertools import chain
from code_groupings import new_feats
from miscellaneous import make_progress_updater, center_header, center_text


IDOSE_FILE = 'idose_npis.txt'
NON_IDOSE_FILE = 'non_idose_npis.txt'
FEATURE_CODE_FILE = 'feature_codes.txt'

def update_info():
    idose_npis = []
    non_idose_npis = []
    with open(IDOSE_FILE, 'r') as infile: 
        for line in infile: 
            if not line.strip(): 
                continue
            line = line.strip()
            idose_npis.append(line)
    with open(NON_IDOSE_FILE, 'r') as infile: 
        for line in infile: 
            if not line.strip(): 
                continue
            line = line.strip()
            non_idose_npis.append(line)

    train_list = idose_npis + non_idose_npis
    all_codes = list(chain.from_iterable(new_feats.values()))
    cpt_codes = [code for code in all_codes if not code[-1].isalpha()]
    drug_list = [drug for drug in all_codes if drug[-1].isalpha() and drug[0].isalpha()]
    
    return train_list, cpt_codes, drug_list, idose_npis

def parse_npi_list(text): 
    return set(x.strip() for x in text.strip().splitlines() if x.strip())


def load_and_prepare_data():
    center_header('Load and Prepare Data for Model Training', 2)
    col1, col_spacer, col2 = st.columns([15, 0.1, 15])
    
    with col1:
        st.subheader('Auto-generate data directly from CMS')
        start_year = st.selectbox("Starting Year -- including more years will take longer to generate data", list(range(2023, 2012, -1)))
        if st.button("Generate/Regenerate Data"): 
            st.session_state['idose_col_name'] = IDOS_VAL_COLUMN
            cancel_button = st.empty()
            if cancel_button.button('Cancel'):
                st.stop()
            
            train_list, cpt_codes, drug_list, idose_npis = update_info()
            
            progress_updater, progress_cleaner = make_progress_updater(len(train_list)*3)
            
            df1, cpt_missing = get_code_data_from_cms(train_list, cpt_codes, str(start_year), progress_updater, 0)
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
            all_data = format_cms_data(all_data, start_year, MOST_UP_TO_DATE_CMS_YEAR)
            st.session_state.generated_df = all_data
            st.start_year = start_year
            prep_data_text.empty()
            
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
            
            cancel_button.empty()
    
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
        
        st.markdown("""<style>div.stButton > button {display: block;margin: 0 auto;}</style>""", unsafe_allow_html=True)
        if st.button('Load File'):
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
                    all_data = format_uploaded_data(df, start_year, MOST_UP_TO_DATE_CMS_YEAR, idose_col_name, upload_updater)
                    st.session_state['generated_df'] = all_data
                    upload_cleaner()
                    text.empty()
                else:
                    if idose_col_name2 not in df.columns: 
                        st.warning('iDose value column not found in data')
                    else:
                        st.session_state['generated_df'] = df
            
                load_success.empty()
                
            else: 
                if medscout_file and cms_file: 
                    st.error('Load one file at a time')
                else:
                    st.error('Load a file first')
    
    col1, col_spacer, col2 = st.columns([5,0.1,2])
            
    if check_data_loaded(): 
        with col1:
            st.success('Data is loaded and ready!')
            st.dataframe(st.session_state.generated_df)
            
        with col_spacer:
            st.markdown("<div style='height: 500px; border-left: 1px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader('You can download the data to avoid future generation (recommended!)')
            st.write('*NOTE: Regeneration will need to be run when changes are made to the npi list, or if wanting the most updated CMS versions')
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S") 
            st.download_button('Download Generated Data', st.session_state['generated_df'].to_csv(), f'idose_data_{formatted_datetime}.csv', 'text/csv')
            
            
def check_data_loaded():
    if len(st.session_state.get('generated_df', [])) > 0:
        return True