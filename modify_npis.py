import streamlit as st 
import os 
from load_data import IDOSE_FILE, NON_IDOSE_FILE
import streamlit_antd_components as sac
from storage_interaction import write_user_environment
import pandas as pd
from url_info import nppes_url
import requests

def parse_npi_list(text): 
    return set(x.strip() for x in text.strip().splitlines() if x.strip())
    
def update_idose_npis(new_idose_contents): 
    new_idose_contents.to_csv(IDOSE_FILE)
        
    st.session_state['idose_contents'] = new_idose_contents
    st.session_state['user_environment']['idose_npis'] = new_idose_contents.to_dict(orient='records')
    write_user_environment()
        
def update_non_idose_npis(new_non_idose_contents):
    new_non_idose_contents.to_csv(NON_IDOSE_FILE)  
        
    st.session_state['non_idose_contents'] = new_non_idose_contents
    st.session_state['user_environment']['non_idose_npis'] = new_non_idose_contents.to_dict(orient='records')
    write_user_environment()

def get_nppes_info_for_npis(npi_list): 
    missing_npis = []
    state_to_mac = pd.read_csv('state_to_mac.csv').set_index('State')['MAC'].to_dict()
    df = pd.DataFrame(columns=['NPI', 'Name', 'City', 'State', 'Zip', 'MAC'])
    for npi in npi_list:
        #st.text(f'Getting nppes for {npi}')
        get_url = nppes_url + f'&number={npi}'
        response = requests.get(get_url)
        response.raise_for_status()
        json_data = response.json()
        
        if not json_data.get('results', []): 
            missing_npis.append(npi)
            continue 
        
        first_name = json_data['results'][0]['basic']['first_name']
        last_name = json_data['results'][0]['basic']['last_name']
        name = first_name + ' ' + last_name
        city = json_data['results'][0]['addresses'][0]['city']
        state = json_data['results'][0]['addresses'][0]['state']
        zip = json_data['results'][0]['addresses'][0]['postal_code'][:5]
        mac = state_to_mac[state]
        
        df = pd.concat([df, pd.DataFrame({
            'NPI':[npi],
            'Name':[name],
            'City':[city],
            'State':[state],
            'Zip':[zip],
            'MAC':[mac]
        })])
    
    return df 

def add_npi_to_table(npi, contents_label):
    if len(npi) != 10: 
        st.error('NPI must be 10 characters long')
    elif int(npi) in st.session_state['idose_contents']['NPI'].to_list(): 
        st.error('NPI already in iDose dataset, if it was just deleted, save lists and try again')
    elif int(npi) in st.session_state['non_idose_contents']['NPI'].to_list():
        st.error('NPI already in Non iDose dataset, if it was just deleted, save lists and try again')
    else:   
        new_df = get_nppes_info_for_npis([npi])
        st.session_state[contents_label] = pd.concat([new_df, st.session_state[contents_label]])
        st.session_state['last_npi'] = npi
        st.rerun()

    
def modify_npi_info():
    st.header('Edit included NPIs for iDose and non-iDose users')
    st.markdown('#### Columns can be deleted or added -- Ensure that all included NPIs are Type-1/Individual NPIs')
    st.markdown('##### **NOTE**: Changes will not go into effect until <Save Lists> is pressed')
    sac.divider(label='add/delete npis', icon='person-vcard', align='center', color='gray', key='npis_insert')
    
    if 'last_npi' not in st.session_state: 
        st.session_state.last_npi = ''
    
    if 'idose_contents' not in st.session_state:
        if os.path.exists(IDOSE_FILE):
            st.session_state.idose_contents = pd.read_csv(IDOSE_FILE, index_col=0, dtype={'Zip':str}).reset_index(drop=True)
        else:
            st.session_state.idose_contents = ''   
    
    if 'non_idose_contents' not in st.session_state:
        if os.path.exists(NON_IDOSE_FILE): 
            st.session_state.non_idose_contents = pd.read_csv(NON_IDOSE_FILE, index_col=0, dtype={'Zip':str}).reset_index(drop=True)
        else: 
            st.session_state.non_idose_contents = ''
    
    col1, col_spacer, col2 = st.columns([5,0.1,5])

    with col1:   
        idose_cols = st.columns(2)
        with idose_cols[0]:
            st.subheader('Edit iDose:')  
        with idose_cols[1]:
            idose_npi = st.text_input('', label_visibility='collapsed', placeholder='New iDose NPI...', icon=':material/cardiology:')
            if idose_npi and idose_npi != st.session_state['last_npi']: 
                add_npi_to_table(idose_npi, 'idose_contents')          
                
        #new_idose_contents = st.text_area("iDose Users (one NPI per line):", value=st.session_state.idose_contents, height=700, key='idose_text')
        st.session_state['idose_contents']['NPI'] = st.session_state['idose_contents']['NPI'].astype(int)
        new_idose_contents = st.data_editor(st.session_state['idose_contents'].reset_index(drop=True), num_rows='dynamic', hide_index=True, 
                                            disabled=('NPI','Name','City','State','Zip','MAC'), height=500, key='idose')

   # st.markdown("---")
    with col_spacer:
        st.markdown("<div style='height: 570px; border-left: 2px solid lightgray; margin: 0 auto;'></div>", unsafe_allow_html=True)

    with col2:
        non_idose_cols = st.columns(2)
        with non_idose_cols[0]:
            st.subheader('Edit Non iDose:')
        with non_idose_cols[1]:
            non_idose_npi = st.text_input('', label_visibility='collapsed', placeholder='New Non iDose NPI...', icon=':material/pulse_alert:')
            if non_idose_npi and non_idose_npi != st.session_state['last_npi']: 
                add_npi_to_table(non_idose_npi, 'non_idose_contents')
                
        #new_non_idose_contents = st.text_area("Non iDose Users (one NPI per line):", value=st.session_state.non_idose_contents, height=700, key='non_idose_text') 
        st.session_state['non_idose_contents']['NPI'] = st.session_state['non_idose_contents']['NPI'].astype(int)
        new_non_idose_contents = st.data_editor(st.session_state['non_idose_contents'].reset_index(drop=True), num_rows='dynamic', hide_index=True,
                                                disabled=('NPI','Name','City','State','Zip','MAC'), height=500, key='non')
    
    if st.button('Save Lists', icon=':material/patient_list:', width='stretch'): 
        # new_idose_contents = st.session_state.idose_text
        # new_non_idose_contents = st.session_state.non_idose_text
        
        # idose_list = parse_npi_list(new_idose_contents)
        # non_idose_list = parse_npi_list(new_non_idose_contents)
        # overlap = idose_list.intersection(non_idose_list)
        
        idose_list = set(new_idose_contents['NPI'])
        non_idose_list = set(new_non_idose_contents['NPI'])
        overlap = idose_list.intersection(non_idose_list)
        
        errors = []
        for npi in idose_list: 
            if len(str(npi)) != 10: 
                errors.append(f'Error: {npi} must be 10 digits')
                st.stop()
        for npi in non_idose_list: 
            if len(str(npi)) != 10: 
                errors.append(f'Error: {npi} must be 10 digits')
                st.stop()
        
        if overlap:
            st.error(f"Conflict: NPIs {', '.join(overlap)} appear in both lists")
        else: 
            # new_idose = [npi for npi in idose_list if npi not in st.session_state.idose_contents['NPI']]
            # new_non = [npi for npi in non_idose_list if npi not in st.session_state.non_idose_contents['NPI']]
            # if len(new_idose) > 0: 
            #     with st.spinner('Getting nppes info for new iDose npis...'):
            #         new_idose_contents = pd.concat([new_idose_contents, new_idose])
            # if len(new_non) > 0:
            #     with st.spinner('Getting nppes info for new non-iDose npis...'):
            #         new_non_idose_contents = pd.concat([new_non_idose_contents, new_non])
            
            update_idose_npis(new_idose_contents)
            update_non_idose_npis(new_non_idose_contents)
            st.success('NPIs updated successfully')
            
            
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='npis_end')