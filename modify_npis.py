import streamlit as st 
import os 
from load_data import IDOSE_FILE, NON_IDOSE_FILE
import streamlit_antd_components as sac
from storage_interaction import write_user_environment

def parse_npi_list(text): 
    return set(x.strip() for x in text.strip().splitlines() if x.strip())
    

def update_idose_npis(new_idose_contents): 
    with open(IDOSE_FILE, 'w') as f: 
        f.write(new_idose_contents)
        
    st.session_state['idose_npis'] = new_idose_contents
    write_user_environment()
        
def update_non_idose_npis(new_non_idose_contents):
    with open(NON_IDOSE_FILE, 'w') as f: 
        f.write(new_non_idose_contents)    
        
    st.session_state['non_idose_npis'] = new_non_idose_contents
    write_user_environment()

    
def modify_npi_info():
    st.header('Edit included NPIs for iDose and non-iDose users')
    st.text('Ensure that all included NPIs are Type-1/Individual NPIs')
    
    if 'idose_contents' not in st.session_state:
        if os.path.exists(IDOSE_FILE): 
            with open(IDOSE_FILE, 'r') as f: 
                st.session_state.idose_contents = f.read()  
        else:
            st.session_state.idose_contents = ''   
    
    if 'non_idose_contents' not in st.session_state:
        if os.path.exists(NON_IDOSE_FILE): 
            with open(NON_IDOSE_FILE, 'r') as f: 
                st.session_state.non_idose_contents = f.read()
        else: 
            st.session_state.non_idose_contents = ''
    
    col1, col2 = st.columns(2)

    with col1:   
        st.subheader('Edit iDose Users')
        new_idose_contents = st.text_area("iDose Users (one NPI per line):", value=st.session_state.idose_contents, height=700, key='idose_text')

   # st.markdown("---")

    with col2:
        st.subheader('Edit Non-iDose Users')
        new_non_idose_contents = st.text_area("Non iDose Users (one NPI per line):", value=st.session_state.non_idose_contents, height=700, key='non_idose_text') 
    
    if st.button('Save Lists', icon=':material/patient_list:', width='stretch'): 
        new_idose_contents = st.session_state.idose_text
        new_non_idose_contents = st.session_state.non_idose_text
        
        idose_list = parse_npi_list(new_idose_contents)
        non_idose_list = parse_npi_list(new_non_idose_contents)
        overlap = idose_list.intersection(non_idose_list)
        
        errors = []
        for npi in idose_list: 
            if len(npi) != 10: 
                errors.append(f'Error: {npi} must be 10 digits')
                st.stop()
        for npi in non_idose_list: 
            if len(npi) != 10: 
                errors.append(f'Error: {npi} must be 10 digits')
                st.stop()
        
        if overlap:
            st.error(f"Conflict: NPIs {', '.join(overlap)} appear in both lists")
        else: 
            update_idose_npis(new_idose_contents)
            update_non_idose_npis(new_non_idose_contents)
            st.success('NPIs updated successfully')
            
            st.session_state.idose_contents = new_idose_contents
            st.session_state.non_idose_contents = new_non_idose_contents
            
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='npis_end')