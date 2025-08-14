import pprint
import streamlit as st
from url_info import code_urls, drug_urls, nppes_url
import streamlit_antd_components as sac
from storage_interaction import write_full_environment

def modify_urls():
    st.header('Modify the URLs associated with the CMS API, or add new years when they become available')
    st.markdown('CPT Code URLs can be found [here](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service)')
    st.markdown('Drug URLs can be found [here](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug)')
    st.markdown('NPPES database URL can be found [here](https://npiregistry.cms.hhs.gov/api-page)')
        
    st.markdown('### Most Recent CMS Year')
    prev_most_current = max([int(code) for code in code_urls.keys()])
    most_current_year = st.text_input(label='Latest year in the CMS database', value=prev_most_current)
    
    if st.button('Update Year', icon=':material/calendar_month:'):
        if int(most_current_year) > int(prev_most_current):
            st.success('Please enter URLs for the new years')
            for i in range(int(prev_most_current)+1, int(most_current_year)+1): 
                code_urls[str(i)] = st.text_input(label=f'CPT code URL for {i}', value='')
                drug_urls[str(i)] = st.text_input(label=f'Drug URL for {i}', value='')
            
            if all([code_urls[str(i)] for i in range(int(prev_most_current)+1, int(most_current_year)+1)]) and all([drug_urls[str(i)] for i in range(int(prev_most_current)+1, int(most_current_year)+1)]): 
                if st.button('Submit New URLs'):
                    write_urls(code_urls, drug_urls, nppes_url, most_current_year)
                        
        if int(most_current_year) < int(prev_most_current): 
            for i in range(int(prev_most_current), int(most_current_year), -1): 
                code_urls.pop(str(i), None)
                drug_urls.pop(str(i), None)
            write_urls(code_urls, drug_urls, nppes_url, most_current_year)
            
    col1, col2 = st.columns(2)
            
    with col1:
        st.markdown('### NPPES API Database URL')
        new_nppes_url = st.text_input(label='NPPES API URL', value=nppes_url)
        st.markdown('### CPT Code Database URLs')  
        new_code_urls = {}
        for year, url in code_urls.items(): 
            new_code_urls[year] = st.text_input(label=year,value=url,key=f'code{year}')
            
    with col2:
        st.markdown('#')
        if st.button('Update URL Info', icon=':material/captive_portal:', width='stretch'): 
            write_urls(new_code_urls, new_drug_urls, new_nppes_url, most_current_year)
        st.markdown('### Drug Database URLs')
        new_drug_urls = {}
        for year, url in drug_urls.items(): 
            new_drug_urls[year] = st.text_input(label=year,value=url,key=f'drug{year}')
            

        
        
    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='load_end')  



def write_urls(code_urls, drug_urls, nppes_url, most_current_year): 
    with open('url_info.py', 'w') as f: 
        f.write('code_urls = ')
        f.write(pprint.pformat(code_urls))
        f.write('\n')
        f.write('drug_urls = ')
        f.write(pprint.pformat(drug_urls))
        f.write('\n')
        f.write('nppes_url = ')
        f.write(pprint.pformat(nppes_url))
        f.write('\n')
        f.write('MOST_UP_TO_DATE_CMS_YEAR = ')
        f.write(str(most_current_year))
        
    st.session_state['code_urls'] = code_urls
    st.session_state['drug_urls'] = drug_urls
    st.session_state['nppes_url'] = nppes_url
    st.session_state['cms_year'] = most_current_year
    write_full_environment()