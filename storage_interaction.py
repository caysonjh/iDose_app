from google.cloud import storage
from google.oauth2 import service_account
from load_data import FEATURE_CODE_FILE
from url_info import code_urls, drug_urls, nppes_url, MOST_UP_TO_DATE_CMS_YEAR
import json
import streamlit as st
import pprint

credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
client = storage.Client(credentials=credentials)

BUCKET_NAME = 'idose_prediction_user_settings'

def load_users():
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'all_users.json')
    
    if not blob.exists(client): 
        blob.upload_from_string(json.dumps({}), content_type='application/json')
        return {}
        
    users = json.loads(blob.download_as_text())
    return users

def write_users(users): 
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'all_users.json')
    
    blob.upload_from_string(json.dumps(users), content_type='application/json')
  

def write_user_environment(): 
    if 'user_environment' not in st.session_state: 
        st.session_state['user_environment'] = {}
        
    user_settings = st.session_state['user_environment']
    user_id = st.session_state['user_id']
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'users/{user_id}.json')
    blob.upload_from_string(json.dumps(user_settings), content_type='application/json')
        
      

def load_user_environment(user_id): 
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'users/{user_id}.json')
    
    if not blob.exists(client): 
        st.session_state['user_environment'] = {}
    else: 
        user_settings = json.loads(blob.download_as_text())
        st.session_state['user_environment'] = user_settings
        if 'code_groupings' in user_settings: 
            new_feats = user_settings['code_groupings']
            loc_update_codes(new_feats)
        
        if 'model_parameters' in user_settings: 
            model_parameters = user_settings['model_parameters']
            loc_update_parameters(model_parameters)
            
        if 'idose_npis' in user_settings: 
            idose_npis = user_settings['idose_npis']
            loc_update_idose_npis(idose_npis)
            
        if 'non_idose_npis' in user_settings: 
            non_idose_npis = user_settings['non_idose_npis']
            loc_update_non_idose_npis(non_idose_npis)
        
        if 'feature_settings' in user_settings: 
            st.session_state['default_settings'] = user_settings['feature_settings']


def write_full_environment(): 
    if 'full_environment' not in st.session_state: 
        st.session_state['full_environment'] = {}

    settings = st.session_state['full_environment']
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'OVERARCHING_CHANGES.json')
    blob.upload_from_string(json.dumps(settings), content_type='application/json')
    

def load_full_environment(): 
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f'OVERARCHING_CHANGES.json')
    
    if blob.exists(client):
        settings = json.loads(blob.download_as_text()) 
        st.session_state['full_environment'] = settings
        new_code_urls = settings['code_urls'] if 'code_urls' in st.session_state else code_urls
        new_drug_urls = settings['drug_urls'] if 'drug_urls' in st.session_state else drug_urls
        new_nppes_url = settings['nppes_url'] if 'nppes_url' in st.session_state else nppes_url
        cms_year = settings['cms_year'] if 'cms_year' in st.session_state else MOST_UP_TO_DATE_CMS_YEAR

        loc_write_urls(new_code_urls, new_drug_urls, new_nppes_url, cms_year)

from load_data import IDOSE_FILE, NON_IDOSE_FILE

def loc_update_idose_npis(new_idose_contents): 
    with open(IDOSE_FILE, 'w') as f: 
        f.write(new_idose_contents)
        
        
def loc_update_non_idose_npis(new_non_idose_contents):
    with open(NON_IDOSE_FILE, 'w') as f: 
        f.write(new_non_idose_contents)    
        
    
def loc_update_codes(new_feats): 
    with open(FEATURE_CODE_FILE, 'w') as f:
        for key, vals in new_feats.items():
            f.write(f"{key}: {', '.join(vals)}\n")
    with open('code_groupings.py', 'w') as f:
        f.write("new_feats = {\n")
        for key, values in new_feats.items():
            values_escaped = [v.replace("'", "\\'") for v in values]
            values_str = ", ".join([f"'{v}'" for v in values_escaped])
            f.write(f"  '{key}': [{values_str}],\n")
        f.write("}\n")
    user_id = st.session_state['user_id']
    USER_FILE = f'user_data/{user_id}.json'
    with open(USER_FILE, 'w') as f: 
        json.dump(new_feats, f, indent=2)
    st.success("Code groups updated and saved successfully!")

    
def loc_update_parameters(new_params): 
    with open('model_parameters.py', 'w') as f: 
        f.write("XGB_PARAMS = ")
        f.write(pprint.pformat(new_params))
    st.success('Successfully updated parameters')
    
    
def loc_write_urls(code_urls, drug_urls, nppes_url, most_current_year): 
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
        