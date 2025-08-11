import streamlit as st 
import os
from load_data import FEATURE_CODE_FILE

def modify_feature_codes():
    st.header('Edit CPT Codes and their associated buckets')
    
    if 'feature_codes' not in st.session_state: 
        if os.path.exists(FEATURE_CODE_FILE): 
            with open(FEATURE_CODE_FILE, 'r') as f: 
                st.session_state.feature_codes = f.read()
        else: 
            st.session_state.feature_codes = ''
            
    feature_code_contents = st.text_area('Codes included and their varying buckets. Ensure that each bucket name is followed by a \":\" with the codes following it separated by commas, and there are no spaces in the bucket names\n\n' + 
                                         'Example -- COMBO_CAT: 66991,66989', height=700, key='feature_codes') 
    
    if st.button('Save Code Groupings'): 
        feature_code_contents = st.session_state.feature_codes
        with open(FEATURE_CODE_FILE, 'w') as f: 
            f.write(feature_code_contents)
        new_feats = {}
        errors = []
        for i, line in enumerate(feature_code_contents.strip().split('\n'), start=1): 
            if not line.strip():
                i += 1 
                continue 
            if ':' not in line: 
                errors.append(f"Line {i}: Missing ':' - {line.strip()}")
                continue 
            
            key, val = line.split(':', 1)
            key = key.strip()
            val = val.strip() 
            
            if not key: 
                errors.append(f"Line {i}: Missing key/bucket name before ':' -- {line.strip()}")
                continue 
            if not val: 
                errors.append(f"Line {i}: Missing values after ':' -- {line.strip()}")
                
            items = [item.strip() for item in val.split(',') if item.strip()]
            new_feats[key] = items
            
        if errors: 
            st.error("Some lines could not be parsed:")
            for err in errors: 
                st.text(err)
        else:
            st.success('Code groups updated successfully!')
            with open('code_groupings.py', 'w') as f: 
                f.write("new_feats = { \n")
                for key, values in new_feats.items(): 
                    values_escaped = [v.replace("'", "\\'") for v in values]
                    values_str = ", ".join([f"'{v}'" for v in values_escaped])
                    f.write(f"  '{key}': [{values_str}],\n")
                f.write("}\n")