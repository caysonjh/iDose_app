import streamlit as st 
import os
from miscellaneous import FEATURE_CODE_FILE
import streamlit_antd_components as sac
import json
from storage_interaction import write_user_environment

# def modify_feature_codes():
#     st.header('Edit CPT Codes and their associated buckets')
    
#     if 'feature_codes' not in st.session_state: 
#         if os.path.exists(FEATURE_CODE_FILE): 
#             with open(FEATURE_CODE_FILE, 'r') as f: 
#                 st.session_state.feature_codes = f.read()
#         else: 
#             st.session_state.feature_codes = ''
            
#     feature_code_contents = st.text_area('Codes included and their varying buckets. Ensure that each bucket name is followed by a \":\" with the codes following it separated by commas, and there are no spaces in the bucket names\n\n' + 
#                                          'Example -- COMBO_CAT: 66991,66989', height=700, key='feature_codes') 
    
#     if st.button('Save Code Groupings'): 
#         feature_code_contents = st.session_state.feature_codes
#         with open(FEATURE_CODE_FILE, 'w') as f: 
#             f.write(feature_code_contents)
#         new_feats = {}
#         errors = []
#         for i, line in enumerate(feature_code_contents.strip().split('\n'), start=1): 
#             if not line.strip():
#                 i += 1 
#                 continue 
#             if ':' not in line: 
#                 errors.append(f"Line {i}: Missing ':' - {line.strip()}")
#                 continue 
            
#             key, val = line.split(':', 1)
#             key = key.strip()
#             val = val.strip() 
            
#             if not key: 
#                 errors.append(f"Line {i}: Missing key/bucket name before ':' -- {line.strip()}")
#                 continue 
#             if not val: 
#                 errors.append(f"Line {i}: Missing values after ':' -- {line.strip()}")
                
#             items = [item.strip() for item in val.split(',') if item.strip()]
#             new_feats[key] = items
            
#         if errors: 
#             st.error("Some lines could not be parsed:")
#             for err in errors: 
#                 st.text(err)
#         else:
#             st.success('Code groups updated successfully!')
#             with open('code_groupings.py', 'w') as f: 
#                 f.write("new_feats = { \n")
#                 for key, values in new_feats.items(): 
#                     values_escaped = [v.replace("'", "\\'") for v in values]
#                     values_str = ", ".join([f"'{v}'" for v in values_escaped])
#                     f.write(f"  '{key}': [{values_str}],\n")
#                 f.write("}\n")

def update_codes(new_feats): 
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
    
    st.session_state['user_environment']['code_groupings'] = new_feats 
    write_user_environment()


def modify_feature_codes():
    st.header('Edit codes/drugs and their associated buckets')

    # Initialize storage of pairs if not in session state
    if 'feature_pairs' not in st.session_state:
        if os.path.exists(FEATURE_CODE_FILE):
            with open(FEATURE_CODE_FILE, 'r') as f:
                lines = f.read().strip().split('\n')
            pairs = []
            for line in lines:
                if ':' in line:
                    key, vals = line.split(':', 1)
                    key = key.strip()
                    vals = vals.strip()
                    pairs.append({'key': key, 'values': vals})
            st.session_state.feature_pairs = pairs
        else:
            st.session_state.feature_pairs = []

    st.markdown("### Current Buckets and Codes (Editable)")

    if not st.session_state.feature_pairs:
        st.info("No key-value pairs added yet.")
    else:
        # We'll display editable inputs for each pair in a form to batch save edits
        with st.form("edit_pairs_form"):
            remove_index = None
            for i, pair in enumerate(st.session_state.feature_pairs):
                cols = st.columns([3, 6, 1])
                with cols[0]:
                    new_key = st.text_input(f"Key {i+1}", value=pair['key'], key=f"key_{i}")
                with cols[1]:
                    new_values = st.text_input(f"Values {i+1} (comma-separated)", value=pair['values'], key=f"values_{i}")
                with cols[2]:
                    if st.form_submit_button(f"Remove {i+1}"):
                        remove_index = i

                # Update the pair in session_state immediately after input (optional)
                pair['key'] = new_key.strip()
                pair['values'] = new_values.strip()

            if remove_index is not None:
                st.session_state.feature_pairs.pop(remove_index)
                st.rerun()

            submitted_edit = st.form_submit_button("Save edits")
            if submitted_edit:
                # Validate edits
                errors = []
                for i, pair in enumerate(st.session_state.feature_pairs, start=1):
                    if not pair['key']:
                        errors.append(f"Line {i}: Bucket name (key) cannot be empty.")
                    if not pair['values']:
                        errors.append(f"Line {i}: Codes (values) cannot be empty.")

                if errors:
                    st.error("Please fix the following errors before saving:")
                    for err in errors:
                        st.text(err)
                else:
                    st.success("Edits saved!")
            
    sac.divider(label='add new code groupings', icon='eye', align='center', color='gray', key='split_end')
    
    st.subheader('Add new bucket-code pairings')
    # Input fields for new key-value pair
    # Add new bucket form
    with st.form("add_pair_form"):
        new_key = st.text_input("Bucket name (key)")
        new_values = st.text_input("Codes (comma-separated)")
        submitted = st.form_submit_button("Add new key-value pair")
        if submitted:
            if not new_key.strip():
                st.warning("Bucket name (key) cannot be empty.")
            elif not new_values.strip():
                st.warning("Codes (values) cannot be empty.")
            else:
                st.session_state.feature_pairs.append({'key': new_key.strip(), 'values': new_values.strip()})
                st.success(f"Added bucket '{new_key.strip()}'.")
                # No need to manually clear inputs — closing and reopening the form resets the inputs
               
    sac.divider(label='save changes', icon='floppy', align='center', color='gray', key='save_codes') 
    # Save button
    if st.button("Save Code Groupings", width='stretch', icon=':material/move_group:'):
        # Validate and parse all pairs
        errors = []
        new_feats = {}
        for i, pair in enumerate(st.session_state.feature_pairs, start=1):
            key = pair['key']
            val = pair['values']
            if not key:
                errors.append(f"Line {i}: Missing bucket name (key).")
                continue
            if not val:
                errors.append(f"Line {i}: Missing codes (values).")
                continue
            items = [item.strip() for item in val.split(',') if item.strip()]
            new_feats[key] = items

        if errors:
            st.error("Some lines could not be parsed:")
            for err in errors:
                st.text(err)
        else:
            update_codes(new_feats)


    sac.divider(label='end', icon='sign-dead-end', align='center', color='gray', key='codes_end')

