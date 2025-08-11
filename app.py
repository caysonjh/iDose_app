import streamlit as st 
from load_data import load_and_prepare_data
from model_exploration import model_explore
from streamlit_option_menu import option_menu
from modify_urls import modify_urls
from modify_params import modify_model_parameters
from modify_npis import modify_npi_info
from modify_codes import modify_feature_codes
from save_models import save_model
from run_predictions import run_prediction
from miscellaneous import center_header, center_text

st.set_page_config(layout='wide')    



center_header('iDose Prediction Model', 1)
st.markdown("<hr style='border: 3px solid black;'>", unsafe_allow_html=True)

#### MAIN MENU FUNCTION ####

with st.sidebar: 
    menu = option_menu('Main Menu', ['Load Data', 'Model Exploration', 'Save Model', 'Run Prediction', 'Modify Feature Codes', 'Modify Included NPIs', 'Modify Model Parameters', 'Modify CMS Urls'],
                       icons=['cloud-upload', 'rocket-takeoff', 'floppy', 'graph-up', 'prescription', 'hospital', 'sliders', 'router'])

if menu == 'Load Data':
    load_and_prepare_data()
    
if menu == 'Model Exploration': 
    model_explore()
             
elif menu == 'Save Model':
    save_model()
    
elif menu == 'Run Prediction': 
    run_prediction()
 
elif menu == 'Modify Feature Codes': 
    modify_feature_codes()

elif menu == 'Modify Included NPIs': 
    modify_npi_info()
        
elif menu == 'Modify Model Parameters': 
    modify_model_parameters()

elif menu == 'Modify CMS Urls': 
    modify_urls()