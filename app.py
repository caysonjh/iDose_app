import streamlit as st 
from load_data import load_and_prepare_data
from model_exploration import model_explore
from streamlit_option_menu import option_menu
from modify_urls import modify_urls, write_urls
from modify_params import modify_model_parameters
from modify_npis import modify_npi_info
from modify_codes import modify_feature_codes
from save_models import save_model
from run_predictions import load_prediction
from miscellaneous import center_header, center_text, set_norm_button
import streamlit_antd_components as sac
import json
import os
import time
from storage_interaction import load_users, load_user_environment, write_users, load_full_environment, load_npi_info

st.set_page_config(layout='wide')    

# center_header('iDose Prediction Model', 1)
# st.markdown("<hr style='border: 3px solid black;'>", unsafe_allow_html=True)

MAIN_COLOR = '#4682b4'
ACCENT_COLOR = '#f28c8c'
BACKGROUND = '#DCECFA'
SAGE = '#8cae9c'


st.markdown("""
    <style>
    /* Target all Streamlit download buttons */
    div.stDownloadButton > button {
        border: 2px solid #000000; /* Outline color & thickness */
        color: #FFFFFF;             /* Text color */
        background-color: #f28c8c; /* Background color */
    }

    /* Hover effect */
    div.stDownloadButton > button:hover {
        background-color: #8cae9c; 
        color: white;
        border-color: #000000; 
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Target the file uploader wrapper */
    div.stFileUploader {
        border: 2px solid #000000;  /* Outer border */
        border-radius: 8px;
        padding: 5px;
        background-color: #DCECFA;  /* Background color */
    }

    /* Style the actual file select button */
    div.stFileUploader button {
        background-color: #8cae9c;
        color: white;
        border: 2px solid black;
        border-radius: 5px;
    }

    /* Hover effect for the button */
    div.stFileUploader button:hover {
        background-color: #f28c8c;
        color: white;
        border-color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .my-container {
        background-color: #cfe3d8;
        padding: 20px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

set_norm_button()

users = load_users()


#### MAIN MENU FUNCTION ####

if 'logged_in' not in st.session_state or not st.session_state.get('logged_in', False): 
    center_header('iDose Prediction Login', 2)
    center_text('Account creation ensures that any modifications/customizations you make to the NPI lists, <br>feature sets, code groupings, ' +
                'or model parameters will be saved when you close and revisit the app.<br> Multiple accounts can be created by the same user to enable' + 
                'testing of various settings.<br>New users will be created with the default settings.')
    st.session_state['user_id'] = None
    st.session_state['logged_in'] = False
    
    mode = option_menu(None, ['Login', 'Create User', 'Reset Password', 'Delete User'], 
                    icons=['door-open', 'person-badge', 'key', 'trash'], orientation='horizontal',
                    styles={
                        'container': {'background-color': BACKGROUND},
                        'nav-link-selected': {'background-color': SAGE, 'color':'#FFFFFF'},
                        'nav-link': {'color': MAIN_COLOR}
                    })   
    
    if mode == 'Login':
        user_id = st.text_input('User ID')
        password = st.text_input('Password', type='password')
        if st.button('Login', icon=':material/login:', width='stretch'):
            if user_id in users and users[user_id] == password: 
                st.session_state['user_id'] = user_id
                st.session_state['logged_in'] = True
                st.success(f'Logged in as {user_id}')
                load_user_environment(user_id)
                load_full_environment()
                load_npi_info()
                st.rerun()
            else: 
                st.error('Invalid user ID or password')
        
    if mode == 'Create User':
        user_id = st.text_input('User ID')
        password = st.text_input('Password', type='password')
        if st.button('Create User', icon=':material/person_add:', width='stretch'):
            if user_id in users and users[user_id] == password: 
                st.error(f'User {user_id} already has an account')
            else: 
                st.session_state['user_id'] = user_id
                st.session_state['logged_in'] = True 
                st.session_state['user_environment'] = {}
                st.success(f'Created account for {user_id}')
                users[user_id] = password
                write_users(users)
                load_full_environment()
                load_npi_info()
                st.rerun()
            
    if mode == 'Reset Password': 
        user_id = st.text_input('User ID')
        new_password = st.text_input('New Password', type='password')
        
        if st.button('Reset Password', icon=':material/lock_reset:', width='stretch'):
            if user_id not in users: 
                st.error('User does not exist')
            elif users[user_id] == new_password: 
                st.error('New password cannot be same as old password')
            else: 
                users[user_id] = new_password
                st.success(f'Reset password for {user_id}')
                time.sleep(2)
                write_users(users)
                st.rerun()
    
    if mode == 'Delete User': 
        user_id = st.text_input('User ID')
        password = st.text_input('Password', type='password')
        
        if st.button('Delete User', icon=':material/delete:', width='stretch'): 
            if user_id not in users: 
                st.error('User does not exist')
            elif users[user_id] != password: 
                st.error('Incorrect password for user')
            else: 
                del users[user_id]
                write_users(users)
                st.success(f'Account for {user_id} was deleted')
                time.sleep(2)
                st.rerun()
    
    
else:
    with st.sidebar:
        selection = sac.menu([
            sac.MenuItem('Home', icon='house'),
            sac.MenuItem('Model Training', icon='lightbulb', children=[
                sac.MenuItem('Load Data', icon='cloud-upload'), 
                sac.MenuItem('Model Exploration', icon='rocket-takeoff'),
                sac.MenuItem('Save Model', icon='floppy'),
                sac.MenuItem('Run Prediction', icon='graph-up')
            ]), 
            sac.MenuItem('Settings', icon='gear', children=[
                sac.MenuItem('Modify Feature Codes', icon='prescription'),
                sac.MenuItem('Modify Included NPIs', icon='hospital'),
                sac.MenuItem('Modify Model Parameters', icon='sliders'),
                sac.MenuItem('Modify CMS Urls', icon='router')
            ]),
            sac.MenuItem('Log Out', icon='door-closed'),
        ], open_all=True, color=MAIN_COLOR)


    if selection == 'Home': 
        center_header('Welcome to iDose Prediction 👀 -- BETA', 2)
        center_text("Load data, run and test models, and make predictions on new data. For all information on each page's function, see below:")
        
        sac.divider(label='more important', icon='emoji-grin', align='center', color='gray', key='more_imp') 
        st.markdown('######')
        
        desc = sac.segmented(
            items=[
                sac.SegmentedItem(label='Load Data', icon='cloud-upload'),
                sac.SegmentedItem(label='Model Exploration', icon='rocket-takeoff'),
                sac.SegmentedItem(label='Save Model', icon='floppy'),
                sac.SegmentedItem(label='Run Prediction', icon='graph-up'),
                sac.SegmentedItem(label='Modify Feature Codes', icon='prescription'),
                sac.SegmentedItem(label='Modify Included NPIs', icon='hospital'),
            ], align='center', color=ACCENT_COLOR, bg_color=BACKGROUND
        )
        
        if desc == 'Load Data': 
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Load Data page's function is to upload data used to train models. There are threem, primary ways to prepare data:
                    <ol>
                        <li>Retrieve data from CMS/NPPES databases – This method uses the NPI list shown and modified under the Modify NPIs tab to automatically retrieve the data using the CMS/NPPES API.</li>
                        <li>Upload previously generated data from a different source such as Medscout – This method performs some data formatting steps to get the data in the proper form for the model, specifically retrieving things such as MAC and distance to nearest iDose user.</li>
                        <li>Upload previously retrieved data from CMS or previously formatted data from Medscout – This method requires no pre-processing and is the quickest method.</li>
                    </ol>
                    Data must be uploaded here before any model training can be done anywhere else on the site.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        elif desc == 'Model Exploration': 
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Model Exploration page's function is to experiment with various data/feature combinations. There are two primary 
                    tab selections, each for a different method of model training. 
                    <ol>
                        <li>Run Model and Split by MAC -- This method will spilt the training dataset and train a completely new model for 
                        each subset of physicians for the respective MAC  </li>
                        <li>Run Model with MAC as Feature -- This method will include the entire dataset in the training, and also allows for
                        inclusion of MAC as a feature in model training</li>
                    </ol>
                    Each time a model is trained, metrics/insights will be generated for each model that is trained and a full pdf 
                    report will be available for download. The saved models will also be saved to be used for later training, which can 
                    be seen in the Save Model tab and the Run Prediction tab.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        elif desc == 'Save Model':
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Save Model page's function is to train and save models for future prediction without generating all the 
                    accompanying analyses (that takes up the majority of the runtime). Previously saved models will show up at the 
                    top section of the page, and options to train a new model will be visible below. 
                    </p>
                    <p>
                    Models can be downloaded to be uploaded and used in the future by clicking on them.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        elif desc == 'Run Prediction': 
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Run Prediction page's function is to predict whether new users (*not included in the training set*) are likely
                    to be iDose users. First, a model must be selected to use for prediction. Models can be chosen in 3 ways:
                    <ol>
                        <li>Use Saved Model -- Selection can be made from any of the models that were previously trained and saved simply 
                        by clicking on the ones that show up </li>
                        <li>Upload Model From File -- A previously trained and downloaded model can be uploaded and used to make predictions</li>
                        <li>Train New Model -- A completely new model can be trained and instantly loaded for prediction</li>
                    </ol>
                    Once a model has been selected, there are three ways to generate data for the physicians you want to predict on:
                    <ol>
                        <li>A text box will allow you to enter NPI values for the physicians that you want to predict on, or a list of NPIs can be uploaded via a txt, csv, or xlsx file</li>
                        <li>Previously generated data for the physicians from Medscout can be uploaded -- data formatting will assure that all the data fields are all available</li>
                        <li>Previously generated data for the physicians from CMS or already-formatted Medscout data -- no formatting will be done because it's assumed that the data has already been formatted</li>
                    </ol>
                    You can download the prediction data to be re-uploaded and used again later.
                    </p>
                    <p>
                    Once data has been uploaded, simply hit 'Run Prediction' to make predictions for the data. Those predictions will 
                    be shown and will also be saved to a csv file. Prediction tables will indicate whether the physician was predicted
                    to be an iDose user or not (1 or 0) as well as the confidence associated with that prediction. This should help 
                    to identify prime targets, ones the model finds most likely to fit in the iDose category.                
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        elif desc == 'Modify Feature Codes': 
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Modify Feature Codes page's function is to allow for customization in regards to the codes/prescriptions
                    and the features that values from each code are combined into. New features can be added with associated codes
                    or drugs that you would like to include in the training. 
                    </p>
                    <p>
                    If using auto-generation of data from CMS, any new cpt code or drug can be added and it will be included if there
                    are indeed physicians with records of the code/drug in the CMS database. However, if you are using Medscout or 
                    previously saved data, then only codes/drugs that were included in the download of that data will be available and 
                    this will only function for category combination.  
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
        elif desc == 'Modify Included NPIs': 
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Modify Included NPIs page's function is to allow for editing of the physicians that are used to train the model.
                    This is particularly important as more data will be the best way to improve model performance and insight moving 
                    forward. You can simply add new NPIs to the bottom of the input box depending on if the NPI is an iDose user or not.
                    If there are overlaps between the two datasets, you will be notified when trying to save. If a non-iDose user becomes
                    a dataset it's important to move that user to the iDose column to avoid model confusion.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('#')
        st.markdown('####')
        sac.divider(label='less important', icon='emoji-neutral', align='center', color='gray', key='less_imp')  
        st.markdown('######')
        
        desc2 = sac.segmented(
            items=[
                sac.SegmentedItem(label='Modify Model Parameters', icon='sliders'),
                sac.SegmentedItem(label='Modify CMS Urls', icon='router')  
            ], align='center', color=ACCENT_COLOR, bg_color=BACKGROUND
        )
        
        if desc2 == 'Modify Model Parameters':
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Modify Model Parameters page's function is to allow for customization in regards to the parameters used to train 
                    the XGBoost model. Editing this is not recommended unless you have a basic knowledge of XGBoost parameters and 
                    have good reason for the change. Most modifications won't make a significant difference on performance.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        elif desc2 == 'Modify CMS Urls': 
            st.markdown("""
                <div class="my-container">
                    <p>
                    The Modify CMS Urls page's function is to allow for updating of the CMS urls in order to connect with CMS api 
                    databases as the CMS database is updated with new years. This shouldn't happen often, and the existing urls shouldn't
                    need to be updated unless the CMS api changes for some reason.
                    </p>
                    <p>
                    Adding new urls when CMS updates their database with new years allows you to train a model with the most recent data.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    elif selection == 'Load Data': 
        load_and_prepare_data() 

    elif selection == 'Model Exploration':
        model_explore()
        
    elif selection == 'Save Model':
        save_model()
        
    elif selection == 'Run Prediction': 
        load_prediction() 

    elif selection == 'Modify Feature Codes': 
        modify_feature_codes()
        
    elif selection == 'Modify Included NPIs': 
        modify_npi_info()
        
    elif selection == 'Modify Model Parameters':
        modify_model_parameters()

    elif selection == 'Modify CMS Urls':
        modify_urls() 

    elif selection == 'Log Out': 
        st.session_state['logged_in'] = False 
        st.session_state['user_id'] = None
        st.rerun()