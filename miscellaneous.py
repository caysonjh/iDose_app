import time
import streamlit as st
import streamlit_antd_components as sac
import folium 
from streamlit_folium import st_folium
import pgeocode
import pandas as pd
import numpy as np

IDOSE_FILE = 'idose_npis.csv'
NON_IDOSE_FILE = 'non_idose_npis.csv'
FEATURE_CODE_FILE = 'feature_codes.txt'

def center_header(text, level=1):
    return st.markdown(f"<h{level} style='text-align: center;'>{text}</h{level}>", unsafe_allow_html=True)

def center_text(text: str):
    st.markdown(f"<p style='text-align: center;'>{text}</p>", unsafe_allow_html=True)

def make_progress_updater(total):
    progress_bar = st.empty()
    progress_bar = progress_bar.progress(0)
    task_text = st.empty()
    time_remaining_text = st.empty()
    
    start_time = time.time()

    def update_progress(current, task_desc): 
        elapsed = time.time() - start_time 
        progress = current/total
        percent = current/total*100
        
        est_total_time = elapsed / progress if progress > 0 else 0
        est_remaining = est_total_time - elapsed 
        
        mins, secs = divmod(int(est_remaining), 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        
        progress_bar.progress(progress)
        task_text.text(f"{task_desc} ({current} / {total}) -- {round(percent, 2)}% Complete")
        time_remaining_text.text(f"Estimated time remaining: {time_str}")
    
    def clear_progress(): 
        progress_bar.empty()
        task_text.empty()
        time_remaining_text.empty()
        
    return update_progress, clear_progress

def sac_button(value, icon=None, color=None): 
    clicked = sac.buttons([
        sac.ButtonsItem(label=value, icon=icon, color=color)
    ], align='center')
    
    return clicked


def set_norm_button():
    st.markdown("""
        <style>
        /* Target all Streamlit buttons */
        div.stButton > button:first-child {
            border: 2px solid #000000; /* Outline color & thickness */
            color: #FFFFFF;             /* Text color */
            background-color: #7c8459; /* Make it look outlined */
        }

        /* Change hover effect */
        div.stButton > button:first-child:hover {
            background-color: #097175; 
            color: white;
            border-color: #000000;  /* Slightly darker outline on hover */
        }
        </style>
    """, unsafe_allow_html=True)

    
def set_cancel_button():
    st.markdown("""
        <style>
        /* Target all Streamlit buttons */
        div.stButton > button:first-child {
            border: 2px solid #000000; /* Outline color & thickness */
            color: #FFFFFF;             /* Text color */
            background-color: #097175; /* Make it look outlined */
        }

        /* Change hover effect */
        div.stButton > button:first-child:hover {
            background-color: #c1941f; 
            color: white;
            border-color: #000000;  /* Slightly darker outline on hover */
        }
        </style>
    """, unsafe_allow_html=True)
    
def add_npis(m, npis, names, zips, dataset): 
    color_map = {'iDose Training Set':'green', 'Non-iDose Training Set':'red', 'iDose Prediction':'blue', 'Non-iDose Prediction':'orange'}
    groups = st.session_state['groups']
    
    nomi = pgeocode.Nominatim('us')
    locations = nomi.query_postal_code(zips)
        
    for i, loc in locations.iterrows():
        if not np.isnan(loc.latitude) and not np.isnan(loc.longitude): 
            folium.Marker(
                location=[loc.latitude, loc.longitude],
                popup=f'NPI:{npis[i]}, NAME:{names[i]}',
                tooltip=f'NPI:{npis[i]}, NAME:{names[i]}',
                icon=folium.Icon(prefix='fa', color=color_map[dataset[i]], icon='user-doctor')
            ).add_to(groups[dataset[i]])

    
    return m      
    
    
def plot_map(npis, names, zips, dataset, show_train=False): 
    color_map = {'iDose Training Set':'green', 'Non-iDose Training Set':'red', 'iDose Prediction':'blue', 'Non-iDose Prediction':'orange'}
    groups = {
        'iDose Training Set':folium.FeatureGroup(name='iDose Training Set', overlay=True, show=show_train),
        'Non-iDose Training Set':folium.FeatureGroup(name='Non-iDose Training Set', overlay=True, show=show_train), 
        'iDose Prediction':folium.FeatureGroup(name='iDose Prediction', overlay=True), 
        'Non-iDose Prediction':folium.FeatureGroup(name='Non-iDose Prediction', overlay=True)
    }
    
    nomi = pgeocode.Nominatim('us')
    locations = nomi.query_postal_code(zips)
    
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    for i, loc in locations.iterrows():
        if not np.isnan(loc.latitude) and not np.isnan(loc.longitude): 
            folium.Marker(
                location=[loc.latitude, loc.longitude],
                popup=f'NPI:{npis[i]}, NAME:{names[i]}, DATASET:{dataset[i]}',
                tooltip=f'NPI:{npis[i]}, NAME:{names[i]}, DATASET:{dataset[i]}',
                icon=folium.Icon(prefix='fa', color=color_map[dataset[i]], icon='user-doctor')
            ).add_to(groups[dataset[i]])
            
    for group in groups.values(): 
        group.add_to(m)

    css_fix = """
        <style>
        .leaflet-control-layers-expanded {
            max-height: 150px !important;
            overflow-y: auto !important;
        }
        </style>
    """
    m.get_root().html.add_child(folium.Element(css_fix))    

    folium.LayerControl(collapsed=True).add_to(m)
    
    st.session_state['groups'] = groups
            
    legend_html = '''
        <div style="
        position: fixed;
        bottom: 50px; left: 50px; width: 220px; height: 180px;
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:whtie; padding: 10px; 
        ">
        <b>Dataset Key</b><br>
        <i class="fa fa-map-marker fa-2x" style="color:green"></i> iDose Training Set<br>
        <i class="fa fa-map-marker fa-2x" style="color:red"></i> Non-iDose Training Set<br>
        <i class="fa fa-map-marker fa-2x" style="color:blue"></i> iDose Prediction<br>
        <i class="fa fa-map-marker fa-2x" style="color:orange"></i> Non-iDose Prediction
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
            
    return m 