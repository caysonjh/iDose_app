import time
import streamlit as st

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