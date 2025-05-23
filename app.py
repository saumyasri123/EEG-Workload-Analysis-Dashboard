import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time
import base64
from pathlib import Path

# Import custom modules
from eeg_processor import EEGProcessor
from visualizations import (
    plot_raw_eeg, plot_band_power_heatmap, plot_fft_spectrum,
    plot_subject_ratings
)
from brain_visualization import (
    create_3d_brain_model, add_electrodes_to_brain, create_activity_surface,
    animate_brain_activity, plot_brain_connectivity, plot_3d_spectrogram,
    create_band_topography, create_3d_time_series
)

#import streamlit as st

# Page configuration for browser tab
st.set_page_config(
    page_title="STEW EEG Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Centered title using markdown and HTML inside a wide container
# with st.container():
#     st.markdown(
#         "<h1 style='text-align: center;'>🧠 STEW EEG Analysis Dashboard</h1>",
#         unsafe_allow_html=True
#     )


# Apply custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0E1117;
    }
    
    /* Card styling for sections */
    .stCard {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #1A1D28;
        border-left: 5px solid #4257B2;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-weight: 600;
        color: #FFFFFF;
    }
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #4257B2, #8B44F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 5px 0;
    }
    h2 {
        font-size: 1.8rem;
        color: #4257B2;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    h3 {
        font-size: 1.3rem;
        color: #8B44F6;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4257B2;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #8B44F6;
        transform: scale(1.05);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #4257B2;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1A1D28;
        border-right: 1px solid #2E3341;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        margin-bottom: 10px;
    }
    section[data-testid="stSidebar"] .stRadio > div {
        display: flex;
        flex-direction: column;
    }
    
    /* Highlight for interactive elements */
    .highlight {
        padding: 8px;
        background-color: rgba(66, 87, 178, 0.1);
        border-radius: 5px;
    }
    
    /* Animation for loading spinner */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SAMPLING_RATE = 128  # Hz
WINDOW_SIZE = 512    # 4 seconds of EEG data
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30)
}

# Initialize session state variables
if 'processor' not in st.session_state:
    st.session_state.processor = EEGProcessor(
        sampling_rate=SAMPLING_RATE,
        window_size=WINDOW_SIZE
    )
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'extracted_features' not in st.session_state:
    st.session_state.extracted_features = None
if 'selected_subject_data' not in st.session_state:
    st.session_state.selected_subject_data = None
if 'selected_3d_data' not in st.session_state:
    st.session_state.selected_3d_data = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0.0
if 'show_loader' not in st.session_state:
    st.session_state.show_loader = False

# Helper functions
def update_progress(increment=0.1):
    st.session_state.progress = min(1.0, st.session_state.progress + increment)
    
def reset_progress():
    st.session_state.progress = 0.0
    
def show_loader():
    st.session_state.show_loader = True
    
def hide_loader():
    st.session_state.show_loader = False

def display_brain_icon():
    # Function to display a decorative brain icon
    brain_html = """
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <div style="text-align: center; animation: float 6s ease-in-out infinite;">
            <svg width="100" height="100" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M32 5C18.7 5 8 15.7 8 29C8 42.3 18.7 53 32 53C45.3 53 56 42.3 56 29C56 15.7 45.3 5 32 5Z" 
                    stroke="#4257B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M32 5C25 5 20 12 20 20C20 28 25 36 32 36C39 36 44 28 44 20C44 12 39 5 32 5Z" 
                    stroke="#8B44F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20 29C20 36 25 42 32 42C39 42 44 36 44 29" 
                    stroke="#8B44F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M32 36V53" stroke="#4257B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M20 29H44" stroke="#4257B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <style>
                @keyframes float {
                    0% { transform: translateY(0px); }
                    50% { transform: translateY(-10px); }
                    100% { transform: translateY(0px); }
                }
            </style>
        </div>
    </div>
    """
    st.markdown(brain_html, unsafe_allow_html=True)

def load_animation(animation_id="pulse"):
    """Display a loading animation"""
    if animation_id == "pulse":
        animation_html = """
        <div class="loading-container" style="display: flex; justify-content: center; margin: 20px 0;">
            <div class="loading-pulse" style="text-align: center;">
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: #4257B2; animation: pulse 1.5s infinite ease-in-out;"></div>
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: #6A73E3; animation: pulse 1.5s infinite ease-in-out 0.2s;"></div>
                    <div style="width: 20px; height: 20px; border-radius: 50%; background: #8B44F6; animation: pulse 1.5s infinite ease-in-out 0.4s;"></div>
                </div>
                <p style="margin-top: 10px; color: #FFFFFF;">Processing data...</p>
            </div>
        </div>
        """
    else:
        animation_html = """
        <div class="loading-container" style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="text-align: center;">
                <div class="spinner" style="width: 40px; height: 40px; border: 4px solid rgba(66, 87, 178, 0.1); border-left-color: #4257B2; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                <p style="margin-top: 10px; color: #FFFFFF;">Loading...</p>
            </div>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """
    st.markdown(animation_html, unsafe_allow_html=True)

def create_custom_card(title, content, icon=None):
    """Create a custom styled card with optional icon"""
    card_html = f"""
    <div class="stCard">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            {f'<div style="margin-right: 10px;">{icon}</div>' if icon else ''}
            <h3 style="margin: 0;">{title}</h3>
        </div>
        <div>{content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def sample_data_info():
    """Show example of sample data format"""
    # Generate sample values
    np.random.seed(42)
    sample_values = [[round(np.random.randn(), 2) for _ in range(14)] for _ in range(5)]
    
    # Create table rows manually
    rows_html = ""
    for i in range(5):
        bg_color = "#2E3341" if i % 2 == 0 else "#1A1D28"
        row_html = f'<tr style="background-color: {bg_color};">'
        row_html += f'<td style="padding: 8px; text-align: center; border: 1px solid #2E3341;">{i/SAMPLING_RATE:.3f}s</td>'
        
        # Add cells for each channel
        for j in range(14):
            row_html += f'<td style="padding: 8px; text-align: center; border: 1px solid #2E3341;">{sample_values[i][j]}</td>'
        
        row_html += '</tr>'
        rows_html += row_html
    
    # Create headers
    headers_html = '<th style="padding: 8px; text-align: center; border: 1px solid #2E3341;">Time</th>'
    for ch in CHANNEL_NAMES:
        headers_html += f'<th style="padding: 8px; text-align: center; border: 1px solid #2E3341;">{ch}</th>'
    
    # Combine into full table
    info_html = f"""
    <div style="overflow-x: auto; margin-top: 10px;">
        <table style="width: 100%; border-collapse: collapse; border-radius: 5px; overflow: hidden;">
            <thead>
                <tr style="background-color: #4257B2;">
                    {headers_html}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    <p style="margin-top: 10px; font-style: italic; color: #AAAAAA;">
        Each row represents one time point (sampling rate = 128 Hz), and each column represents one EEG channel.
    </p>
    """
    return info_html

# Initialize page selection in session state if not present
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "Data Loading"

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="text-align: center;">Navigation</h2>', unsafe_allow_html=True)
    display_brain_icon()
    
    pages = {
        "Data Loading": "📊",
        "Data Visualization": "📈",
        "3D Visualization": "🧠",
        "Feature Extraction": "🔍",
        "About Dataset": "ℹ️"
    }
    
    # Create a custom navigation menu
    st.markdown('<div style="margin-bottom: 20px;">', unsafe_allow_html=True)
    
    for page_name, icon in pages.items():
        if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", 
                  help=f"Navigate to {page_name} page",
                  use_container_width=True):
            st.session_state.selected_page = page_name
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get current selected page from session state
    selected_page = st.session_state.selected_page
    
    # Additional sidebar info
    st.markdown('---')
    st.markdown('<h3 style="text-align: center;">STEW Dataset</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.9rem; color: #CCCCCC;">
        <p>• 48 subjects</p>
        <p>• 14 EEG channels</p>
        <p>• 128Hz sampling rate</p>
        <p>• 2.5 minutes per condition</p>
        <p>• Rest & multitasking conditions</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area - only show title and intro if on the data loading page
# if selected_page == "Data Loading":
#     st.markdown('<h1 style="text-align: center;">🧠 STEW EEG Analysis Dashboard</h1>', unsafe_allow_html=True)
if selected_page == "Data Loading":
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px; margin-bottom: 20px;">
            <h1 style="color: white;">🧠 STEW EEG Analysis Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Animated dashboard introduction
    intro_text = """
    <div style="text-align: center; margin-bottom: 30px; animation: fadeIn 1.5s ease-in-out;">
        <p style="font-size: 1.2rem; color: #CCCCCC; max-width: 800px; margin: 0 auto;">
            Visualize and analyze EEG data from the STEW dataset, containing recordings during rest and multitasking conditions.
        </p>
    </div>
    <style>
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
    """
    st.markdown(intro_text, unsafe_allow_html=True)

# Progress indicator
if st.session_state.show_loader:
    load_animation()
if st.session_state.progress > 0:
    progress_bar = st.progress(st.session_state.progress)

# Data Loading page
if selected_page == "Data Loading":
    st.markdown('<h2>Data Loading</h2>', unsafe_allow_html=True)
    
    # Main card for file upload instructions
    st.markdown("""
    <div class="stCard">
        <h3>Upload Dataset Files</h3>
        <p>Upload a directory containing the STEW dataset files or use the file uploader to select individual EEG recordings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File naming convention info in a separate card with proper styling
    st.markdown("""
    <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; margin: 15px 0;">
        <h4 style="margin-top: 0;">File naming convention:</h4>
        <ul>
            <li><code>subXX_lo.txt</code>: Subject XX at rest</li>
            <li><code>subXX_hi.txt</code>: Subject XX during multitasking</li>
            <li><code>ratings.txt</code>: Subject workload ratings (optional)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader for dataset directory
    uploaded_files = st.file_uploader(
        "Upload STEW dataset files",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload subject EEG files and ratings.txt"
    )
    
    # Create a new temporary directory if needed
    if uploaded_files and st.session_state.temp_dir is None:
        # Create persistent temp directory
        temp_dir = tempfile.mkdtemp()
        st.session_state.temp_dir = temp_dir
    
    if uploaded_files:
        st.info(f"Uploaded {len(uploaded_files)} files")
        
        # Save uploaded files to temporary directory
        if st.session_state.temp_dir:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            
            # Process data button
            if st.button("Process Data", key="process_data_btn", help="Process the uploaded dataset files"):
                show_loader()
                reset_progress()
                
                with st.spinner("Loading and processing data..."):
                    try:
                        # Load the dataset
                        update_progress(0.3)
                        dataset = st.session_state.processor.load_dataset(st.session_state.temp_dir)
                        update_progress(0.3)
                        st.session_state.dataset = dataset
                        
                        # Display dataset information
                        update_progress(0.3)
                        st.success(f"Successfully loaded {len(dataset['files'])} files")
                        
                        # Display subject files
                        files_df = pd.DataFrame({
                            'Filename': dataset['files'],
                            'Path': [os.path.basename(p) for p in dataset['file_paths']]
                        })
                        
                        st.markdown('<h3>Dataset Files</h3>', unsafe_allow_html=True)
                        st.dataframe(files_df, use_container_width=True)
                        
                        # Display ratings if available
                        if dataset['ratings']:
                            st.markdown('<h3>Subject Ratings</h3>', unsafe_allow_html=True)
                            ratings_df = pd.DataFrame([
                                {
                                    'Subject': subject,
                                    'Rest Rating': rating['rest'],
                                    'Test Rating': rating['test'],
                                    'Difference': rating['test'] - rating['rest']
                                } for subject, rating in dataset['ratings'].items()
                            ])
                            
                            # Create a styled ratings table
                            st.dataframe(
                                ratings_df,
                                use_container_width=True,
                                column_config={
                                    "Difference": st.column_config.ProgressColumn(
                                        "Rating Difference",
                                        format="%f",
                                        min_value=-8,
                                        max_value=8,
                                    ),
                                },
                            )
                            
                            # Plot ratings
                            st.plotly_chart(plot_subject_ratings(dataset['ratings']), use_container_width=True)
                        
                        st.session_state.data_loaded = True
                        update_progress(0.1)
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                    finally:
                        hide_loader()
    
    # Display sample data when no files are uploaded
    if not st.session_state.data_loaded:
        st.markdown('<h3>Sample Data Format</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="stCard">
            <p>The STEW dataset files should contain raw EEG data with 14 channels in the following format:</p>
        """, unsafe_allow_html=True)
        
        # Generate and display sample data
        st.markdown(sample_data_info(), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show an interactive demo of EEG processing
        st.markdown('<h3>How EEG Data is Processed</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_custom_card(
                "Raw EEG Signal",
                """
                <p>EEG data is initially captured as electrical signals from the scalp.</p>
                <p>The raw signal contains information across many frequency bands.</p>
                """,
                icon='<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M2 12H4M20 12H22M5 5L7 7M17 7L19 5M12 2V4" stroke="#4257B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 8C9.79086 8 8 9.79086 8 12C8 14.2091 9.79086 16 12 16C14.2091 16 16 14.2091 16 12C16 9.79086 14.2091 8 12 8Z" stroke="#8B44F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            )
            
        with col2:
            create_custom_card(
                "Frequency Analysis",
                """
                <p>The raw signals are transformed into the frequency domain using FFT.</p>
                <p>This reveals the power in different frequency bands (Delta, Theta, Alpha, Beta).</p>
                """,
                icon='<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M3 18H21M3 12H21M3 6H21" stroke="#8B44F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            )
            
        with col3:
            create_custom_card(
                "Feature Extraction",
                """
                <p>Band power features are extracted from the frequency analysis.</p>
                <p>These features can be used for cognitive workload classification.</p>
                """,
                icon='<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M9 5H7C5.89543 5 5 5.89543 5 7V19C5 20.1046 5.89543 21 7 21H17C18.1046 21 19 20.1046 19 19V7C19 5.89543 18.1046 5 17 5H15" stroke="#4257B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 12H12.01M12 16H12.01M12 8H12.01" stroke="#8B44F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            )

# Data Visualization page
elif selected_page == "Data Visualization":
    st.markdown('<h2>Data Visualization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first on the 'Data Loading' page.")
        
        # Show demo visualization with random data
        st.markdown('<h3>Demo Visualization</h3>', unsafe_allow_html=True)
        st.markdown("<p>Here's a demonstration of the visualizations with sample data:</p>", unsafe_allow_html=True)
        
        # Generate sample data
        np.random.seed(42)
        sample_data = np.random.randn(1000, 14)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Raw EEG", "Power Spectrum", "Band Power"])
        
        with tab1:
            # Raw EEG visualization
            st.markdown('<h3 style="margin-top: 0;">Raw EEG Signals</h3>', unsafe_allow_html=True)
            fig = plot_raw_eeg(
                sample_data[:500],
                sampling_rate=SAMPLING_RATE,
                channels=CHANNEL_NAMES,
                title="Demo Raw EEG Signals"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Power spectrum visualization
            st.markdown('<h3 style="margin-top: 0;">Power Spectrum</h3>', unsafe_allow_html=True)
            fig = plot_fft_spectrum(
                sample_data[:512],
                sampling_rate=SAMPLING_RATE,
                channel_idx=0,
                channel_name=CHANNEL_NAMES[0]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Band power visualization
            st.markdown('<h3 style="margin-top: 0;">Band Power</h3>', unsafe_allow_html=True)
            band_matrix, band_names = st.session_state.processor.compute_band_power_matrix(sample_data[:512])
            fig = plot_band_power_heatmap(
                band_matrix,
                band_names,
                channel_names=CHANNEL_NAMES
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        # Create a card for subject selection
        st.markdown("""
        <div class="stCard">
            <h3>Select Data to Visualize</h3>
            <p>Choose a subject and condition to visualize the EEG data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for selection controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject selection
            subjects = []
            for file in st.session_state.dataset['files']:
                parts = file.split('_')
                subject_id = parts[0][3:]  # Remove 'sub' prefix
                if subject_id not in subjects:
                    subjects.append(subject_id)
            
            selected_subject = st.selectbox(
                "Select Subject",
                sorted(subjects, key=int),
                help="Choose a subject to visualize"
            )
        
        with col2:
            # Condition selection
            selected_condition = st.radio(
                "Select Condition",
                ["Rest (lo)", "Test (hi)"],
                help="Choose the experimental condition"
            )
        
        condition_suffix = "lo" if selected_condition == "Rest (lo)" else "hi"
        selected_file = f"sub{selected_subject}_{condition_suffix}.txt"
        
        # Store the selections in session state so they're preserved on page reload
        if 'selected_subject' not in st.session_state:
            st.session_state.selected_subject = selected_subject
        else:
            st.session_state.selected_subject = selected_subject
            
        if 'selected_condition' not in st.session_state:
            st.session_state.selected_condition = selected_condition
        else:
            st.session_state.selected_condition = selected_condition
            
        # Find file in dataset
        file_idx = None
        for i, file in enumerate(st.session_state.dataset['files']):
            if file == selected_file:
                file_idx = i
                break
        
        if file_idx is not None:
            file_path = st.session_state.dataset['file_paths'][file_idx]
            
            # Check if file exists
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
            else:
                # Load and process data
                with st.spinner("Loading and processing data..."):
                    show_loader()
                    try:
                        data = st.session_state.processor.load_subject_data(file_path)
                        if data is None:
                            hide_loader()
                            st.error(f"Failed to load data for {selected_file}. Please check if the file exists and is in the correct format.")
                        else:
                            st.session_state.selected_subject_data = data
                            
                            # Display basic information
                            st.markdown(f"""
                            <div class="stCard">
                                <h3>Subject {selected_subject} - {selected_condition}</h3>
                                <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 10px;">
                                    <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; flex: 1;">
                                        <h4 style="margin-top: 0; font-size: 1rem;">Data Shape</h4>
                                        <p style="font-size: 1.2rem; margin-bottom: 0;">{data.shape[0]} × {data.shape[1]}</p>
                                        <p style="font-size: 0.8rem; color: #AAAAAA; margin-bottom: 0;">samples × channels</p>
                                    </div>
                                    <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; flex: 1;">
                                        <h4 style="margin-top: 0; font-size: 1rem;">Duration</h4>
                                        <p style="font-size: 1.2rem; margin-bottom: 0;">{data.shape[0] / SAMPLING_RATE:.2f}s</p>
                                        <p style="font-size: 0.8rem; color: #AAAAAA; margin-bottom: 0;">at {SAMPLING_RATE}Hz</p>
                                    </div>
                        """, unsafe_allow_html=True)
                        
                        # Rating information if available
                        subject_id = int(selected_subject)
                        if subject_id in st.session_state.dataset['ratings']:
                            rating_key = 'rest' if condition_suffix == 'lo' else 'test'
                            rating = st.session_state.dataset['ratings'][subject_id][rating_key]
                            st.markdown(f"""
                                <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; flex: 1;">
                                    <h4 style="margin-top: 0; font-size: 1rem;">Workload Rating</h4>
                                    <p style="font-size: 1.2rem; margin-bottom: 0;">{rating}/9</p>
                                    <p style="font-size: 0.8rem; color: #AAAAAA; margin-bottom: 0;">reported by subject</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # Create tabs for different visualizations
                        tab1, tab2, tab3 = st.tabs(["Raw EEG", "Power Spectrum", "Band Power"])
                        
                        with tab1:
                            # Raw EEG visualization
                            st.markdown('<h3>Raw EEG Signals</h3>', unsafe_allow_html=True)
                            
                            # Time range selection
                            max_time = data.shape[0] / SAMPLING_RATE
                            time_range = st.slider(
                                "Time Window (seconds)",
                                0.0, max_time, (0.0, min(10.0, max_time)),
                                step=0.1
                            )
                            
                            # Calculate sample indices
                            start_sample = int(time_range[0] * SAMPLING_RATE)
                            end_sample = int(time_range[1] * SAMPLING_RATE)
                            
                            # Plot raw EEG
                            fig = plot_raw_eeg(
                                data[start_sample:end_sample],
                                sampling_rate=SAMPLING_RATE,
                                channels=CHANNEL_NAMES,
                                title=f"Raw EEG - Subject {selected_subject} ({selected_condition})"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            # Power spectrum visualization
                            st.markdown('<h3>Power Spectrum</h3>', unsafe_allow_html=True)
                            
                            # Channel selection
                            selected_channel = st.selectbox(
                                "Select Channel",
                                range(len(CHANNEL_NAMES)),
                                format_func=lambda i: CHANNEL_NAMES[i]
                            )
                            
                            # Time window selection
                            window_options = [
                                (0, min(WINDOW_SIZE, data.shape[0])),
                                (min(WINDOW_SIZE, data.shape[0]), min(2*WINDOW_SIZE, data.shape[0])),
                                (min(2*WINDOW_SIZE, data.shape[0]), min(3*WINDOW_SIZE, data.shape[0]))
                            ]
                            window_labels = [f"{i*4}-{(i+1)*4} seconds" for i in range(3)]
                            
                            selected_window_idx = st.selectbox(
                                "Select Time Window",
                                range(len(window_options)),
                                format_func=lambda i: window_labels[i]
                            )
                            
                            start_sample, end_sample = window_options[selected_window_idx]
                            if end_sample > data.shape[0]:
                                end_sample = data.shape[0]
                            
                            # Plot power spectrum
                            fig = plot_fft_spectrum(
                                data[start_sample:end_sample],
                                sampling_rate=SAMPLING_RATE,
                                channel_idx=selected_channel,
                                channel_name=CHANNEL_NAMES[selected_channel]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab3:
                            # Band power visualization
                            st.markdown('<h3>Band Power</h3>', unsafe_allow_html=True)
                            
                            # Time window selection
                            window_options = [
                                (0, min(WINDOW_SIZE, data.shape[0])),
                                (min(WINDOW_SIZE, data.shape[0]), min(2*WINDOW_SIZE, data.shape[0])),
                                (min(2*WINDOW_SIZE, data.shape[0]), min(3*WINDOW_SIZE, data.shape[0]))
                            ]
                            window_labels = [f"{i*4}-{(i+1)*4} seconds" for i in range(3)]
                            
                            selected_window_idx = st.selectbox(
                                "Select Time Window for Band Power",
                                range(len(window_options)),
                                format_func=lambda i: window_labels[i]
                            )
                            
                            start_sample, end_sample = window_options[selected_window_idx]
                            if end_sample > data.shape[0]:
                                end_sample = data.shape[0]
                            
                            # Compute band power matrix
                            window_data = data[start_sample:end_sample]
                            band_matrix, band_names = st.session_state.processor.compute_band_power_matrix(window_data)
                            
                            # Plot band power heatmap
                            fig = plot_band_power_heatmap(
                                band_matrix,
                                band_names,
                                channel_names=CHANNEL_NAMES
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            hide_loader()
                    except Exception as e:
                        hide_loader()
                        st.error(f"Error processing data: {str(e)}")
        else:
            st.error(f"File not found: {selected_file}")

# 3D Visualization page
elif selected_page == "3D Visualization":
    st.markdown('<h2>3D Brain Visualization</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first on the 'Data Loading' page.")
        
        # Show demo 3D visualization with random data
        st.markdown('<h3>Demo 3D Brain Visualization</h3>', unsafe_allow_html=True)
        st.markdown("<p>Here's a demonstration of 3D visualizations with sample data:</p>", unsafe_allow_html=True)
        
        # Generate sample data
        np.random.seed(42)
        sample_data = np.random.randn(1000, 14)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["3D Brain Model", "Brain Activity", "Brain Connectivity", "Time Series"])
        
        with tab1:
            # Basic 3D brain with electrodes
            st.markdown('<h3 style="margin-top: 0;">3D Brain with Electrode Positions</h3>', unsafe_allow_html=True)
            
            with st.spinner("Generating 3D brain model..."):
                try:
                    # Create 3D brain model
                    fig = create_3d_brain_model()
                    # Add electrodes
                    fig = add_electrodes_to_brain(fig)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating 3D brain model: {str(e)}")
        
        with tab2:
            # Brain with activity visualization
            st.markdown('<h3 style="margin-top: 0;">Brain Activity Visualization</h3>', unsafe_allow_html=True)
            
            # Generate random activity data for demo
            activity_data = np.abs(np.random.randn(14))
            
            with st.spinner("Generating brain activity visualization..."):
                try:
                    # Create 3D brain model with activity data
                    fig = create_3d_brain_model()
                    fig = add_electrodes_to_brain(fig, activity_data)
                    fig = create_activity_surface(fig, activity_data)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating brain activity visualization: {str(e)}")
        
        with tab3:
            # Brain connectivity visualization
            st.markdown('<h3 style="margin-top: 0;">Brain Connectivity</h3>', unsafe_allow_html=True)
            
            with st.spinner("Generating brain connectivity visualization..."):
                try:
                    # Create brain connectivity visualization
                    fig = plot_brain_connectivity(sample_data, threshold=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating brain connectivity visualization: {str(e)}")
        
        with tab4:
            # 3D time series visualization
            st.markdown('<h3 style="margin-top: 0;">3D Time Series</h3>', unsafe_allow_html=True)
            
            with st.spinner("Generating 3D time series visualization..."):
                try:
                    # Create 3D time series visualization for a subset of channels
                    fig = create_3d_time_series(sample_data[:200], channel_indices=[0, 3, 7, 10])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating 3D time series visualization: {str(e)}")
    
    else:
        # User has loaded data, show real visualizations
        # Create columns for selection controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Subject selection
            subjects = []
            for file in st.session_state.dataset['files']:
                parts = file.split('_')
                subject_id = parts[0][3:]  # Remove 'sub' prefix
                if subject_id not in subjects:
                    subjects.append(subject_id)
            
            selected_subject = st.selectbox(
                "Select Subject",
                sorted(subjects, key=int),
                key="3d_subject",
                help="Choose a subject to visualize"
            )
        
        with col2:
            # Condition selection
            selected_condition = st.radio(
                "Select Condition",
                ["Rest (lo)", "Test (hi)"],
                key="3d_condition",
                help="Choose the experimental condition"
            )
        
        condition_suffix = "lo" if selected_condition == "Rest (lo)" else "hi"
        selected_file = f"sub{selected_subject}_{condition_suffix}.txt"
        
        # Find file in dataset
        file_idx = None
        for i, file in enumerate(st.session_state.dataset['files']):
            if file == selected_file:
                file_idx = i
                break
        
        if file_idx is not None:
            file_path = st.session_state.dataset['file_paths'][file_idx]
            
            # Check if file exists
            if not os.path.exists(file_path):
                st.error(f"File not found: {file_path}")
            else:
                # Load and process data
                with st.spinner("Loading and processing data for 3D visualization..."):
                    show_loader()
                    try:
                        data = st.session_state.processor.load_subject_data(file_path)
                        if data is None:
                            hide_loader()
                            st.error(f"Failed to load data for {selected_file}. Please check if the file exists and is in the correct format.")
                        else:
                            # Store data for visualization
                            st.session_state.selected_3d_data = data
                            
                            # Create tabs for different visualizations
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "3D Brain Model", 
                                "Band Activity", 
                                "Brain Connectivity",
                                "Animated Activity",
                                "3D Spectrogram"
                            ])
                            
                            with tab1:
                                # Basic 3D brain with electrodes
                                st.markdown('<h3 style="margin-top: 0;">3D Brain with Electrode Positions</h3>', unsafe_allow_html=True)
                                st.markdown("""
                                <p>This shows the standard 10-20 electrode positions used in the STEW dataset, placed on a 3D brain model.</p>
                                <p>You can interact with the model by clicking and dragging to rotate, scrolling to zoom, and right-clicking to pan.</p>
                                """, unsafe_allow_html=True)
                                
                                with st.spinner("Generating 3D brain model..."):
                                    try:
                                        # Create 3D brain model
                                        fig = create_3d_brain_model()
                                        # Add electrodes
                                        fig = add_electrodes_to_brain(fig)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating 3D brain model: {str(e)}")
                            
                            with tab2:
                                # Brain with frequency band activity
                                st.markdown('<h3 style="margin-top: 0;">Frequency Band Activity</h3>', unsafe_allow_html=True)
                                
                                # Band selection
                                selected_band = st.selectbox(
                                    "Select Frequency Band",
                                    ["delta", "theta", "alpha", "beta"],
                                    help="Choose a frequency band to visualize on the brain"
                                )
                                
                                # Time window selection
                                window_options = [
                                    (0, min(WINDOW_SIZE, data.shape[0])),
                                    (min(WINDOW_SIZE, data.shape[0]), min(2*WINDOW_SIZE, data.shape[0])),
                                    (min(2*WINDOW_SIZE, data.shape[0]), min(3*WINDOW_SIZE, data.shape[0]))
                                ]
                                window_labels = [f"{i*4}-{(i+1)*4} seconds" for i in range(3)]
                                
                                selected_window_idx = st.selectbox(
                                    "Select Time Window",
                                    range(len(window_options)),
                                    format_func=lambda i: window_labels[i],
                                    help="Choose a time window for band power visualization"
                                )
                                
                                start_sample, end_sample = window_options[selected_window_idx]
                                if end_sample > data.shape[0]:
                                    end_sample = data.shape[0]
                                
                                window_data = data[start_sample:end_sample]
                                
                                with st.spinner("Creating band activity visualization..."):
                                    try:
                                        # Create band topography
                                        fig = create_band_topography(window_data, band=selected_band)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating band activity visualization: {str(e)}")
                            
                            with tab3:
                                # Brain connectivity visualization
                                st.markdown('<h3 style="margin-top: 0;">Brain Connectivity</h3>', unsafe_allow_html=True)
                                st.markdown("""
                                <p>This visualization shows the functional connectivity between different brain regions.</p>
                                <p>Red lines indicate positive correlations, while blue lines indicate negative correlations.</p>
                                <p>The thickness of each line represents the strength of the correlation.</p>
                                """, unsafe_allow_html=True)
                                
                                # Correlation threshold adjustment
                                threshold = st.slider(
                                    "Correlation Threshold",
                                    min_value=0.1,
                                    max_value=0.9,
                                    value=0.4,
                                    step=0.05,
                                    help="Adjust the threshold for displaying connections between electrodes"
                                )
                                
                                # Time window selection
                                window_size = min(WINDOW_SIZE, data.shape[0])
                                window_start = st.slider(
                                    "Time Window Start (seconds)",
                                    min_value=0.0,
                                    max_value=max(0.0, (data.shape[0] - window_size) / SAMPLING_RATE),
                                    value=0.0,
                                    step=1.0,
                                    help="Choose the starting time for the connectivity analysis"
                                )
                                
                                start_sample = int(window_start * SAMPLING_RATE)
                                end_sample = min(start_sample + window_size, data.shape[0])
                                window_data = data[start_sample:end_sample]
                                
                                with st.spinner("Generating brain connectivity visualization..."):
                                    try:
                                        # Create brain connectivity visualization
                                        fig = plot_brain_connectivity(window_data, threshold=threshold)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating brain connectivity visualization: {str(e)}")
                            
                            with tab4:
                                # Animated brain activity
                                st.markdown('<h3 style="margin-top: 0;">Animated Brain Activity</h3>', unsafe_allow_html=True)
                                st.markdown("""
                                <p>This visualization shows how brain activity changes over time.</p>
                                <p>Use the play button to start the animation, and the slider to navigate to specific time points.</p>
                                """, unsafe_allow_html=True)
                                
                                # Animation options
                                window_size = st.slider(
                                    "Window Size for Animation (samples)",
                                    min_value=64,
                                    max_value=512,
                                    value=128,
                                    step=64,
                                    help="Set the window size for each animation frame"
                                )
                                
                                # Maximum duration to avoid performance issues
                                max_duration = min(10 * SAMPLING_RATE, data.shape[0])
                                
                                if data.shape[0] > max_duration:
                                    st.info(f"Using only the first {max_duration/SAMPLING_RATE:.1f} seconds for animation to ensure performance")
                                    animation_data = data[:max_duration]
                                else:
                                    animation_data = data
                                
                                with st.spinner("Generating animated brain activity..."):
                                    try:
                                        # Create animated brain activity
                                        fig = animate_brain_activity(animation_data, window_size=window_size)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating animated brain activity: {str(e)}")
                            
                            with tab5:
                                # 3D spectrogram
                                st.markdown('<h3 style="margin-top: 0;">3D Spectrogram</h3>', unsafe_allow_html=True)
                                st.markdown("""
                                <p>This visualizes the power spectral density of a selected channel in 3D.</p>
                                <p>The x-axis represents time, the y-axis represents frequency, and the z-axis (height) represents power.</p>
                                """, unsafe_allow_html=True)
                                
                                # Channel selection
                                selected_channel = st.selectbox(
                                    "Select Channel for Spectrogram",
                                    range(len(CHANNEL_NAMES)),
                                    format_func=lambda i: CHANNEL_NAMES[i],
                                    help="Choose an EEG channel for the 3D spectrogram"
                                )
                                
                                with st.spinner("Generating 3D spectrogram..."):
                                    try:
                                        # Create 3D spectrogram
                                        fig, brain_fig = plot_3d_spectrogram(
                                            data, 
                                            channel_idx=selected_channel,
                                            fs=SAMPLING_RATE
                                        )
                                        
                                        # Create columns for layout
                                        spec_col, brain_col = st.columns([2, 1])
                                        
                                        with spec_col:
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        with brain_col:
                                            st.markdown("<h4>Selected Channel Position</h4>", unsafe_allow_html=True)
                                            st.plotly_chart(brain_fig, use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Error creating 3D spectrogram: {str(e)}")
                            
                            hide_loader()
                    except Exception as e:
                        hide_loader()
                        st.error(f"Error processing data: {str(e)}")
        else:
            st.error(f"File not found: {selected_file}")

# Feature Extraction page
elif selected_page == "Feature Extraction":
    st.markdown('<h2>Feature Extraction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first on the 'Data Loading' page.")
        
        # Show explanation with sample data
        st.markdown("""
        <div class="stCard">
            <h3>Feature Extraction Explanation</h3>
            <p>This page demonstrates how we extract features from EEG data for analysis.
            We process the raw EEG signals to compute:</p>
            
            <ol>
                <li><strong>Spectral Power Features</strong>: Power in different frequency bands (Delta, Theta, Alpha, Beta)</li>
                <li><strong>Time Window Segmentation</strong>: Splitting long recordings into fixed-size windows</li>
            </ol>
            
            <p>Once data is loaded, you'll be able to extract these features from the actual dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a visual explanation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_custom_card(
                "Raw EEG",
                """
                <div style="text-align: center;">
                    <svg width="200" height="100" viewBox="0 0 200 100">
                        <path d="M0,50 C10,30 20,70 30,50 C40,30 50,70 60,50 C70,30 80,70 90,50 C100,30 110,70 120,50 C130,30 140,70 150,50 C160,30 170,70 180,50 C190,30 200,70 200,50" 
                            stroke="#4257B2" fill="none" stroke-width="2"/>
                    </svg>
                </div>
                <p>Time-domain signal from electrodes</p>
                """,
            )
            
        with col2:
            create_custom_card(
                "Frequency Analysis",
                """
                <div style="text-align: center;">
                    <svg width="200" height="100" viewBox="0 0 200 100">
                        <rect x="20" y="70" width="20" height="20" fill="#4257B2" opacity="0.7"/>
                        <rect x="60" y="40" width="20" height="50" fill="#6A73E3" opacity="0.7"/>
                        <rect x="100" y="60" width="20" height="30" fill="#8B44F6" opacity="0.7"/>
                        <rect x="140" y="50" width="20" height="40" fill="#B088F9" opacity="0.7"/>
                        <line x1="0" y1="90" x2="200" y2="90" stroke="#FFFFFF" stroke-width="1"/>
                        <text x="30" y="95" font-size="8" fill="#AAAAAA">Delta</text>
                        <text x="70" y="95" font-size="8" fill="#AAAAAA">Theta</text>
                        <text x="110" y="95" font-size="8" fill="#AAAAAA">Alpha</text>
                        <text x="150" y="95" font-size="8" fill="#AAAAAA">Beta</text>
                    </svg>
                </div>
                <p>Frequency-domain representation</p>
                """,
            )
            
        with col3:
            create_custom_card(
                "Band Power Features",
                """
                <div style="text-align: center;">
                    <svg width="200" height="100" viewBox="0 0 200 100">
                        <rect x="40" y="20" width="120" height="60" fill="#1A1D28"/>
                        <rect x="50" y="30" width="20" height="10" fill="#FF0000" opacity="0.5"/>
                        <rect x="50" y="40" width="20" height="10" fill="#00FF00" opacity="0.5"/>
                        <rect x="50" y="50" width="20" height="10" fill="#0000FF" opacity="0.5"/>
                        <rect x="50" y="60" width="20" height="10" fill="#FFFF00" opacity="0.5"/>
                        
                        <rect x="80" y="30" width="20" height="10" fill="#FF0000" opacity="0.7"/>
                        <rect x="80" y="40" width="20" height="10" fill="#00FF00" opacity="0.3"/>
                        <rect x="80" y="50" width="20" height="10" fill="#0000FF" opacity="0.8"/>
                        <rect x="80" y="60" width="20" height="10" fill="#FFFF00" opacity="0.2"/>
                        
                        <rect x="110" y="30" width="20" height="10" fill="#FF0000" opacity="0.3"/>
                        <rect x="110" y="40" width="20" height="10" fill="#00FF00" opacity="0.9"/>
                        <rect x="110" y="50" width="20" height="10" fill="#0000FF" opacity="0.4"/>
                        <rect x="110" y="60" width="20" height="10" fill="#FFFF00" opacity="0.7"/>
                    </svg>
                </div>
                <p>Feature matrix for analysis</p>
                """,
            )
    else:
        st.markdown("""
        <div class="stCard">
            <h3>Extract Features</h3>
            <p>This process will extract spectral features from all subjects in the dataset.
            The extracted features can be used for visualization and analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Extract features button
        if st.button("Extract Features", key="extract_features_btn", help="Extract features from the entire dataset"):
            show_loader()
            reset_progress()
            
            with st.spinner("Extracting features from all subjects..."):
                try:
                    # Extract features from dataset
                    update_progress(0.4)
                    
                    # Fix the file not found error by ensuring we have a valid directory path
                    data_dir = os.path.dirname(st.session_state.dataset['file_paths'][0])
                    
                    extracted = st.session_state.processor.extract_features(data_dir)
                    update_progress(0.4)
                    
                    st.session_state.extracted_features = extracted
                    
                    # Display information
                    st.success(f"Successfully extracted features from {len(extracted['metadata'])} windows")
                    
                    # Show data summary
                    st.markdown("""
                    <div class="stCard">
                        <h3>Extracted Features Summary</h3>
                        <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px;">
                    """, unsafe_allow_html=True)
                    
                    # Display shape information in cards
                    st.markdown(f"""
                        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
                            <h4 style="margin-top: 0; font-size: 1rem;">Raw Data</h4>
                            <p style="font-size: 1.2rem; margin-bottom: 0;">{extracted['X_raw'].shape}</p>
                            <p style="font-size: 0.8rem; color: #AAAAAA; margin-bottom: 0;">windows × samples × channels</p>
                        </div>
                        
                        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
                            <h4 style="margin-top: 0; font-size: 1rem;">Spectral Features</h4>
                            <p style="font-size: 1.2rem; margin-bottom: 0;">{extracted['X_spec'].shape}</p>
                            <p style="font-size: 0.8rem; color: #AAAAAA; margin-bottom: 0;">windows × features</p>
                        </div>
                        
                        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
                            <h4 style="margin-top: 0; font-size: 1rem;">Labels</h4>
                            <p style="font-size: 1.2rem; margin-bottom: 0;">{extracted['y'].shape}</p>
                            <p style="font-size: 0.8rem; color: #AAAAAA; margin-bottom: 0;">windows (0=low, 1=high)</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Show class distribution
                    classes, counts = np.unique(extracted['y'], return_counts=True)
                    class_names = ['Low Workload', 'High Workload']
                    
                    # Display as a pie chart
                    class_dist_fig = go.Figure(data=[go.Pie(
                        labels=[class_names[c] for c in classes],
                        values=counts,
                        hole=.4,
                        marker_colors=['#4257B2', '#8B44F6']
                    )])
                    
                    class_dist_fig.update_layout(
                        title_text="Class Distribution",
                        annotations=[dict(text='Distribution', x=0.5, y=0.5, font_size=15, showarrow=False)],
                        height=350
                    )
                    
                    st.plotly_chart(class_dist_fig, use_container_width=True)
                    
                    # Visualize sample window
                    st.markdown('<h3>Sample Feature Visualization</h3>', unsafe_allow_html=True)
                    
                    # Create tabs for different visualizations
                    tab1, tab2 = st.tabs(["Raw EEG Window", "Band Power Features"])
                    
                    with tab1:
                        # Select a random sample
                        sample_idx = np.random.randint(0, len(extracted['X_raw']))
                        sample_window = extracted['X_raw'][sample_idx]
                        sample_label = extracted['y'][sample_idx]
                        sample_metadata = extracted['metadata'][sample_idx]
                        
                        # Display sample metadata
                        st.markdown(f"""
                        <div class="stCard">
                            <h4>Sample Window Information</h4>
                            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px;">
                                <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; flex: 1;">
                                    <p style="font-size: 0.9rem; margin: 0;">Subject</p>
                                    <p style="font-size: 1.3rem; margin: 0;">{sample_metadata['subject']}</p>
                                </div>
                                <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; flex: 1;">
                                    <p style="font-size: 0.9rem; margin: 0;">Condition</p>
                                    <p style="font-size: 1.3rem; margin: 0;">{sample_metadata['condition']}</p>
                                </div>
                                <div style="background-color: #2E3341; padding: 10px; border-radius: 5px; flex: 1;">
                                    <p style="font-size: 0.9rem; margin: 0;">Class Label</p>
                                    <p style="font-size: 1.3rem; margin: 0;">{class_names[sample_label]}</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Plot raw EEG window
                        fig = plot_raw_eeg(
                            sample_window,
                            sampling_rate=SAMPLING_RATE,
                            channels=CHANNEL_NAMES,
                            title=f"Raw EEG Window - Subject {sample_metadata['subject']} ({sample_metadata['condition']})"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Compute band power matrix for the sample
                        band_matrix, band_names = st.session_state.processor.compute_band_power_matrix(sample_window)
                        
                        # Plot band power heatmap
                        fig = plot_band_power_heatmap(
                            band_matrix,
                            band_names,
                            channel_names=CHANNEL_NAMES
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    update_progress(0.2)
                except Exception as e:
                    st.error(f"Error extracting features: {str(e)}")
                finally:
                    hide_loader()

# About Dataset page
elif selected_page == "About Dataset":
    st.markdown('<h2>About the STEW Dataset</h2>', unsafe_allow_html=True)
    
    # Create an animated introduction - main card header
    st.markdown("""
    <div class="stCard" style="animation: fadeIn 1s ease-in-out;">
        <h3>STEW: Simultaneous Task EEG Workload Dataset</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset description in a separate markdown to avoid HTML rendering issues
    st.markdown("""
    <div class="stCard">
    This dataset consists of raw EEG data from 48 subjects who participated in a multitasking workload 
    experiment utilizing the SIMKAP multitasking test. The subjects' brain activity at rest was also 
    recorded before the test and is included as well.
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset specs in cards
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
            <h4 style="margin-top: 0;">Device</h4>
            <p>Emotiv EPOC with 14 channels</p>
        </div>
        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
            <h4 style="margin-top: 0;">Sampling Rate</h4>
            <p>128 Hz</p>
        </div>
        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
            <h4 style="margin-top: 0;">Duration</h4>
            <p>2.5 minutes per condition</p>
        </div>
        <div style="background-color: #2E3341; padding: 15px; border-radius: 5px; flex: 1;">
            <h4 style="margin-top: 0;">Conditions</h4>
            <p>Rest (lo) and Multitasking Test (hi)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # File naming convention
    st.markdown("""
    <div class="stCard">
        <h4>File Naming Convention</h4>
        <p><code>subXX_YY.txt</code> where XX is the subject number (01-48) and YY is the condition (lo for rest, hi for test).</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Channels information
    st.markdown("""
    <div class="stCard">
        <h4>Channels</h4>
        <p>The 14 EEG channels are arranged in the following order:</p>
        <p>AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Workload ratings
    st.markdown("""
    <div class="stCard">
        <h4>Workload Ratings</h4>
        <p>After each recording session, subjects rated their perceived mental workload on a scale of 1 to 9.
        These ratings are provided in the <code>ratings.txt</code> file.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display channel positions
    st.markdown('<h3>EEG Channel Positions</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a custom interactive head map using Plotly
        # Approximate positions on a 2D head model
        channels = CHANNEL_NAMES
        positions = {
            'AF3': (-0.3, 0.8), 'F7': (-0.8, 0.4), 'F3': (-0.4, 0.6), 'FC5': (-0.7, 0.2),
            'T7': (-1.0, 0.0), 'P7': (-0.7, -0.6), 'O1': (-0.3, -0.9), 'O2': (0.3, -0.9),
            'P8': (0.7, -0.6), 'T8': (1.0, 0.0), 'FC6': (0.7, 0.2), 'F4': (0.4, 0.6), 
            'F8': (0.8, 0.4), 'AF4': (0.3, 0.8)
        }
        
        x = [positions[ch][0] for ch in channels]
        y = [positions[ch][1] for ch in channels]
        text = channels
        
        # Create a head outline (circle)
        theta = np.linspace(0, 2*np.pi, 100)
        head_x = np.cos(theta)
        head_y = np.sin(theta)
        
        # Nose at (0, 1)
        nose_x = [0, 0.1, 0, -0.1]
        nose_y = [1, 1.1, 1.2, 1.1]
        
        # Create figure
        fig = go.Figure()
        
        # Add head outline
        fig.add_trace(go.Scatter(
            x=head_x, y=head_y,
            mode='lines',
            line=dict(color='#4257B2', width=2),
            name='Head Outline'
        ))
        
        # Add nose
        fig.add_trace(go.Scatter(
            x=nose_x, y=nose_y,
            mode='lines',
            line=dict(color='#4257B2', width=2),
            name='Nose'
        ))
        
        # Add channel positions with hover info
        for i, ch in enumerate(channels):
            fig.add_trace(go.Scatter(
                x=[x[i]], y=[y[i]],
                mode='markers+text',
                marker=dict(
                    size=15, 
                    color='#8B44F6',
                    line=dict(width=2, color='#FFFFFF')
                ),
                text=[ch],
                textposition="top center",
                name=ch,
                hoverinfo='text',
                hovertext=f"Channel: {ch}"
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Emotiv EPOC 14-Channel EEG Positions",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(range=[-1.2, 1.2], title="", showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1.2, 1.2], title="", showgrid=False, zeroline=False, showticklabels=False),
            width=600,
            height=500,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Add annotations for regions
        annotations = [
            dict(x=0, y=0.9, text="Frontal", showarrow=False, font=dict(color="#AAAAAA")),
            dict(x=0, y=-0.9, text="Occipital", showarrow=False, font=dict(color="#AAAAAA")),
            dict(x=-0.9, y=0, text="Left<br>Temporal", showarrow=False, font=dict(color="#AAAAAA")),
            dict(x=0.9, y=0, text="Right<br>Temporal", showarrow=False, font=dict(color="#AAAAAA")),
        ]
        fig.update_layout(annotations=annotations)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Channel groupings - use a header
        st.markdown('<h4>Channel Groupings</h4>', unsafe_allow_html=True)
        
        # Create a card for the groupings
        st.markdown("""
        <div class="stCard">
            <div style="margin-bottom: 10px;">
                <p style="margin-bottom: 5px; color: #4257B2; font-weight: bold;">Frontal</p>
                <p style="margin: 0;">AF3, F7, F3, F4, F8, AF4</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add each channel group separately
        st.markdown("""
        <div class="stCard">
            <div style="margin-bottom: 10px;">
                <p style="margin-bottom: 5px; color: #4257B2; font-weight: bold;">Temporal</p>
                <p style="margin: 0;">T7, T8</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stCard">
            <div style="margin-bottom: 10px;">
                <p style="margin-bottom: 5px; color: #4257B2; font-weight: bold;">Central</p>
                <p style="margin: 0;">FC5, FC6</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stCard">
            <div style="margin-bottom: 10px;">
                <p style="margin-bottom: 5px; color: #4257B2; font-weight: bold;">Parietal</p>
                <p style="margin: 0;">P7, P8</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stCard">
            <div style="margin-bottom: 10px;">
                <p style="margin-bottom: 5px; color: #4257B2; font-weight: bold;">Occipital</p>
                <p style="margin: 0;">O1, O2</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Add note about the 10-20 system
        st.markdown("""
        <div style="margin-top: 15px; font-style: italic; color: #AAAAAA;">
            These channel positions follow the international 10-20 system for EEG electrode placement.
        </div>
        """, unsafe_allow_html=True)
    
    # Research applications section
    st.markdown('<h3>Research Applications</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_custom_card(
            "Cognitive Workload Detection",
            """
            <p>EEG signals can be used to detect and quantify cognitive workload levels in real-time.</p>
            <p>Applications include:</p>
            <ul>
                <li>Adaptive user interfaces</li>
                <li>Operator monitoring systems</li>
                <li>Educational technology</li>
            </ul>
            """,
            icon='<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z" fill="#4257B2"/><path d="M12 17C14.7614 17 17 14.7614 17 12C17 9.23858 14.7614 7 12 7C9.23858 7 7 9.23858 7 12C7 14.7614 9.23858 17 12 17Z" fill="#8B44F6"/></svg>'
        )
    
    with col2:
        create_custom_card(
            "Brain-Computer Interfaces",
            """
            <p>The dataset can be used to develop algorithms for Brain-Computer Interfaces (BCIs).</p>
            <p>Potential uses include:</p>
            <ul>
                <li>Assistive technology for disabled individuals</li>
                <li>Neurofeedback training</li>
                <li>Novel human-computer interaction methods</li>
            </ul>
            """,
            icon='<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M4 6H20M4 12H20M4 18H20" stroke="#4257B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M9 10C9.55228 10 10 9.55228 10 9C10 8.44772 9.55228 8 9 8C8.44772 8 8 8.44772 8 9C8 9.55228 8.44772 10 9 10Z" fill="#8B44F6"/><path d="M15 16C15.5523 16 16 15.5523 16 15C16 14.4477 15.5523 14 15 14C14.4477 14 14 14.4477 14 15C14 15.5523 14.4477 16 15 16Z" fill="#8B44F6"/></svg>'
        )
    
    # References
    st.markdown('<h3>References</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stCard">
        <ol>
            <li>STEW Dataset: <a href="https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset" target="_blank" style="color: #4257B2;">IEEE Dataport - STEW: Simultaneous Task EEG Workload Dataset</a></li>
            <li>Emotiv EPOC Technical Specifications: <a href="https://www.emotiv.com/epoc/" target="_blank" style="color: #4257B2;">https://www.emotiv.com/epoc/</a></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)