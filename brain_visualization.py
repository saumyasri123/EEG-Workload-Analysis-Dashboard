import numpy as np
import plotly.graph_objects as go
import streamlit as st
import os
import tempfile
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Standard 10-20 EEG electrode positions in 3D coordinates
ELECTRODE_POSITIONS = {
    'AF3': [-0.27, 0.83, 0.48],
    'F7': [-0.80, 0.55, 0.22],
    'F3': [-0.40, 0.69, 0.60],
    'FC5': [-0.72, 0.38, 0.58],
    'T7': [-0.95, 0.01, 0.28],
    'P7': [-0.76, -0.64, 0.08],
    'O1': [-0.33, -0.94, 0.04],
    'O2': [0.33, -0.94, 0.04],
    'P8': [0.76, -0.64, 0.08],
    'T8': [0.95, 0.01, 0.28],
    'FC6': [0.72, 0.38, 0.58],
    'F4': [0.40, 0.69, 0.60],
    'F8': [0.80, 0.55, 0.22],
    'AF4': [0.27, 0.83, 0.48]
}

# Channel names from the STEW dataset to match with standard positions
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

def create_3d_brain_model():
    """
    Create a simplified 3D brain model with Plotly.
    
    Returns:
    --------
    plotly.graph_objects.Figure
        3D brain model
    """
    # Create points for a simplified brain surface
    theta = np.linspace(0, 2*np.pi, 30)
    phi = np.linspace(0, np.pi, 20)
    
    # Create a sphere
    r = 0.9  # Radius of the brain
    x = r * np.outer(np.cos(theta), np.sin(phi))
    y = r * np.outer(np.sin(theta), np.sin(phi))
    z = r * np.outer(np.ones(30), np.cos(phi))
    
    # Split into two hemispheres
    x_left = x.copy()
    x_left[x_left > 0] = np.nan
    
    x_right = x.copy()
    x_right[x_right < 0] = np.nan

    # Create surface for left hemisphere
    brain_left = go.Surface(
        x=x_left,
        y=y,
        z=z,
        colorscale=[[0, '#444444'], [1, '#999999']],
        opacity=0.2,
        showscale=False,
        hoverinfo='none'
    )
    
    # Create surface for right hemisphere
    brain_right = go.Surface(
        x=x_right,
        y=y,
        z=z,
        colorscale=[[0, '#444444'], [1, '#999999']],
        opacity=0.2,
        showscale=False,
        hoverinfo='none'
    )
    
    # Create a figure
    fig = go.Figure(data=[brain_left, brain_right])
    
    # Update the layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=0, z=0)
            ),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700
    )
    
    return fig

def add_electrodes_to_brain(fig, activity_data=None):
    """
    Add electrodes to the 3D brain model with optional activity data.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        3D brain model figure
    activity_data : numpy.ndarray, optional
        Electrode activity data to visualize (default: None)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Updated figure with electrodes
    """
    # Extract electrode positions
    x = []
    y = []
    z = []
    labels = []
    
    for ch_name in CHANNEL_NAMES:
        pos = ELECTRODE_POSITIONS[ch_name]
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
        labels.append(ch_name)
    
    # Determine marker properties based on activity data
    marker_size = 8
    marker_color = 'blue'
    marker_opacity = 0.7
    colorscale = 'Viridis'
    
    if activity_data is not None:
        # Normalize data to 0-1 range for coloring
        norm_data = (activity_data - np.min(activity_data)) / (np.max(activity_data) - np.min(activity_data))
        marker_color = norm_data
        marker_size = 10
    
    # Add electrodes as scatter 3d
    electrodes = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers+text',
        marker=dict(
            size=marker_size,
            color=marker_color,
            colorscale=colorscale,
            opacity=marker_opacity,
            colorbar=dict(
                title='Activity',
                thickness=15,
                len=0.5,
                x=0.85,
                xanchor='left'
            ) if activity_data is not None else None
        ),
        text=labels,
        textposition='top center',
        hoverinfo='text',
        hovertext=[f"{ch}: {activity_data[i]:.2f}" if activity_data is not None else ch for i, ch in enumerate(labels)]
    )
    
    fig.add_trace(electrodes)
    return fig

def create_activity_surface(fig, activity_data):
    """
    Add a continuous activity surface interpolated from electrode data.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        3D brain model figure
    activity_data : numpy.ndarray
        Electrode activity data to visualize
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Updated figure with activity surface
    """
    # Extract electrode positions
    points = np.array([ELECTRODE_POSITIONS[ch] for ch in CHANNEL_NAMES])
    
    # Create a grid on top of the brain for interpolation
    x_range = np.linspace(-1, 1, 50)
    y_range = np.linspace(-1, 1, 50)
    z_range = np.linspace(-0.2, 1, 20)
    
    # Filter grid points to roughly match brain shape
    grid_points = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                r = np.sqrt(x**2 + y**2 + z**2)
                if 0.8 < r < 1.1:  # Only keep points near the brain surface
                    grid_points.append([x, y, z])
    
    grid_points = np.array(grid_points)
    
    # Interpolate activity values onto the grid
    grid_activity = griddata(points, activity_data, grid_points, method='linear', fill_value=np.min(activity_data))
    
    # Normalize data to 0-1 range for coloring
    norm_data = (grid_activity - np.min(grid_activity)) / (np.max(grid_activity) - np.min(grid_activity))
    
    # Add activity surface
    activity_surface = go.Scatter3d(
        x=grid_points[:, 0],
        y=grid_points[:, 1],
        z=grid_points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=norm_data,
            colorscale='Viridis',
            opacity=0.5,
            showscale=False
        ),
        hoverinfo='none'
    )
    
    fig.add_trace(activity_surface)
    return fig

def animate_brain_activity(eeg_data, window_size=128):
    """
    Create an animated 3D brain model showing changes in EEG activity over time.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    window_size : int
        Window size for animation frames
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Animated 3D brain model
    """
    if eeg_data.shape[0] < window_size:
        window_size = eeg_data.shape[0]
    
    # Number of frames for animation
    num_frames = eeg_data.shape[0] // window_size
    if num_frames > 10:  # Limit to 10 frames for performance
        step_size = num_frames // 10
        frame_indices = [i * window_size for i in range(0, num_frames, step_size)]
    else:
        frame_indices = [i * window_size for i in range(num_frames)]
    
    # Create base brain model
    fig = create_3d_brain_model()
    
    # Create frames for animation
    frames = []
    
    for i, idx in enumerate(frame_indices):
        # Get data for this frame
        frame_data = eeg_data[idx:idx+window_size, :]
        
        # Calculate activity metric (e.g., power)
        activity = np.mean(np.abs(frame_data), axis=0)
        
        # Create a new frame
        frame_fig = go.Figure(fig.data)
        frame_fig = add_electrodes_to_brain(frame_fig, activity)
        frame_fig = create_activity_surface(frame_fig, activity)
        
        # Add to frames
        frames.append(go.Frame(
            data=frame_fig.data,
            name=f"frame_{i}",
            traces=[0, 1, 2, 3]  # brain_left, brain_right, electrodes, activity_surface
        ))
    
    # Add the frames to the figure
    fig.frames = frames
    
    # Add slider and play button
    sliders = [dict(
        active=0,
        steps=[dict(
            method="animate",
            args=[[f"frame_{i}"], dict(
                frame=dict(duration=500, redraw=True),
                mode="immediate",
                transition=dict(duration=300)
            )],
            label=f"{(idx / eeg_data.shape[0]) * 100:.0f}%"
        ) for i, idx in enumerate(frame_indices)],
        transition=dict(duration=300),
        x=0.1,
        xanchor="left",
        y=0,
        yanchor="bottom",
        currentvalue=dict(
            font=dict(size=12),
            prefix="Time: ",
            visible=True,
            xanchor="right"
        ),
        len=0.9
    )]
    
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=500, redraw=True),
                    mode="immediate",
                    fromcurrent=True,
                    transition=dict(duration=300)
                )]
            ), dict(
                label="Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0)
                )]
            )]
        )],
        sliders=sliders
    )
    
    return fig

def plot_brain_connectivity(eeg_data, threshold=0.5):
    """
    Create a 3D brain model with connectivity visualization based on correlation.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    threshold : float
        Correlation threshold for drawing connections (default: 0.5)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        3D brain model with connectivity
    """
    # Create base brain model
    fig = create_3d_brain_model()
    
    # Add electrodes
    fig = add_electrodes_to_brain(fig)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(eeg_data.T)
    
    # Extract electrode positions
    positions = np.array([ELECTRODE_POSITIONS[ch] for ch in CHANNEL_NAMES])
    
    # Add connections between electrodes
    for i in range(len(CHANNEL_NAMES)):
        for j in range(i + 1, len(CHANNEL_NAMES)):
            # Check if correlation exceeds threshold
            if abs(corr_matrix[i, j]) > threshold:
                # Create a line between electrodes
                x_line = [positions[i, 0], positions[j, 0]]
                y_line = [positions[i, 1], positions[j, 1]]
                z_line = [positions[i, 2], positions[j, 2]]
                
                # Set line properties based on correlation
                line_width = abs(corr_matrix[i, j]) * 6
                line_color = 'red' if corr_matrix[i, j] > 0 else 'blue'
                
                # Add the line
                fig.add_trace(go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode='lines',
                    line=dict(
                        width=line_width,
                        color=line_color
                    ),
                    opacity=abs(corr_matrix[i, j]),
                    hoverinfo='text',
                    hovertext=f"{CHANNEL_NAMES[i]} - {CHANNEL_NAMES[j]}: {corr_matrix[i, j]:.2f}"
                ))
    
    # Update layout for better visualization
    fig.update_layout(
        title="Brain Connectivity",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=0, z=0)
            )
        ),
        height=700
    )
    
    return fig

def plot_3d_spectrogram(eeg_data, channel_idx=0, fs=128, nperseg=128):
    """
    Create a 3D spectrogram visualization for a selected EEG channel.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    channel_idx : int
        Index of the channel to visualize (default: 0)
    fs : int
        Sampling frequency in Hz (default: 128)
    nperseg : int
        Length of each segment for FFT (default: 128)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        3D spectrogram visualization
    """
    from scipy.signal import spectrogram
    
    # Extract selected channel data
    channel_data = eeg_data[:, channel_idx]
    
    # Calculate spectrogram
    f, t, Sxx = spectrogram(channel_data, fs=fs, nperseg=nperseg)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create 3D surface
    fig = go.Figure(data=[go.Surface(
        z=Sxx_db,
        x=t,
        y=f,
        colorscale='Viridis',
        opacity=0.8
    )])
    
    # Update layout
    fig.update_layout(
        title=f"3D Spectrogram - Channel {CHANNEL_NAMES[channel_idx]}",
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis_title="Power (dB)",
            xaxis=dict(nticks=10),
            yaxis=dict(nticks=10),
            zaxis=dict(nticks=10)
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Add electrode position indicator
    brain_fig = create_3d_brain_model()
    
    # Highlight the selected electrode
    highlight_data = np.zeros(len(CHANNEL_NAMES))
    highlight_data[channel_idx] = 1.0
    
    brain_fig = add_electrodes_to_brain(brain_fig, highlight_data)
    
    return fig, brain_fig

def create_band_topography(eeg_data, band='alpha'):
    """
    Create a topographic map of brain activity for a specific frequency band.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    band : str
        Frequency band to visualize ('delta', 'theta', 'alpha', 'beta', 'gamma')
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Topographic map of brain activity
    """
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    # Calculate band power for each channel
    from scipy.signal import welch
    fs = 128  # Sampling frequency
    
    band_power = []
    for ch in range(eeg_data.shape[1]):
        # Calculate PSD
        f, psd = welch(eeg_data[:, ch], fs=fs, nperseg=4*fs)
        
        # Find indices corresponding to the frequency band
        idx_band = np.logical_and(f >= bands[band][0], f <= bands[band][1])
        
        # Calculate mean power in the band
        band_power.append(np.mean(psd[idx_band]))
    
    # Create figure with brain model
    fig = create_3d_brain_model()
    
    # Add electrodes with activity data
    fig = add_electrodes_to_brain(fig, np.array(band_power))
    
    # Add activity surface
    fig = create_activity_surface(fig, np.array(band_power))
    
    # Update layout
    fig.update_layout(
        title=f"{band.capitalize()} Band Activity",
        height=700
    )
    
    return fig

def create_3d_time_series(eeg_data, channel_indices=None):
    """
    Create a 3D visualization of EEG time series data.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    channel_indices : list
        Indices of channels to visualize (default: None, all channels)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        3D time series visualization
    """
    if channel_indices is None:
        channel_indices = range(eeg_data.shape[1])
    
    # Extract electrode positions
    positions = np.array([ELECTRODE_POSITIONS[CHANNEL_NAMES[i]] for i in channel_indices])
    
    # Create figure
    fig = go.Figure()
    
    # Time points
    time = np.arange(eeg_data.shape[0]) / 128  # Assuming 128 Hz
    
    # Scale factor for visualization
    scale = 0.2 / np.max(np.abs(eeg_data))
    
    # Add time series for each channel
    for i, ch_idx in enumerate(channel_indices):
        # Base position
        x0, y0, z0 = positions[i]
        
        # Create x, y, z coordinates for the time series
        x = x0 + scale * eeg_data[:, ch_idx]
        y = y0 + np.zeros_like(eeg_data[:, ch_idx])
        z = z0 + time / np.max(time) * 0.5  # Scale time axis
        
        # Add the time series
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(
                color=f'hsl({int(i * 360 / len(channel_indices))}, 70%, 50%)',
                width=2
            ),
            name=CHANNEL_NAMES[ch_idx]
        ))
        
        # Add electrode marker
        fig.add_trace(go.Scatter3d(
            x=[x0], y=[y0], z=[z0],
            mode='markers+text',
            marker=dict(
                size=8,
                color=f'hsl({int(i * 360 / len(channel_indices))}, 70%, 50%)'
            ),
            text=[CHANNEL_NAMES[ch_idx]],
            textposition='top center',
            name=f"{CHANNEL_NAMES[ch_idx]} (position)"
        ))
    
    # Update layout
    fig.update_layout(
        title="3D EEG Time Series",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Time (s)",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        width=800,
        height=700,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig