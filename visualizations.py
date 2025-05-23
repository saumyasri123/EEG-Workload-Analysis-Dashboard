import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

def plot_raw_eeg(eeg_data, sampling_rate=128, channels=None, title="Raw EEG Signal"):
    """
    Plot raw EEG signals.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    sampling_rate : int
        Sampling rate in Hz
    channels : list
        List of channel names (default: None)
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if channels is None:
        channels = [f'Channel {i+1}' for i in range(eeg_data.shape[1])]
    
    # Create time vector
    time = np.arange(eeg_data.shape[0]) / sampling_rate
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each channel (offset for clarity)
    for i in range(min(eeg_data.shape[1], 14)):  # Limit to 14 channels max
        offset = i * 50  # Add offset for visualization
        fig.add_trace(
            go.Scatter(
                x=time,
                y=eeg_data[:, i] + offset,
                name=channels[i],
                line=dict(width=1)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        height=500
    )
    
    return fig

def plot_band_power_heatmap(band_power_matrix, band_names, channel_names=None):
    """
    Plot band power heatmap.
    
    Parameters:
    -----------
    band_power_matrix : numpy.ndarray
        Matrix of band powers with shape (channels, bands)
    band_names : list
        List of band names
    channel_names : list
        List of channel names (default: None)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if channel_names is None:
        channel_names = [f'Channel {i+1}' for i in range(band_power_matrix.shape[0])]
    
    # Create heatmap
    fig = px.imshow(
        band_power_matrix,
        x=band_names,
        y=channel_names,
        color_continuous_scale='viridis',
        aspect='auto'
    )
    
    # Update layout
    fig.update_layout(
        title="EEG Band Power Heatmap",
        xaxis_title="Frequency Bands",
        yaxis_title="EEG Channels",
        coloraxis_colorbar=dict(title="Power"),
        height=500
    )
    
    return fig

def plot_fft_spectrum(eeg_data, sampling_rate=128, channel_idx=0, channel_name=None):
    """
    Plot FFT spectrum for a single EEG channel.
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (samples, channels)
    sampling_rate : int
        Sampling rate in Hz
    channel_idx : int
        Index of the channel to plot (default: 0)
    channel_name : str
        Name of the channel (default: None)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if channel_name is None:
        channel_name = f'Channel {channel_idx+1}'
    
    # Compute FFT
    n = eeg_data.shape[0]
    signal = eeg_data[:, channel_idx]
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(n, 1.0/sampling_rate)
    
    # Plot only frequencies up to 50 Hz
    mask = fft_freq <= 50
    
    # Create figure
    fig = go.Figure()
    
    # Add trace
    fig.add_trace(
        go.Scatter(
            x=fft_freq[mask],
            y=fft_vals[mask],
            mode='lines',
            line=dict(width=2),
            name=channel_name
        )
    )
    
    # Add frequency band regions
    bands = [
        {'name': 'Delta', 'range': (1, 4), 'color': 'rgba(255, 0, 0, 0.1)'},
        {'name': 'Theta', 'range': (4, 8), 'color': 'rgba(0, 255, 0, 0.1)'},
        {'name': 'Alpha', 'range': (8, 13), 'color': 'rgba(0, 0, 255, 0.1)'},
        {'name': 'Beta', 'range': (13, 30), 'color': 'rgba(255, 0, 255, 0.1)'}
    ]
    
    for band in bands:
        fig.add_vrect(
            x0=band['range'][0],
            x1=band['range'][1],
            fillcolor=band['color'],
            layer="below",
            line_width=0,
            annotation_text=band['name'],
            annotation_position="top left"
        )
    
    # Update layout
    fig.update_layout(
        title=f"FFT Spectrum - {channel_name}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=400
    )
    
    return fig

def plot_training_history(history):
    """
    Plot model training history.
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        Training history object
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
    
    # Add loss traces
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name="Training Loss", line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name="Validation Loss", line=dict(color='red')),
        row=1, col=1
    )
    
    # Add accuracy traces
    fig.add_trace(
        go.Scatter(y=history.history['accuracy'], name="Training Accuracy", line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_accuracy'], name="Validation Accuracy", line=dict(color='red')),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="Model Training History",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    # Update axes
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    return fig

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix with shape (n_classes, n_classes)
    class_names : list
        List of class names (default: None)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if class_names is None:
        class_names = ['Low Workload', 'High Workload']
    
    # Create figure
    fig = px.imshow(
        cm,
        x=class_names,
        y=class_names,
        color_continuous_scale='blues',
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        text_auto=True
    )
    
    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        height=400
    )
    
    return fig

def plot_roc_curve(fpr, tpr, auc_value):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    fpr : numpy.ndarray
        False positive rates
    tpr : numpy.ndarray
        True positive rates
    auc_value : float
        Area under the ROC curve
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_value:.3f})',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99
        ),
        height=400
    )
    
    return fig

def plot_subject_ratings(ratings, title="Subject Workload Ratings"):
    """
    Plot subject workload ratings.
    
    Parameters:
    -----------
    ratings : dict
        Dictionary mapping subject IDs to their ratings
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    subjects = []
    rest_ratings = []
    test_ratings = []
    
    for subject, rating in ratings.items():
        subjects.append(subject)
        rest_ratings.append(rating['rest'])
        test_ratings.append(rating['test'])
    
    # Create subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar for rest ratings
    fig.add_trace(
        go.Bar(
            x=subjects,
            y=rest_ratings,
            name="Rest Ratings",
            marker_color='blue',
            opacity=0.7
        )
    )
    
    # Add bar for test ratings
    fig.add_trace(
        go.Bar(
            x=subjects,
            y=test_ratings,
            name="Test Ratings",
            marker_color='orange',
            opacity=0.7
        )
    )
    
    # Add line for rating difference
    fig.add_trace(
        go.Scatter(
            x=subjects,
            y=[test - rest for test, rest in zip(test_ratings, rest_ratings)],
            name="Difference",
            line=dict(color='green', width=2),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Subject ID",
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Rating (1-9)", secondary_y=False)
    fig.update_yaxes(title_text="Difference", secondary_y=True)
    
    return fig
