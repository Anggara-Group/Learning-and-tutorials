#Specific display streamlit functions
import matplotlib.pyplot as plt

def display_plot(created_image, title, raw_image=False):
    """
    Display or save a plot of the created image.
    
    Parameters:
    - created_image: 2D array of the image data
    - title: Title for the plot (ignored if raw_image=True)
    - raw_image: Boolean flag
        - True: Save raw image without colorbar, title, or axes
        - False: Display formatted plot with colorbar, title, and axes
    
    Returns:
    - fig: matplotlib figure object
    """
    if raw_image:
        # Create figure with no axes, titles, or colorbars
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(created_image, cmap='viridis', origin='lower')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
    else:
        # Original formatted plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(created_image, cmap='viridis', origin='lower')
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(im, ax=ax, label="Height (Å)")
    
    return fig