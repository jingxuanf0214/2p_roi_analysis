import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.draw import polygon2mask  # Requires skimage, install it if necessary
import imageio
import scipy
import scipy.io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# anatomy check 

def select_template(tif_path, method='mip'):
    """
    Selects a template from a .tif stack using the specified method.
    
    Parameters:
    - tif_path: Path to the .tif file.
    - method: Method to use for template selection ('mip', 'average', 'median', or 'specific').
    - frame_number: Specific frame number to use as a template if method is 'specific'.
    """
    with tiff.TiffFile(tif_path) as tif:
        stack = tif.asarray()  # Assuming the stack is in the shape (time, height, width)
        
    if method == 'mip':
        template = np.max(stack, axis=0)
    elif method == 'average':
        template = np.mean(stack, axis=0)
    elif method == 'median':
        template = np.median(stack, axis=0)
    elif method == 'specific':
        frame_number = int(input("Enter the frame number to use as a template: "))
        template = stack[frame_number]
    else:
        raise ValueError("Invalid method selected.")
    
    return template

# Example usage
tif_path = '20231205_3_1_5_3554_256_512_uint16__E_05_Iter_10368_output.tif'
template = select_template(tif_path, method='mip')  # Change method as needed

# Display the selected template
#plt.imshow(template, cmap='gray')
#plt.title('Selected Template')
#plt.show()

class FreehandROI:
    def __init__(self, ax, template):
        self.ax = ax
        self.template = template
        self.xs = []  # Store x coordinates of the ROI
        self.ys = []  # Store y coordinates of the ROI
        self.line, = ax.plot([], [], 'r-')  # Empty line object for the ROI
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.drawing = False

    def on_press(self, event):
        self.drawing = True
        self.xs = [event.xdata]
        self.ys = [event.ydata]

    def on_motion(self, event):
        if self.drawing:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.ax.figure.canvas.draw()

    def on_release(self, event):
        self.drawing = False
        # Close the ROI by connecting the last point to the first
        if len(self.xs) > 1 and (self.xs[0] != self.xs[-1] or self.ys[0] != self.ys[-1]):
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
        self.line.set_data(self.xs, self.ys)
        self.ax.figure.canvas.draw()
        # Save the drawing and create mask
        self.save_roi('roi_coordinates.txt')
        self.create_mask()

    def save_roi(self, filename):
        """Save the ROI coordinates to a text file."""
        np.savetxt(filename, np.column_stack([self.xs, self.ys]), header='x, y')

    def create_mask(self):
        """Create a binary mask from the ROI and save it as a .npy file."""
        mask = polygon2mask(self.template.shape, np.array(list(zip(self.ys, self.xs))))
        np.save('roi_mask.npy', mask)  # Save mask as .npy file

def draw_freehand_roi(template):
    fig, ax = plt.subplots()
    ax.imshow(template, cmap='gray')
    roi_drawer = FreehandROI(ax, template)
    plt.show()

# Function to save .tif time-lapse data as .npy
def save_tif_as_npy(tif_path, npy_path):
    """
    Load a .tif file and save the data as a .npy file.
    
    Parameters:
    - tif_path: Path to the .tif file.
    - npy_path: Path where the .npy file will be saved.
    """
    with tiff.TiffFile(tif_path) as tif:
        data = tif.asarray()
    np.save(npy_path, data)

# Assuming 'template' is your selected template image
draw_freehand_roi(template)

# Example for saving .tif data as .npy
tif_path = '20231205_3_1_5_3554_256_512_uint16__E_05_Iter_10368_output.tif'
npy_path = 'denoised.npy'
save_tif_as_npy(tif_path, npy_path)


def create_time_series_video_and_save_npy(roi_mask_path, time_lapse_video_path, output_npy_path):
    """
    Creates a time-series video by applying an ROI mask to each frame of the input video,
    saves it to the specified output path, and also saves the masked frames as a .npy file.
    
    Parameters:
    - roi_mask_path: Path to the .npy file containing the ROI mask.
    - time_lapse_video_path: Path to the .npy file containing the time-lapse video data.
    - output_video_path: Path where the output video will be saved.
    - output_npy_path: Path where the masked frames will be saved as a .npy file.
    """
    # Load the ROI mask and time-lapse video data
    roi_mask = np.load(roi_mask_path)
    time_lapse_video = np.load(time_lapse_video_path)
    
    # Ensure the mask is boolean for masking operations
    roi_mask = roi_mask.astype(bool)
    masked_frames = np.where(roi_mask[np.newaxis, :, :], time_lapse_video, 0)
    avg_raw = np.mean(masked_frames, axis=(1, 2))
    std_raw = np.std(masked_frames, axis=(1, 2))/np.sqrt(roi_mask.size)
    # Convert the list of masked frames to a numpy array and save as .npy file
    np.save(output_npy_path, masked_frames)
    
    #print(f"Video saved to {output_video_path}")
    print(f"Masked frames saved to {output_npy_path}")
    return std_raw, avg_raw

# Example usage
roi_mask_path = 'roi_mask.npy'
time_lapse_video_path = 'denoised.npy'
output_npy_path = 'roi_extract.npy'
std_raw, avg_raw = create_time_series_video_and_save_npy(roi_mask_path, time_lapse_video_path, output_npy_path)

preprocessed_vars_ds = scipy.io.loadmat('preprocessed_vars_ds_trial1.mat')
preprocessed_vars_odor = scipy.io.loadmat('preprocessed_vars_odor_trial1.mat')

def make_bh_df(preprocessed_vars_ds,preprocessed_vars_odor):
    fl_df = pd.DataFrame()
    fl_df['fwV'] = np.squeeze(preprocessed_vars_ds['ftT_fwSpeedDown2'])
    fl_df['sideV'] = np.squeeze(preprocessed_vars_ds['ftT_sideSpeedDown2'])
    fl_df['yawV'] = np.squeeze(preprocessed_vars_ds['ftT_yawSpeedDown2'])
    fl_df['heading'] = np.squeeze(preprocessed_vars_ds['ftT_intHDDown2'])
    fl_df['abssideV'] = np.abs(np.squeeze(preprocessed_vars_ds['ftT_sideSpeedDown2']))
    fl_df['absyawV'] = np.abs(np.squeeze(preprocessed_vars_ds['ftT_yawSpeedDown2']))
    fl_df['net_motion'] = fl_df['abssideV']+fl_df['absyawV']+np.abs(fl_df['fwV'])
    fl_df['net_motion_state'] = (fl_df['net_motion']>2.5).astype(int)
    fl_df['heading_adj'] = np.unwrap(fl_df['heading'])
    odor_all = preprocessed_vars_odor['odorDown']
    fl_df['odor'] = np.squeeze(odor_all)
    fl_df['odor_mask'] = (fl_df['odor'] > 5)
    return fl_df

fl_df = make_bh_df(preprocessed_vars_ds,preprocessed_vars_odor)
#print(fl_df.head(5))

plt.figure(figsize=(10, 6))
plt.plot(avg_raw)
plt.fill_between(np.arange(len(avg_raw)),avg_raw-std_raw,avg_raw+std_raw, alpha=0.5)
plt.fill_between(np.arange(len(avg_raw)), 2, 4.3, where=np.array(fl_df.odor_mask), alpha=0.3)
plt.plot()
plt.xlabel('Frame')
plt.ylabel('avg raw fluorescence')
plt.title('avg raw fluorescence in ROI vs frames')

# Save the plot
plt.savefig('avg_raw.png')
plt.close()  # Close the plot explicitly after saving to free resources

roi_mask = np.load(roi_mask_path)
time_lapse_data = np.load(time_lapse_video_path)

def roi_crop(roi_mask):
    rows, cols = np.where(roi_mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    return min_row, max_row, min_col, max_col

def perform_pca_and_visualize_roi(time_lapse_data, roi_mask, n_components=2):
    """
    Perform PCA on pixel values within an ROI across all frames and visualize the
    top principal components as heatmaps in their original spatial arrangement.
    
    Parameters:
    - time_lapse_data: A 3D numpy array of shape (frames, height, width) containing the time-lapse video data.
    - roi_mask: A 2D numpy array of shape (height, width) indicating the ROI, with True/1 for pixels within the ROI.
    - n_components: Number of principal components to retain.
    """

    # Flatten the spatial dimensions and filter by the ROI mask
    pixels_in_roi = time_lapse_data[:, roi_mask].T  # Shape: (N_pixels, T_frames)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(pixels_in_roi)
    
    pca_component_maps = np.full((n_components, *roi_mask.shape), np.nan)  # Use NaN for areas outside the ROI
    
    min_row, max_row, min_col, max_col = roi_crop(roi_mask)
    # Find the bounding box of the ROI to crop the visualization
    
    for i in range(n_components):
        component_map = np.full(roi_mask.shape, np.nan)  # Initialize with NaNs
        component_map[roi_mask] = pca_result[:, i]  # Map component values to their original locations
        pca_component_maps[i] = component_map  # Store in the corresponding layer

    # Creating subplots for each principal component
    fig, axes = plt.subplots(1, n_components, figsize=(n_components * 5, 5))
    
    for i in range(n_components):
        ax = axes[i] if n_components > 1 else axes
        cropped_pca_map = pca_component_maps[i, min_row:max_row+1, min_col:max_col+1]
        im = ax.imshow(cropped_pca_map, cmap='viridis', aspect='auto')
        
        # Overlay gray color for areas outside the ROI
        overlay_mask = ~np.isnan(cropped_pca_map)
        ax.imshow(np.where(overlay_mask, np.nan, 1), cmap='gray', aspect='auto', alpha=0.5)
        
        ax.set_title(f'PCA Component {i+1}')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    plt.savefig('pca_map')
    return pca, pca_result

pca_obj, pca_result = perform_pca_and_visualize_roi(time_lapse_data, roi_mask, n_components=3)

def cluster_pixels_by_pca_components(pca_result, n_clusters):
    """
    Cluster pixels within the ROI based on their top 3 PCA component values using K-means.
    
    Parameters:
    - pca_results: A 2D numpy array of shape (N_pixels, 3), where N_pixels is the number of pixels
      within the ROI and each row contains the top 3 PCA component values for a pixel.
    - n_clusters: The number of clusters to form.
    
    Returns:
    - cluster_labels: A numpy array of shape (N_pixels,) containing the cluster label for each pixel.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pca_result)
    cluster_labels = kmeans.labels_
    return cluster_labels

cluster_labels = cluster_pixels_by_pca_components(pca_result, 2)

def visualize_cluster_heatmap(cluster_labels, roi_mask):
    """
    Visualize the cluster labels as a heatmap on the ROI mask, with pixels outside the ROI in gray and cropped to fit the ROI.
    
    Parameters:
    - cluster_labels: A 1D numpy array containing the cluster label for each pixel within the ROI.
    - roi_mask: A 2D numpy array of the same shape as the original image, indicating the ROI with True/1.
    - min_row, max_row, min_col, max_col: The bounding box coordinates of the ROI.
    """
    # mi max bound
    min_row, max_row, min_col, max_col = roi_crop(roi_mask)

    # Initialize an array to store the cluster labels in their original spatial positions
    cluster_map = np.full(roi_mask.shape, np.nan)  # Use NaN for areas outside the ROI
    
    # Map the cluster labels back to their original spatial positions
    cluster_map[roi_mask] = cluster_labels
    
    # Crop the cluster_map to focus on the ROI
    cropped_cluster_map = cluster_map[min_row:max_row+1, min_col:max_col+1]
    
    # Plotting
    plt.figure(figsize=(5, 5))
    plt.imshow(cropped_cluster_map, cmap='viridis', aspect='auto')
    
    # Overlay gray color for areas outside the ROI
    overlay_mask = ~np.isnan(cropped_cluster_map)
    plt.imshow(np.where(overlay_mask, np.nan, 1), cmap='gray', aspect='auto', alpha=0.5)
    
    plt.colorbar()
    plt.title('Cluster Labels Heatmap')
    plt.show()

visualize_cluster_heatmap(cluster_labels, roi_mask)

def calculate_and_plot_cluster_statistics(time_lapse_data, roi_mask, cluster_labels, n_clusters):
    """
    Calculate the average and standard error of pixel values by cluster on each frame, 
    and plot the time series curves for each cluster.
    
    Parameters:
    - time_lapse_data: A 3D numpy array of shape (frames, height, width) containing the time-lapse video data.
    - roi_mask: A 2D numpy array of shape (height, width) indicating the ROI, with True/1 for pixels within the ROI.
    - cluster_labels: A 1D numpy array containing the cluster label for each pixel within the ROI.
    - n_clusters: The number of clusters.
    """
    # Reshape cluster_labels to match the spatial dimensions of roi_mask
    cluster_label_map = np.full(roi_mask.shape, -1)  # Initialize with -1 for pixels outside ROI
    cluster_label_map[roi_mask] = cluster_labels
    
    # Initialize arrays to store mean and standard error for each cluster
    means = np.zeros((time_lapse_data.shape[0], n_clusters))
    stderrs = np.zeros((time_lapse_data.shape[0], n_clusters))
    
    for cluster_idx in range(n_clusters):
        # Create a boolean mask for the current cluster across all frames
        cluster_mask = (cluster_label_map == cluster_idx)[np.newaxis, :, :]
        
        # Broadcast and select pixels for the current cluster across all frames
        cluster_pixels = np.where(cluster_mask, time_lapse_data, 0)
        
        # Calculate mean and standard error, ignoring NaNs
        means[:, cluster_idx] = np.mean(cluster_pixels, axis=(1, 2))
        stderrs[:, cluster_idx] = np.std(cluster_pixels, axis=(1, 2)) / np.sqrt(np.count_nonzero(~np.isnan(cluster_pixels), axis=(1, 2)))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for cluster_idx in range(n_clusters):
        plt.fill_between(range(time_lapse_data.shape[0]), means[:, cluster_idx] - stderrs[:, cluster_idx], means[:, cluster_idx] + stderrs[:, cluster_idx], alpha=0.2)
        plt.plot(range(time_lapse_data.shape[0]), means[:, cluster_idx], label=f'Cluster {cluster_idx}')
    
    plt.xlabel('Frame')
    plt.ylabel('Average Pixel Value')
    plt.title('Average Pixel Value by Cluster Over Time with Standard Error')
    plt.legend()
    plt.savefig('avg_bycluster')
    plt.close()
    #print(f"Plot saved to {save_path}")

calculate_and_plot_cluster_statistics(time_lapse_data, roi_mask, cluster_labels, 2)

'''def plot_frame_pixels_in_pc_space(pca, time_lapse_data, roi_mask, frame_index):
    """
    Plot the pixels of a selected frame in the PC1 and PC2 space based on earlier PCA results.
    
    Parameters:
    - pca: The PCA object from scikit-learn that has been fit to the pixel data.
    - time_lapse_data: A 3D numpy array of shape (frames, height, width) containing the time-lapse video data.
    - roi_mask: A 2D numpy array of shape (height, width) indicating the ROI, with True/1 for pixels within the ROI.
    - frame_index: The index of the frame to be plotted.
    """
    # Extract pixel values for the selected frame within the ROI
    selected_frame_data = time_lapse_data[frame_index]
    pixels_in_roi = selected_frame_data[roi_mask].reshape(-1, 1)
    
    # Project the pixel values of the selected frame onto the PC1 and PC2 space
    # Note: Ensure pca was fit with the pixels reshaped properly and matches this reshaping
    pca_projection = pca.transform(pixels_in_roi)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_projection[:, 0], pca_projection[:, 1], alpha=0.6, edgecolor='w', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Frame {frame_index} Pixels in PC1-PC2 Space')
    plt.grid(True)
    plt.show()
plot_frame_pixels_in_pc_space(pca_obj, time_lapse_data, roi_mask, 0)'''

def plot_pca_components_3d_with_clusters(pca_result, cluster_labels):
    """
    Plot the PCA projection of the pixels in the ROI onto the first three principal components in a 3D scatter plot,
    colored by K-means cluster labels.
    
    Parameters:
    - pca_result: The result of PCA transformation, shape (N_pixels, at least 3), with the first three columns
      corresponding to the first three principal components.
    - cluster_labels: A 1D numpy array containing the K-means cluster label for each pixel within the ROI.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate a color map
    cmap = plt.get_cmap("tab10")  # Adjust the number of colors based on the number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    for cluster in range(n_clusters):
        # Select data points that belong to the current cluster
        idx = cluster_labels == cluster
        ax.scatter(pca_result[idx, 0], pca_result[idx, 1], pca_result[idx, 2], 
                   s=50, alpha=0.6, edgecolor='w', label=f'Cluster {cluster}', 
                   color=cmap(cluster))
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title('Pixels in PCA Component Space (3D) by Cluster')
    ax.legend()
    plt.show()

plot_pca_components_3d_with_clusters(pca_result, cluster_labels)

def plot_component_contributions(pca):
    """
    Plot the contribution (loadings) of the first three PCA components across all features (time frames).
    
    Parameters:
    - pca: PCA object from scikit-learn after fitting to the dataset.
    """
    components = pca.components_[:3]  # Get the first three components
    time_frames = np.arange(components.shape[1])  # Assuming features correspond to time frames
    
    plt.figure(figsize=(12, 6))
    for i, component in enumerate(components):
        plt.plot(time_frames, component, label=f'PC{i+1}')
        
    plt.xlabel('Time Frame')
    plt.ylabel('Component Weight')
    plt.title('Contribution of First Three PCA Components Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_component_contributions(pca_obj)