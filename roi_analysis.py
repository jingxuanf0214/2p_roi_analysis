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
import mat73
import matplotlib.image as mpimg
import os

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
base_path = 'C:/Users/wilson/OneDrive - Harvard University/Thesis - Wilson lab/2P imaging/scopa inspect/20231205_3/'
tif_name = 'data/20231205_3_1_4_3554_256_512_uint16__E_05_Iter_10368_output.tif'
tif_path = base_path + tif_name
template = select_template(tif_path, method='mip')  # Change method as needed

# Display the selected template
#plt.imshow(template, cmap='gray')
#plt.title('Selected Template')
#plt.show()

def extract_dimensions_and_plane(filename):
    """
    Extract the plane number, time steps, height, and width from the filename.
    
    Args:
    - filename (str): The filename to parse.
    
    Returns:
    - tuple: (plane_num, time_steps, height, width)
    """
    # Split the filename on underscores
    parts = filename.split('_')
    
    # Extract the relevant parts based on the known sequence
    plane_num = int(parts[3])
    time_steps = int(parts[4])
    height = int(parts[5])
    width = int(parts[6])
    
    return plane_num, time_steps, height, width

# Example usage
plane_num, time_steps, height, width = extract_dimensions_and_plane(tif_name)
print(f"Plane Number: {plane_num}, Time Steps: {time_steps}, Height: {height}, Width: {width}")
 

def load_and_convert_matlab_table(filepath):
    """
    Load a MATLAB table from a .mat file and convert it into a Python dictionary.
    
    Args:
    - filepath (str): Path to the .mat file containing the MATLAB table.
    
    Returns:
    - dict: A dictionary with ROI names as keys and a list of dictionaries (one for each subROI) as values.
            Each subROI dictionary contains 'plane' and 'masks' data.
    """
    # Load the .mat file
    try:
        data = scipy.io.loadmat(filepath)
    except NotImplementedError as e:
        # If scipy.io.loadmat fails due to version incompatibility, try with mat73.loadmat
        print(f"Loading with scipy.io failed: {e}. Trying with mat73.")
        data = mat73.loadmat(filepath)
    # Assuming the table is stored under the variable name 'tableData' in the .mat file
    struct = data['roiDefs']
    
    # Initialize a dictionary to hold the converted data
    roi_dict = {}
    
    # Iterate through the table rows
    for i in range(len(struct['name'])):
        roi_name = struct['name'][i]  # Assuming 'name' is the field for ROI names
        subrois = struct['subROIs'][i]  # Assuming 'subROIs' is the field for the struct array
        
        # Initialize a list to hold subROI data for this ROI
        subroi_list = []
        
        # Iterate through each subROI struct
        for j in range(len(subrois['plane'])):
            # Extract 'plane' and 'masks' fields from the struct
            plane = int(subrois['plane'][j])
            mask = subrois['mask'][j]
            
            # Append this subROI's data to the list
            subroi_list.append({'plane': plane, 'mask': mask})
        
        # Add the list of subROI data to the dictionary, keyed by the ROI name
        roi_dict[roi_name] = subroi_list
    
    return roi_dict

roi_dict = load_and_convert_matlab_table(base_path+'data/roiDefs_trial_001.mat')
#print(roi_dict)


def filter_and_save_rois(roi_dict, plane_num, output_path):
    """
    Filters ROIs by plane number, stacks matching ROI masks into a 3D array, and saves the result.
    
    Args:
    - roi_dict (dict): The dictionary containing ROI names and their subROIs with planes and masks.
    - plane_num (int): The plane number to filter ROIs by.
    - output_path (str): Path to save the numpy file containing the names and 3D mask array.
    """
    # Initialize list for storing ROI names and an empty list for collecting masks
    roi_names = []
    masks = []

    # Iterate through all ROIs in the data
    for roi_name, subrois in roi_dict.items():
        for subroi in subrois:
            if subroi['plane'] == plane_num:
                # If the current plane matches, add the mask to the masks list and store the ROI name
                masks.append(subroi['mask'])
                roi_names.append(roi_name)
                break  # Assuming only one match per ROI is needed
    
    # Stack masks along the z-dimension to create a 3D array
    if masks:
        masks_all = np.dstack(masks)
    else:
        # Return or handle the case where no masks match the specified plane
        print("No matching ROIs found for the specified plane.")
        return

    # Save the ROI names and 3D mask array to a file
    np.save(output_path, {'roi_names': roi_names, 'mask_all': masks_all})
    
    print(f"Saved {len(roi_names)} ROIs and their masks to {output_path}.")

save_path = base_path + 'results/roi_mask.npy'
filter_and_save_rois(roi_dict, plane_num, save_path)

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
        self.save_roi(base_path + 'results/roi_coordinates.txt')
        self.create_mask()

    def save_roi(self, filename):
        """Save the ROI coordinates to a text file."""
        np.savetxt(filename, np.column_stack([self.xs, self.ys]), header='x, y')

    def create_mask(self):
        """Create a binary mask from the ROI and save it as a .npy file."""
        mask = polygon2mask(self.template.shape, np.array(list(zip(self.ys, self.xs))))
        np.save(base_path + 'results/roi_mask.npy', mask)  # Save mask as .npy file

# TODO: continue working on this class
class EnhancedMultipleROIs:
    def __init__(self, ax, template, base_path):
        self.ax = ax
        self.template = template
        self.base_path = base_path  # Store base_path as an instance variable
        self.rois = []  # List of ROIs, each a list of (x, y) tuples
        self.polygons = []  # Polygon objects for visualization
        self.current_mode = 'freehand'  # Can be 'freehand' or 'subroi'
        self.primary_roi = None  # Store the primary ROI for sub-ROI drawing
        
        # Event connections
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.drawing = False
        
    def on_press(self, event):
        if self.current_mode == 'freehand':
            self.start_roi(event)
        elif self.current_mode == 'subroi' and self.primary_roi:
            self.start_subroi(event)

    def on_motion(self, event):
        if self.current_mode == 'freehand':
            self.update_roi(event)

    def on_release(self, event):
        if self.current_mode == 'freehand':
            self.finalize_roi()

    def on_key_press(self, event):
        if event.key == 'enter':
            self.finalize_all_rois()
        elif event.key == '1':
            self.current_mode = 'freehand'
            print("Switched to freehand drawing mode.")
        elif event.key == '2':
            self.current_mode = 'subroi'
            print("Switched to sub-ROI drawing mode. Draw the primary ROI first if not drawn.")

    def start_roi(self, event):
        # Initialize a new ROI or a primary ROI for sub-ROI mode
        if self.current_mode == 'freehand' or (self.current_mode == 'subroi' and not self.primary_roi):
            self.current_xs, self.current_ys = [event.xdata], [event.ydata]
            self.drawing = True

    def update_roi(self, event):
        # Update the current ROI with new points if in freehand mode
        if self.drawing and self.current_mode == 'freehand':
            self.current_xs.append(event.xdata)
            self.current_ys.append(event.ydata)
            self.ax.plot(self.current_xs, self.current_ys, 'r-')  # This dynamically draws; consider optimizing for performance
            self.ax.figure.canvas.draw()

    def finalize_roi(self):
        # Finalize the current ROI; for freehand, this closes the polygon and stores it
        if self.drawing and len(self.current_xs) > 2 and self.current_mode == 'freehand':
            # Close the ROI by connecting the last point to the first
            self.current_xs.append(self.current_xs[0])
            self.current_ys.append(self.current_ys[0])
            polygon = Polygon(np.column_stack([self.current_xs, self.current_ys]), closed=True, fill=False, color='r')
            self.ax.add_patch(polygon)
            self.polygons.append(polygon)
            
            # Store the ROI; if it's the first in sub-ROI mode, treat it as the primary ROI
            if self.current_mode == 'subroi' and not self.primary_roi:
                self.primary_roi = (self.current_xs, self.current_ys)
            else:
                self.rois.append((self.current_xs, self.current_ys))
            
            self.current_xs, self.current_ys = [], []  # Reset current points
            self.drawing = False

    def start_subroi(self, event):
        # Assuming a primary ROI exists, start defining sub-ROIs based on clicks on the boundary
        if self.current_mode == 'subroi' and self.primary_roi:
            # For simplicity, this example doesn't implement logic to ensure clicks are on the boundary.
            # You could add this by checking if the event coordinates are close to any segment of the primary_roi
            self.current_xs, self.current_ys = [event.xdata], [event.ydata]
            self.drawing = True  # This allows drawing within the primary ROI boundary

    def finalize_all_rois(self):
        # Finalizes all ROIs, creates masks, and disconnects event handlers
        self.finalize_roi()  # Ensure the current ROI is finalized
        
        # Create a mask for each ROI
        self.masks = [polygon2mask(self.template.shape, np.array(list(zip(ys, xs)))) for xs, ys in self.rois]
        
        # Optionally save masks to a file here
        
        # Disconnect event handlers to prevent further drawing
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        self.ax.figure.canvas.mpl_disconnect(self.cid_release)
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        self.ax.figure.canvas.mpl_disconnect(self.cid_key)
    
    def save_rois_and_masks(self):
        """
        Save the coordinates of ROIs and the generated masks.

        Parameters:
        - base_path: The base directory path where the files will be saved.
        """
        # Ensure base_path ends with a slash
        if not base_path.endswith('/'):
            base_path += '/'
        
        # Save ROI coordinates
        roi_coords_filename = base_path + 'roi_coordinates.txt'
        with open(roi_coords_filename, 'w') as f:
            for idx, (xs, ys) in enumerate(self.rois):
                coords = np.column_stack([xs, ys])
                np.savetxt(f, coords, header=f'ROI {idx}', comments='')
                f.write('\n\n')  # Separate ROIs by blank lines

        print(f'Saved ROI coordinates to {roi_coords_filename}')
        
        # Save masks
        masks_filename = self.base_path + 'results/roi_masks.npy'
        # Assuming self.masks is a list of 2D arrays, each representing a mask for an ROI
        if self.masks:
            masks_array = np.stack(self.masks, axis=-1)  # Stack masks along the third dimension
            np.save(masks_filename, masks_array)
            print(f'Saved ROI masks to {masks_filename}')
        else:
            print('No masks to save.')


def draw_freehand_roi(template):
    fig, ax = plt.subplots()
    ax.imshow(template, cmap='gray')
    roi_drawer = EnhancedMultipleROIs(ax, template,base_path)
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
#draw_freehand_roi(template)

# Example for saving .tif data as .npy
#tif_path = '20231205_3_1_5_3554_256_512_uint16__E_05_Iter_10368_output.tif'

#npy_path = base_path + 'results/denoised.npy'
#save_tif_as_npy(tif_path, npy_path)


def avg_by_roi(roi_mask, time_lapse_video, output_npy_path):
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
    #roi_mask = np.load(roi_mask_path)
    #time_lapse_video = np.load(time_lapse_video_path)
    
    # Ensure the mask is boolean for masking operations
    roi_mask = roi_mask.astype(bool)
    masked_frames = np.where(roi_mask[np.newaxis, :, :], time_lapse_video, 0)
    avg_raw = np.mean(masked_frames, axis=(1, 2))
    std_raw = np.std(masked_frames, axis=(1, 2))/np.sqrt(roi_mask.size)
    # Convert the list of masked frames to a numpy array and save as .npy file
    #np.save(output_npy_path, masked_frames)
    
    #print(f"Video saved to {output_video_path}")
    #print(f"Masked frames saved to {output_npy_path}")
    return std_raw, avg_raw

def avg_by_roi_loop(roi_mask_path, time_lapse_video_path):
    data = np.load(roi_mask_path, allow_pickle=True).item()
    roi_names = data['roi_names']
    roi_all = data['mask_all']
    std_all =[]
    avg_all = []
    time_lapse_video = np.load(time_lapse_video_path)
    for i in range(len(roi_names)):
        roi_mask = roi_all[:,:,i]
        output_npy_path = base_path + 'results/'+ roi_names[i] +'.npy'
        std_raw, avg_raw = avg_by_roi(roi_mask, time_lapse_video, output_npy_path)
        std_all.append(std_raw)
        avg_all.append(avg_raw)
    return std_all, avg_all

# Example usage
roi_mask_path = base_path + 'results/roi_mask.npy'
time_lapse_video_path = base_path + 'results/denoised.npy'
#std_all, avg_all = avg_by_roi_loop(roi_mask_path, time_lapse_video_path)
#output_npy_path = base_path + 'results/roi_extract.npy'
#std_raw, avg_raw = avg_by_roi(roi_mask_path, time_lapse_video_path, output_npy_path)

preprocessed_vars_ds = scipy.io.loadmat(base_path + 'data/preprocessed_vars_ds trial1.mat')
preprocessed_vars_odor = scipy.io.loadmat(base_path + 'data/preprocessed_vars_odor trial1.mat')

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

'''plt.figure(figsize=(10, 6))
plt.plot(avg_raw)
plt.fill_between(np.arange(len(avg_raw)),avg_raw-std_raw,avg_raw+std_raw, alpha=0.5)
plt.fill_between(np.arange(len(avg_raw)), 2, 4.3, where=np.array(fl_df.odor_mask), alpha=0.3)
plt.plot()
plt.xlabel('Frame')
plt.ylabel('avg raw fluorescence')
plt.title('avg raw fluorescence in ROI vs frames')

# Save the plot
plt.savefig('avg_raw.png')
plt.close()  # Close the plot explicitly after saving to free resources'''


def plot_avg_fluorescence_with_std(avg_values_list, std_values_list, odor_mask, num_cols=2):
    """
    Plots average fluorescence traces with standard deviation for multiple ROIs.
    
    Args:
    - avg_values_list (list of lists): List containing average fluorescence values for each ROI.
    - std_values_list (list of lists): List containing standard deviation values for each ROI.
    - odor_mask (np.array): Boolean array indicating specific frames (e.g., odor presentation).
    - num_rows (int): Number of rows in the subplot grid.
    - num_cols (int): Number of columns in the subplot grid. Default is 2.
    """
    num_rows = int(np.ceil(len(avg_values_list) / 2))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, num_rows*3))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for i, (avg_raw, std_raw) in enumerate(zip(avg_values_list, std_values_list)):
        ax = axes[i]
        frames = np.arange(len(avg_raw))
        
        # Plot average fluorescence
        ax.plot(frames, avg_raw, label='Average Fluorescence')
        
        # Plot standard deviation area
        ax.fill_between(frames, avg_raw-std_raw, avg_raw+std_raw, alpha=0.5, label='Std Dev')
        
        # Highlight specific regions (e.g., odor presentation)
        ax.fill_between(frames, min(avg_raw-std_raw), max(avg_raw+std_raw), 
                        where=odor_mask, color='gray', alpha=0.3, label='Odor Presentation')
        
        # Set titles and labels
        ax.set_title(f'ROI {i+1}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Avg raw fluorescence')
    
    # Add a legend in the first subplot
    axes[0].legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(base_path+'results/avg_raw_multiple_rois.png')
    plt.close(fig)  # Close the figure to free resources


#plot_avg_fluorescence_with_std(avg_all, std_all, fl_df.odor_mask, num_cols=2)

data = np.load(roi_mask_path, allow_pickle=True).item()
roi_names = data['roi_names']
roi_all = data['mask_all']
time_lapse_data = np.load(time_lapse_video_path)

def roi_crop(roi_mask):
    rows, cols = np.where(roi_mask)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    return min_row, max_row, min_col, max_col

def perform_pca_and_visualize_roi(time_lapse_data, roi_mask, save_path,n_components=3):
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
    #plt.show()
    plt.savefig(save_path +'/pca_map.png')
    return pca, pca_result

#pca_obj, pca_result = perform_pca_and_visualize_roi(time_lapse_data, roi_all[:,:,1], roi_names[1], n_components=3)

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

#cluster_labels = cluster_pixels_by_pca_components(pca_result, 2)

def visualize_cluster_heatmap(cluster_labels, roi_mask, save_path):
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
    #plt.show()
    plt.savefig(save_path + '/cluster_map.png')

#visualize_cluster_heatmap(cluster_labels, roi_all[:,:,1], roi_names[1])

def calculate_and_plot_cluster_statistics(time_lapse_data, roi_mask, save_path, cluster_labels, n_clusters):
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
    plt.figure(figsize=(15, 6))
    for cluster_idx in range(n_clusters):
        plt.fill_between(range(time_lapse_data.shape[0]), means[:, cluster_idx] - stderrs[:, cluster_idx], means[:, cluster_idx] + stderrs[:, cluster_idx], alpha=0.2)
        plt.plot(range(time_lapse_data.shape[0]), means[:, cluster_idx], label=f'Cluster {cluster_idx}')
    
    plt.xlabel('Frame')
    plt.ylabel('Average Pixel Value')
    plt.title('Average Pixel Value by Cluster Over Time with Standard Error')
    plt.legend()
    plt.savefig(save_path+'/avg_bycluster.png')
    plt.close()
    #print(f"Plot saved to {base_path}")

#calculate_and_plot_cluster_statistics(time_lapse_data, roi_all[:,:,1], roi_names[1], cluster_labels, 2)

# Path to your PNG images
    
def stitch_image(image_path1,image_path2,image_path3,save_path):
    # Load the images
    img1 = mpimg.imread(image_path1)
    img2 = mpimg.imread(image_path2)
    img3 = mpimg.imread(image_path3)
    stitched_img_1 = np.concatenate((img1, img2), axis=1)
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    # Display first image
    axs[0].imshow(stitched_img_1)
    axs[0].axis('off')  # Turn off axis

    # Display second image
    axs[1].imshow(img3)
    axs[1].axis('off')  # Turn off axis

    # Remove space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path+'/stitched_img.png')
    plt.close()

save_base = base_path + 'results/'
def summary_plot(time_lapse_data,roi_all,roi_names,query_idx,save_base):
    save_path = save_base + roi_names[query_idx]
    if not os.path.exists(save_path):
        # If the folder doesn't exist, create it
        os.makedirs(save_path)
        print(f"Folder created at: {save_path}")
    else:
        print(f"Folder already exists at: {save_path}")
    pca_obj, pca_result = perform_pca_and_visualize_roi(time_lapse_data, roi_all[:,:,query_idx], save_path, n_components=3)
    cluster_labels = cluster_pixels_by_pca_components(pca_result, 2)
    visualize_cluster_heatmap(cluster_labels, roi_all[:,:,query_idx], save_path)
    calculate_and_plot_cluster_statistics(time_lapse_data, roi_all[:,:,query_idx], save_path, cluster_labels, 2)
    image_path1 = save_path +'/pca_map.png'
    image_path2 = save_path +'/cluster_map.png'
    image_path3 = save_path+'/avg_bycluster.png'
    stitch_image(image_path1,image_path2,image_path3,save_path)
for i in range(len(roi_names)):
    summary_plot(time_lapse_data,roi_all,roi_names,i,save_base)


############################################################################################################

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

#plot_pca_components_3d_with_clusters(pca_result, cluster_labels)

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

#plot_component_contributions(pca_obj)