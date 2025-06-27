import numpy as np
from scipy.stats import entropy
import numba
from numba import njit, prange
import numba.typed
import numba.types
'''
some functions to calculate entropy and flow direction
outside of class to avoid circular imports
'''

def calculate_window_entropy(window):
    """Calculate entropy for a given window."""
    _, counts = np.unique(window, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities)

# Move direction_encoding to module level
direction_encoding = np.array([[32, 64, 128],
                               [16, 0, 1],
                               [8, 4, 2]])

def flow_direction_window(window):
    """Process a 3x3 window to calculate flow direction."""
    # Reshape the 1D window to 3x3
    window = window.reshape((3, 3))
    if np.isnan(window).all():
        return 0  # No flow direction if all values are NaN or -9999
    diff = window[1, 1] - window  # Differences between center and neighbors
    max_diff = np.max(diff)
    return direction_encoding[diff == max_diff].sum() if max_diff > 0 else 0

def fast_entropy(arr):
    """Fast entropy calculation for a 1D array."""
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-12))

# Remove parallel=True for GIL safety
@njit
def fast_entropy_numba(windows):
    """Numba-accelerated entropy for 3D windowed array."""
    out = np.empty((windows.shape[0], windows.shape[1]), dtype=np.float32)
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            window = windows[i, j].ravel()
            # Numba-compatible unique/counts using typed.Dict
            unique = numba.typed.Dict.empty(
                key_type=numba.types.float32,
                value_type=numba.types.int64
            )
            for val in window:
                if np.isnan(val):
                    continue
                v = np.float32(val)
                if v in unique:
                    unique[v] += 1
                else:
                    unique[v] = 1
            total = 0
            for count in unique.values():
                total += count
            ent = 0.0
            for count in unique.values():
                p = count / total
                ent -= p * np.log2(p + 1e-12)
            out[i, j] = ent
    return out

## # ShannonIndex class to calculate Shannon diversity index
## Using python's built-in functions for simplicity and clarity.
'''
functions to calculate entropy and flow direction
outside of class to avoid circular imports
'''

# def calculate_window_entropy(window):
#     """Calculate entropy for a given window."""
#     _, counts = np.unique(window, return_counts=True)
#     probabilities = counts / counts.sum()
#     return entropy(probabilities)

# def flow_direction_window(window):
#     # Define the direction encoding
#     direction_encoding = np.array([[32, 64, 128],
#                                     [16, 0, 1],
#                                     [8, 4, 2]])
#     """Process a 3x3 window to calculate flow direction."""
#     if np.isnan(window).all():
#         return 0  # No flow direction if all values are NaN or -9999
#     diff = window[1, 1] - window  # Differences between center and neighbors
#     max_diff = np.max(diff)
#     return direction_encoding[diff == max_diff].sum() if max_diff > 0 else 0

   
    # def calculate_flow_direction(self, dem_data):
    #     """
    #     Calculate the Flow Direction similar to ArcGIS (Spatial Analyst).
    #     https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-flow-direction-works.htm
    #     """

    #     # Initialize the flow direction array
    #     flow_direction = np.zeros(dem_data.shape, dtype=np.int32)

    #     # Apply the sliding window approach
    #     flow_direction[1:-1, 1:-1] = np.array([
    #         flow_direction_window(dem_data[i - 1:i + 2, j - 1:j + 2])
    #         for i in range(1, dem_data.shape[0] - 1)
    #         for j in range(1, dem_data.shape[1] - 1)
    #     ]).reshape(dem_data.shape[0] - 2, dem_data.shape[1] - 2)

    #     return flow_direction

    # def calculate_shannon_index_2d(self, dem_data):
    #     """
    #     Calculate the Shannon diversity index for each window in a 2D grid.
    #     """
    #     window_size = self.shannon_window
    #     flow_direction = self.calculate_flow_direction(dem_data)
    #     rows, cols = flow_direction.shape
    #     shannon_indices = np.zeros((rows - window_size + 1, cols - window_size + 1))

    #     def _serial_entropy():
    #         return [
    #             calculate_window_entropy(
    #                 flow_direction[i:i + window_size, j:j + window_size]
    #             )
    #             for i in range(rows - window_size + 1)
    #             for j in range(cols - window_size + 1)
    #         ]

    #     results = _serial_entropy()

    #     for idx, value in enumerate(results):
    #         i = idx // (cols - window_size + 1)
    #         j = idx % (cols - window_size + 1)
    #         shannon_indices[i, j] = value

    #     return shannon_indices

## Using the ndimage.generic_filter for a more efficient implementation

    # def calculate_flow_direction(self, dem_data):
    #     """
    #     Calculate the Flow Direction similar to ArcGIS (Spatial Analyst).
    #     https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-flow-direction-works.htm
    #     """

    #     # Use ndimage.generic_filter to apply flow_direction_window over the DEM
    #     flow_direction = ndimage.generic_filter(
    #         dem_data,
    #         function=flow_direction_window,
    #         size=3,
    #         mode='nearest'
    #     ).astype(np.int32)

    #     return flow_direction

    # def calculate_shannon_index_2d(self, dem_data):
        #     """
        #     Calculate the Shannon diversity index for each window in a 2D grid.
        #     Output shape matches input DEM, with borders handled by padding.
        #     """
        #     window_size = self.shannon_window
        #     flow_direction = self.calculate_flow_direction(dem_data)

        #     def _window_entropy(window):
        #         return calculate_window_entropy(window)

        #     shannon_indices = ndimage.generic_filter(
        #         flow_direction,
        #         _window_entropy,
        #         size=window_size,
        #         mode='nearest'
        #     )
        #     return shannon_indices