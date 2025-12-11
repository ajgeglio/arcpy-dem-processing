import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import json
import os

# Ensure your local modules are imported
from landforms import Landforms 
from utils import Utils
from gdalUtils import GdalUtils
from rasterUtils import RasterUtils 
from derivatives import HabitatDerivatives
from shannon import ShannonDerivatives

class PlotDEM:
    def __init__(self, dem_path):
        self.dem_path = dem_path

        ## creating DEM data array object
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
            self.tags = src.tags()
            self.metadata = src.meta
            self.nodata = src.nodata
            
        # Handle Nodata (Robustly check if nodata exists in meta)
        if self.nodata is not None:
            # Create a masked array or fill with NaN
            # If using float DEMs, standard practice is np.nan
            if np.issubdtype(dem_data.dtype, np.floating):
                dem_data[dem_data == self.nodata] = np.nan
            else:
                # For integer DEMs, keep as is or use a mask
                pass
        
        self.dem_data = dem_data
        
        self.hd = HabitatDerivatives(self.dem_path, self.dem_data, transform=self.transform)
        self.shannon_tool = ShannonDerivatives(self.dem_data, self.dem_path)

    def __repr__(self):
        return f"PlotDEM(dem_path={self.dem_path})"

    def plot_dem(self, title="Raster Heat Map", cmap="terrain", vmin=-100, vmax=100, save_fig=True, show=True):
        """
        Displays the DEM as a heatmap with a color bar and saves it.

        Parameters:
        output_heatmap_path (str): Path to save the DEM heatmap. If None, does not save.
        show (bool): Whether to display the plot interactively.
        """
        # Load DEM using Rasterio
        dem_array = self.dem_data
        # Create a custom colormap with transparency for zero values
        cmap = plt.cm.terrain
        # cmap = plt.cm.terrain_r  # Reverse the colormap
        # cmap = plt.cm.jet
        cmap.set_under(color='none')  # Set color for values below the minimum (zero)

        # Plot DEM heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(dem_array, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax, alpha=0.7)  # DEM background
        plt.colorbar(heatmap, ax=ax, label="Elevation (m)")
        ax.set_title(title)
        ax.axis("off")
        if save_fig:
            output_heatmap_path= f"{self.dem_path.split('.')[0]}_heatmap.png"
            plt.savefig(output_heatmap_path, dpi=600, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        del fig, ax, heatmap

    def plot_lbp_neighbors(self, n_points=8, radius=1, show=True, output_heatmap_path=None):
        """
        Visualizes the LBP neighbor locations around a central pixel.

        Parameters:
        n_points (int): Number of neighboring points.
        radius (int): Distance from the central pixel.
        show (bool): Whether to display the plot interactively.
        output_heatmap_path (str): Path to save the plot. If None, does not save.
        """
        # Compute neighbor coordinates based on circular pattern
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x_coords = np.round(radius * np.cos(angles)).astype(int)
        y_coords = np.round(radius * np.sin(angles)).astype(int)

        # Plotting setup
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlim(-radius-1, radius+1)
        ax.set_ylim(-radius-1, radius+1)
        ax.set_xticks(range(-radius-1, radius+2))
        ax.set_yticks(range(-radius-1, radius+2))
        ax.grid(True)

        # Plot center pixel
        ax.scatter(0, 0, color='red', s=200, label="Center Pixel (x)")

        # Plot LBP neighbor pixels
        for i in range(n_points):
            ax.scatter(x_coords[i], y_coords[i], color='blue', s=100)
            ax.text(x_coords[i], y_coords[i], str(i), fontsize=12, ha='center', va='center', color="white")

        ax.set_title(f"LBP Neighbors (n_points={n_points}, radius={radius})")
        ax.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        if output_heatmap_path:
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def generate_lbp_heatmap(self, n_points=24, radius=3, method='uniform', save_fig=False, show=True):
        """
        Visualizes LBP. Uses a discrete colormap if method is 'uniform'.
        """
        lbp_array = self.hd.calculate_lbp(n_points, radius, method=method)
        
        # 1. Mask the Background
        # Mask NaNs so they appear white
        lbp_masked = np.ma.masked_invalid(lbp_array)

        fig, ax = plt.subplots(figsize=(12, 10))

        # 2. Configure Colormap based on method
        if method == 'uniform' or method == 'nri_uniform':
            # Uniform LBP produces a limited number of integer codes (0 to P+1)
            # We treat this as categorical/discrete data.
            # n_points + 2 is the max number of unique labels in uniform LBP
            num_labels = n_points + 2
            cmap = plt.cm.get_cmap('tab20c', num_labels) # Discrete colors
            cmap.set_bad(color='white')
            
            heatmap = ax.imshow(lbp_masked, cmap=cmap, interpolation='nearest')
            
            # Discrete Colorbar
            cbar = plt.colorbar(heatmap, ax=ax, ticks=range(num_labels), label="LBP Pattern Class")
            cbar.ax.set_yticklabels([str(i) for i in range(num_labels)])
            
        else:
            # Default/Rotation Invariant LBP produces 0-255 (or higher)
            # Use grayscale or plasma to show "roughness/texture intensity"
            cmap = plt.cm.inferno
            cmap.set_bad(color='white')
            
            # Robust scaling to ignore outliers
            vmin, vmax = np.nanpercentile(lbp_array, [2, 98])
            
            heatmap = ax.imshow(lbp_masked, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(heatmap, ax=ax, label="LBP Code Value")

        ax.set_title(f"LBP Texture ({method}, P={n_points}, R={radius})")
        ax.axis("off")

        if save_fig:
            # Use dem_name if available
            name = getattr(self, 'dem_name', 'output')
            output_heatmap_path = f"{name}_lbp_{method}_heatmap.png"
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close(fig)

    def generate_flow_heatmap(self, save_fig=False, show=True):
        """
        Computes the flow direction of a DEM and overlays it as a heatmap.

        Parameters:
        output_heatmap_path (str): Path to save the flow heatmap. If None, does not save.
        show (bool): Whether to display the plot interactively.
        """
        
        flow_array = self.shannon_tool.calculate_flow_direction_arcpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        custom_cmap = ListedColormap([
            (0, 0, 0, 0), "#440154", "#31688E", "#35B779", "#FDE725",
            "#DCE319", "#F8961E", "#D41159", "#6A00A8"
        ])
        custom_cmap.set_under(color='none')
        ax.imshow(flow_array, cmap=custom_cmap, alpha=0.7)
        legend_labels = [
            "0 = background", "1 - East", "2 - Southeast", "4 - South", "8 - Southwest",
            "16 - West", "32 - Northwest", "64 - North", "128 - Northeast"
        ]
        patches = [mpatches.Patch(color=custom_cmap(i), label=legend_labels[i]) for i in range(len(legend_labels))]
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.set_title(f"Flow Heatmap")
        ax.axis("off")
        if save_fig:
            output_heatmap_path = f"{self.dem_path.split('.')[0]}_flow_heatmap.png"
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def visualize_shannon_index(self, window_size=21, save_fig=False, show=True):
        """
        Computes the shannon diversity index of flow directions overlays it as a heatmap.
        Background (0) is masked and shown as white.

        Parameters:
        window_size (int): window size for flow direction calculation
        save_fig (bool): Whether to save the figure.
        show (bool): Whether to display the plot interactively.
        """
        shannon_array = self.hd.calculate_shannon_index_2d(window_size)
        
        # --- FIX: MASK THE BACKGROUND (0) ---
        # Mask values equal to 0 so they are treated as 'bad' values by imshow
        shannon_masked = np.ma.masked_equal(shannon_array, 0)
        
        cmap = plt.cm.jet
        # Set the color for masked values (the background) to white (or 'none' for transparent)
        cmap.set_bad(color='white') 
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the masked array
        # vmin=0 is fine because the actual 0s are now masked out
        heatmap = ax.imshow(shannon_masked, cmap=cmap, alpha=0.7, vmin=0, vmax=2)
        
        plt.colorbar(heatmap, ax=ax, label="Shannon index")
        ax.set_title(f"Shannon diversity Heatmap (window={window_size})")
        ax.axis("off")
        
        if save_fig:
            # Use dem_name if available, else generic name
            name = getattr(self, 'dem_name', 'output')
            output_heatmap_path = f"{name}_shannon_heatmap.png"
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_heatmap_path}")
            
        if show:
            plt.show()
            
        plt.close(fig)

    def generate_tpi_tri_roughness_slope_aspect_heatmap(self, variable, save_fig=None, show=True):
        """
        Computes TPI, TRI, Roughness, Slope, or Aspect of a DEM and overlays it as a heatmap.
        Handles both memory-based (NumPy) and disk-based (GDAL) derivative generation.
        """
        import tempfile
        import numpy as np
        import os

        # 1. Create a temporary file path.
        # GDAL-based tools (Slope/Aspect) need a physical path to write to.
        # NumPy-based tools (TRI) will likely ignore this or return an array anyway.
        fd, temp_path = tempfile.mkstemp(suffix=".tif")
        os.close(fd) # Close the file handle so other tools can write to it

        derivative_data = None

        try:
            # 2. Call the calculation method
            # We pass temp_path to all methods.
            result = None
            if variable == "TPI":
                result = self.hd.calculate_tpi(temp_path)
            elif variable == "TRI":
                result = self.hd.calculate_tri(temp_path)
            elif variable == "Roughness":
                result = self.hd.calculate_roughness(temp_path)
            elif variable == "Slope":
                result = self.hd.calculate_slope(temp_path)
            elif variable == "Aspect":
                result = self.hd.calculate_aspect(temp_path)
            elif variable == "Hillshade":
                result = self.hd.calculate_hillshade(temp_path)
            else:
                print(f"Unknown variable: {variable}")
                return

            # 3. Determine how to get the data
            if isinstance(result, np.ndarray):
                # Case A: The method returned the data directly (NumPy implementation)
                derivative_data = result
            elif os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # Case B: The method wrote to disk (GDAL implementation)
                # We use your utility to read it back
                derivative_data = GdalUtils.read_raster(temp_path)
            else:
                print(f"Error: {variable} calculation failed to return data or write to file.")
                return

            # 4. Visualization
            if derivative_data is not None:
                # Handle MaskedArrays (from rasterio) or NaNs (from numpy)
                if np.ma.is_masked(derivative_data):
                    # It's already masked, just ensure 0 is also masked if needed
                    # derivative_data = np.ma.masked_equal(derivative_data, 0) # Optional
                    pass
                else:
                    # It's a standard array, mask NaNs and potentially 0s
                    derivative_data = np.ma.masked_invalid(derivative_data)
                
                cmap = plt.cm.jet
                cmap.set_bad(color='white') # Show NoData as white

                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Compute robust vmin/vmax to ignore outliers in visualization
                # (TRI/Slope often have massive outliers at edges)
                valid_data = derivative_data.compressed() # Get valid data only
                if valid_data.size > 0:
                    vmin, vmax = np.percentile(valid_data, [2, 98])
                else:
                    vmin, vmax = None, None

                heatmap = ax.imshow(derivative_data, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)
                plt.colorbar(heatmap, ax=ax, label=f"{variable} value")
                ax.set_title(f"Heatmap (derivative={variable})")
                ax.axis("off")

                if save_fig:
                    # Use dem_name if available
                    name = getattr(self, 'dem_name', 'output')
                    output_heatmap_path = f"{name}_{variable.lower()}_heatmap.png"
                    plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
                    print(f"Saved figure to {output_heatmap_path}")
                
                if show:
                    plt.show()
                
                plt.close(fig)

        except Exception as e:
            print(f"An error occurred visualizing {variable}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 5. Cleanup
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def _get_geomorphon_data(self, n_classes):
        """Helper to calculate the specific geomorphon array requested."""
        print(f"Calculating {n_classes}-class Geomorphons...")
        
        # 1. Calculate Raw and 10-Class using the static method
        # Note: We pass self.dem_path because ArcPy needs the file on disk
        raw_array, class10_array = Landforms.calculate_landforms_chunk(
            self.dem_path, 
        )

        if raw_array is None:
            raise RuntimeError("Failed to generate geomorphons via ArcPy/Landforms class.")

        # 2. Return the correct array based on n_classes
        if n_classes == 10:
            return class10_array
        elif n_classes in [6, 5, 4]:
            # Reclassify the RAW data
            key = f"{n_classes}c"
            return Landforms.classify_chunk(raw_array, classes=key)
        else:
            raise ValueError(f"Unsupported class count: {n_classes}. Choose 10, 6, 5, or 4.")

    def visualize_geomorphons(self, n_classes=10, save_fig=False, show=True):
        """
        Visualizes discrete geomorphon classes.
        Remaps discontinuous integer IDs (from subset classifications) to 
        contiguous indices for correct color mapping.
        """
        try:
            # 1. Get Data
            class_array = self._get_geomorphon_data(n_classes)
            
            # 2. Define Styles AND Expected IDs
            # The 'ids' list must be sorted and correspond exactly to the 'labels' list order.
            styles = {
                10: {
                    'ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'colors': ['#d3d3d3', '#ff0000', '#ff5500', '#ffff00', '#aaff00', 
                               '#00aa00', '#00ffaa', '#0000ff', '#aa00ff', '#000000'],
                    'labels': ['Flat', 'Peak', 'Ridge', 'Shoulder', 'Spur', 
                               'Slope', 'Hollow', 'Footslope', 'Valley', 'Pit']
                },
                6: {
                    # FL(1), RI(3), SH(4), SL(6), FS(8), VL(9)
                    'ids': [1, 3, 4, 6, 8, 9],
                    'colors': ['#d3d3d3', '#ff5500', '#ffff00', '#00aa00', '#0000ff', '#aa00ff'],
                    'labels': ['Flat', 'Ridge', 'Shoulder', 'Slope', 'Footslope', 'Valley']
                },
                5: {
                    # FL(1), PK(2), RI(3), SL(6), VL(9)
                    'ids': [1, 2, 3, 6, 9],
                    'colors': ['#d3d3d3', '#ff0000', '#ff5500', '#00aa00', '#aa00ff'],
                    'labels': ['Flat', 'Peak', 'Ridge', 'Slope', 'Valley']
                },
                4: {
                    # FL(1), RI(3), SL(6), VL(9)
                    'ids': [1, 3, 6, 9],
                    'colors': ['#d3d3d3', '#ff5500', '#00aa00', '#aa00ff'],
                    'labels': ['Flat', 'Ridge', 'Slope', 'Valley']
                }
            }

            if n_classes not in styles:
                print(f"Visualization not defined for {n_classes} classes. Using default jet.")
                plot_data = class_array
                cmap = plt.cm.jet
                vmin, vmax = None, None
                labels, colors = None, None
            else:
                style = styles[n_classes]
                colors = style['colors']
                labels = style['labels']
                expected_ids = style['ids']
                
                # --- CRITICAL FIX: REMAP IDS TO CONTIGUOUS INDICES ---
                # We want: ID 1 -> Plot Val 1
                #          ID 9 -> Plot Val 4 (if 4-class)
                # This ensures the 4th color in the list is applied to ID 9.
                
                plot_data = np.zeros_like(class_array, dtype=np.int32)
                
                # Iterate through the expected IDs and map them to 1..N
                for idx, true_id in enumerate(expected_ids):
                    # Mask where the array equals the true ID (e.g. 9)
                    # Set those pixels to the sequential index (e.g. 4)
                    plot_data[class_array == true_id] = idx + 1
                
                # Create discrete colormap
                cmap = ListedColormap(colors)
                # Center colors on integers 1, 2, ..., N
                vmin = 0.5
                vmax = len(colors) + 0.5

            # 3. Mask Background
            # 0 is always background/NoData in both original and remapped arrays
            morph_masked = np.ma.masked_equal(plot_data, 0)

            # 4. Plotting
            fig, ax = plt.subplots(figsize=(12, 10))
            
            im = ax.imshow(morph_masked, cmap=cmap, interpolation='nearest', 
                           vmin=vmin, vmax=vmax)

            ax.set_title(f"{n_classes}-Class Geomorphons")
            ax.axis("off")

            # 5. Legend
            if labels and colors:
                patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
                ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', 
                          borderaxespad=0., title="Landforms")

            plt.tight_layout()

            if save_fig:
                name = Utils.sanitize_path_to_name(self.dem_path)
                output_path = f"{name}_{n_classes}class_geomorphons.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved figure to {output_path}")

            if show:
                plt.show()
                
            plt.close(fig)
            
        except Exception as e:
            print(f"Error visualizing geomorphons: {e}")
            import traceback
            traceback.print_exc()

    def analyze_geomorphon_data(self, n_classes=10):
        """
        Analyze classified geomorphons and plot a histogram.
        Calculates the data on the fly since we are working from a raw DEM.
        """
        try:
            # 1. Get Data
            class_array = self._get_geomorphon_data(n_classes)
            
            # Mask NoData (0)
            valid_mask = class_array > 0
            masked_data = class_array[valid_mask]

            if masked_data.size == 0:
                print("No valid geomorphon data found (all 0/NoData).")
                return

            # 2. Get Labels (Reusing the definitions from visualizer for consistency)
            # You could also move 'styles' to the class level to avoid duplication
            labels_map = {
                10: ['Flat', 'Peak', 'Ridge', 'Shoulder', 'Spur', 'Slope', 'Hollow', 'Footslope', 'Valley', 'Pit'],
                6: ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'],
                5: ['High', 'Mid-High', 'Flat/Slope', 'Mid-Low', 'Low'],
                4: ['Ridge/Peak', 'Slope', 'Flat', 'Valley/Pit']
            }
            
            labels = labels_map.get(n_classes, [f"Class {i}" for i in range(1, n_classes+1)])
            colors = plt.cm.get_cmap("tab10", n_classes).colors

            # 3. Plot Histogram
            fig, ax = plt.subplots(figsize=(6, 3))
            
            # We iterate through possible class IDs (1 to N)
            counts = []
            x_ticks = range(1, n_classes + 1)
            
            for i in x_ticks:
                count = np.sum(masked_data == i)
                counts.append(count)

            ax.bar(x_ticks, counts, color=colors, edgecolor='black')

            # Formatting
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_xlabel("Landform Class")
            ax.set_ylabel("Frequency (Pixel Count)")
            ax.set_title(f"Histogram of {n_classes}-Class Geomorphons")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing geomorphons: {e}")
# example usage
if __name__=="__main__":
    dem_path = r"C:\Users\ageglio\Documents\NLM_DataRelease\NLM_DataRelease\IngallsPoint_2021\0.5m\IP_BY_0.5m.tif"
    plotter = PlotDEM(dem_path)
    plotter.plot_dem(title="IP Bathy Heatmap", cmap="terrain", vmin=-100, vmax=100)