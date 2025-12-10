import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from derivatives import HabitatDerivatives
from rasterUtils import RasterUtils
from shannon import ShannonDerivatives
from gdalUtils import GdalUtils


class PlotDEM:
    def __init__(self, dem_path):
        self.dem_path = dem_path

        ## creating DEM data array object
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)  # Read the DEM data
            transform = src.transform  # Get the affine transform
            crs = src.crs  # Get the coordinate reference system
            metadata = src.meta  # Get metadata
            nodata = metadata['nodata']  # Get no-data value
        dem_data = RasterUtils.replace_nodata_with_nan(dem_data, nodata=-9999)
        # Check if the DEM data is empty after filling no-data values
        if RasterUtils.is_empty(dem_data, nodata):
            print("The DEM data is empty after filling no-data values.")
        self.dem_data = dem_data
        self.transform = transform
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

    def generate_lbp_heatmap(self, n_points, radius, method='default', save_fig=False, show=True):
        """
        Computes the LBP of a DEM and overlays it as a heatmap.

        Parameters:
        output_heatmap_path (str): Path to save the LBP heatmap. If None, does not save.
        show (bool): Whether to display the plot interactively.
        n_points (int): Number of neighboring points for LBP.
        radius (int): Radius for LBP calculation.

        Returns:
        None (Saves the heatmap and displays it).
        """
        lbp_array = self.hd.calculate_lbp(n_points, radius, method=method)
        cmap = plt.cm.jet
        cmap.set_under(color='none')
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(lbp_array, cmap=cmap, alpha=0.7, vmin=0.1)
        plt.colorbar(heatmap, ax=ax, label="LBP Intensity")
        ax.set_title(f"LBP Heatmap (window=NONE, n_points={n_points}, radius={radius}, method={method})")
        ax.axis("off")
        if save_fig:
            output_heatmap_path = f"{self.dem_path.split('.')[0]}_lbp_heatmap.png"
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

    def visualize_geomorphons(self, n_classes=10, save_fig=False, show=True):
        """
        Visualizes discrete geomorphon classes (10, 6, 5, or 4) with a categorical legend.

        Parameters:
        n_classes (int): The number of classes to visualize (10, 6, 5, or 4).
        save_fig (bool): Whether to save the figure.
        show (bool): Whether to display the plot.
        """
        
        # 1. Get the Data
        # Replace 'generate_bathymorphons' with your actual method name
        morph_array = self.hd.generate_bathymorphons(n_classes)
        
        # Mask NoData (assuming 0 is NoData)
        morph_masked = np.ma.masked_equal(morph_array, 0)

        # 2. Define Styling (Colors and Labels) based on class count
        # You may need to adjust the labels to match your specific reclassification logic
        styles = {
            10: {
                # Standard Geomorphon Colors (r.geomorphon style)
                # 1:Flat, 2:Peak, 3:Ridge, 4:Shoulder, 5:Spur, 6:Slope, 7:Hollow, 8:Footslope, 9:Valley, 10:Pit
                'colors': ['#d3d3d3', '#ff0000', '#ff5500', '#ffff00', '#aaff00', 
                           '#00aa00', '#00ffaa', '#0000ff', '#aa00ff', '#000000'],
                'labels': ['Flat', 'Peak', 'Ridge', 'Shoulder', 'Spur', 
                           'Slope', 'Hollow', 'Footslope', 'Valley', 'Pit']
            },
            6: {
                'colors': ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba', '#808080'],
                'labels': ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']
            },
            5: {
                'colors': ['#ca0020', '#f4a582', '#f7f7f7', '#92c5de', '#0571b0'],
                'labels': ['High', 'Mid-High', 'Flat/Slope', 'Mid-Low', 'Low']
            },
            4: {
                'colors': ['#e66101', '#fdb863', '#b2abd2', '#5e3c99'],
                'labels': ['Ridge/Peak', 'Slope', 'Flat', 'Valley/Pit']
            }
        }

        if n_classes not in styles:
            print(f"Visualization not defined for {n_classes} classes. Using default jet.")
            cmap = plt.cm.jet
            labels = None
        else:
            colors = styles[n_classes]['colors']
            labels = styles[n_classes]['labels']
            # Create a discrete colormap
            cmap = mcolors.ListedColormap(colors)

        # 3. Plotting
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # We use vmin=0.5 and vmax=n_classes+0.5 to center colors on integers 1..N
        im = ax.imshow(morph_masked, cmap=cmap, interpolation='nearest', 
                       vmin=0.5, vmax=n_classes + 0.5)

        ax.set_title(f"{n_classes}-Class Geomorphons")
        ax.axis("off")

        # 4. Create Custom Legend
        if labels:
            # Create a list of patches (colored rectangles) for the legend
            patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
            
            # Place legend outside the plot to the right
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', 
                      borderaxespad=0., title="Landforms")

        plt.tight_layout()

        # 5. Save and Show
        if save_fig:
            # Use dem_name if available, otherwise generic
            name = getattr(self, 'dem_name', 'output') 
            output_path = f"{name}_{n_classes}class_geomorphons.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_path}")

        if show:
            plt.show()
            
        plt.close(fig)

# example usage
if __name__=="__main__":
    dem_path = r"C:\Users\ageglio\Documents\NLM_DataRelease\NLM_DataRelease\IngallsPoint_2021\0.5m\IP_BY_0.5m.tif"
    plotter = PlotDEM(dem_path)
    plotter.plot_dem(title="IP Bathy Heatmap", cmap="terrain", vmin=-100, vmax=100)