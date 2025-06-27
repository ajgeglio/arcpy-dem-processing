import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.patches as mpatches
from derivatives import HabitatDerivatives

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
            hd = HabitatDerivatives(dem_path)
            dem_data = hd.fill_nodata(dem_data, nodata)  # Fill no-data values
            # Check if the DEM data is empty after filling no-data values
            if hd.is_empty(dem_data, nodata):
                print("The DEM data is empty after filling no-data values.")
        self.dem_data = dem_data
        self.transform = transform

    def plot_dem(self, output_heatmap_path=None, show=True):
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
        heatmap = ax.imshow(dem_array, cmap=cmap, interpolation='none', vmin=150, alpha=0.7)  # DEM background
        plt.colorbar(heatmap, ax=ax, label="Elevation (m)")
        ax.set_title("DEM Heatmap")
        ax.axis("off")
        if output_heatmap_path:
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

    def generate_lbp_heatmap(self, n_points, radius, method='default', output_heatmap_path=None, show=True):
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
        dem_array = self.dem_data
        hd = HabitatDerivatives(input_dem=self.dem_path)
        lbp_array = hd.calculate_lbp(dem_array, n_points, radius, method=method)
        cmap = plt.cm.jet
        cmap.set_under(color='none')
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(lbp_array, cmap=cmap, alpha=0.7, vmin=0.1)
        plt.colorbar(heatmap, ax=ax, label="LBP Intensity")
        ax.set_title(f"LBP Heatmap (window=NONE, n_points={n_points}, radius={radius}, method={method})")
        ax.axis("off")
        if output_heatmap_path:
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def generate_flow_heatmap(self, output_heatmap_path=None, show=True):
        """
        Computes the flow direction of a DEM and overlays it as a heatmap.

        Parameters:
        output_heatmap_path (str): Path to save the flow heatmap. If None, does not save.
        show (bool): Whether to display the plot interactively.
        """
        dem_array = self.dem_data
        hd = HabitatDerivatives(input_dem=self.dem_path)
        flow_array = hd.calculate_flow_direction(dem_array)
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
        if output_heatmap_path:
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return flow_array

    def visualize_flow_direction_and_shannon_index(self, window_size=21, output_heatmap_path=None, show=True):
        """
        Computes the shannon diversity index of flow directions overlays it as a heatmap.

        Parameters:
        output_heatmap_path (str): Path to save the heatmap. If None, does not save.
        show (bool): Whether to display the plot interactively.
        window = window size for flow direction calculation
        Returns:
        None (Saves the heatmap and displays it).
        """
        dem_array = self.dem_data
        hd = HabitatDerivatives(input_dem=self.dem_path)
        nodata = dataset.nodata if dataset.nodata is not None else -9999
        shannon_array = hd.calculate_shannon_index_2d(dem_array, nodata, window_size)
        cmap = plt.cm.jet
        cmap.set_under(color='none')
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(shannon_array, cmap=cmap, alpha=0.7, vmin=0, vmax=2)
        plt.colorbar(heatmap, ax=ax, label="Shannon index")
        ax.set_title(f"Shannon diversity Heatmap (window={window_size})")
        ax.axis("off")
        if output_heatmap_path:
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return shannon_array

    def generate_tpi_tri_roughness_slope_aspect_heatmap(self, variable, output_heatmap_path=None, show=True):
        """
        Computes TPI, TRI, Roughness, Slope, or Aspect of a DEM and overlays it as a heatmap.

        Parameters:
        output_heatmap_path (str): Path to save the heatmap. If None, does not save.
        show (bool): Whether to display the plot interactively.
        variable (str): The variable to compute and plot (TPI, TRI, Roughness, Slope, Aspect).

        Returns:
        None (Saves the heatmap and displays it).
        """
        dem_array = self.dem_data
        transform = self.transform
        hd = HabitatDerivatives(input_dem=self.dem_path)
        if variable == "TPI":
            derivative = hd.calculate_tpi(dem_array, transform, output_tpi=None)
        elif variable == "TRI":
            derivative = hd.calculate_tri(dem_array, transform, output_tri=None)
        elif variable == "Roughness":
            derivative = hd.calculate_roughness(dem_array, transform, output_roughness=None)
        elif variable == "Slope":
            derivative = hd.calculate_slope(dem_array, transform, output_slope=None)
        elif variable == "Aspect":
            derivative = hd.calculate_aspect(dem_array, transform, output_aspect=None)
        cmap = plt.cm.jet
        cmap.set_under(color='none')
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap = ax.imshow(derivative, cmap=cmap, alpha=0.7)
        plt.colorbar(heatmap, ax=ax, label=f"{variable} value")
        ax.set_title(f"Heatmap (derivative={variable})")
        ax.axis("off")
        if output_heatmap_path:
            plt.savefig(output_heatmap_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return
    
    # example usage

if __name__=="__main__":
    dem_path = r"C:\Users\ageglio\Documents\NLM_DataRelease\NLM_DataRelease\IngallsPoint_2021\0.5m\IP_BY_0.5m.tif"
    plotter = PlotDEM(dem_path)
    variable = "TRI"
    plotter.generate_tpi_tri_roughness_slope_aspect_heatmap(
        variable=variable, output_heatmap_path=None, show=True
    )

    n_points = 9
    radius = 1
    method = 'default'  # or 'uniform', 'nri_uniform', 'ror'
    # Generate LBP heatmap
    plotter.generate_lbp_heatmap(dem_data, radius, n_points, method)