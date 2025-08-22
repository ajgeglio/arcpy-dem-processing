from osgeo import gdal
import rasterio
import numpy as np
import os
import glob
import subprocess
from pathlib import Path
from utils import Utils

class GdalUtils:
    """Utility class for handling GDAL operations, including compression and conversion of DEM files."""
    @staticmethod
    def compress_tiff_with_rasterio(dem_path):
        """
        Function that replaces nodata values with None, compresses using GDAL BigTIFF, 
        and compresses the DEM file using LZW compression.
        Parameters:
        dem_path (str): Path to the input DEM file.
        Returns:
        output_path (str): Path to the compressed DEM file.
        """
        # Open the input DEM file
        with rasterio.open(dem_path) as src:
            metadata = src.meta.copy()  # Copy metadata
            metadata.update({
                'dtype': 'float32',
                'compress': 'LZW',
                'nodata': None,  # Set no-data value to None (no explicit nodata)
                'driver': 'GTiff',
                'BIGTIFF': 'YES'
            })

            # Define output file path
            output_path = os.path.join(os.path.dirname(dem_path), os.path.basename(dem_path).split(".")[0] + "_compressed.tif")

            # Write the modified DEM to a new file in chunks
            with rasterio.open(output_path, 'w', **metadata) as dst:
                for ji, window in src.block_windows(1):  # Process by blocks
                    dem_data = src.read(1, window=window, masked=True)  # Ensure masked=True
                    dem_data = np.where(np.isnan(dem_data) | np.isinf(dem_data) | (dem_data <= -9999), None, dem_data)
                    dst.write(dem_data.astype('float32'), 1, window=window)
        print(f"Reduced and compressed DEM saved to {output_path}")
        return output_path

    @staticmethod
    def compress_tiff_with_gdal(input_dem_path, out_dem_folder, compress=True):
        """
        Convert a TIFF file to a GDAL raster TIFF using gdal_translate.
        
        :param input_dem_path: Path to the input TIFF file.
        :param output_tiff: Path to save the output raster TIFF file.
        """
        # Open the input TIFF file
        src_ds = gdal.Open(input_dem_path)
        dem_name = Utils.sanitize_path_to_name(input_dem_path)
        output_tiff = os.path.join(out_dem_folder, dem_name+"_gdal.tif")
        # Use gdal_translate to convert the file
        if compress:
            gdal.Translate(output_tiff, src_ds, format='GTiff', creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'], outputType=gdal.GDT_Float32)
        else:
            gdal.Translate(output_tiff, src_ds, format='GTiff', outputType=gdal.GDT_Float32)
        # Remove the original input TIFF file
        gdal.Dataset.__swig_destroy__(src_ds)  # Close the dataset
        del src_ds  # Delete the dataset reference
        os.remove(input_dem_path)
        print(f"Converted {input_dem_path} to {output_tiff} with compression={compress} as GDAL raster TIFF.")
        return output_tiff
    
    def convert_to_cog(input_path, output_path=None, compress="DEFLATE", tile_size=512):
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".cog.tif")
        else:
            output_path = Path(output_path)

        cmd = [
            "gdal_translate",
            str(input_path),
            str(output_path),
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", f"COMPRESS={compress}",
            "-co", f"BLOCKXSIZE={tile_size}",
            "-co", f"BLOCKYSIZE={tile_size}",
            "-co", "COPY_SRC_OVERVIEWS=YES",
            "-co", "BIGTIFF=YES"
        ]

        print("Running command:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        print(f"\nConversion complete: {output_path}")

    @staticmethod
    def merge_raster_gdal(raster_chunks_path, output_format="LZW", tile_size=512, remove=False):
        """
        Merges DEM files in the specified path using GDAL.
        Supports output as LZW-compressed TIFF or Deflate-compressed COG.
        Removes individual tiles after merge if requested.
        Replaces NaN and infinite values with nodata (-9999) and updates metadata.

        Parameters:
            dem_chunks_path (str): Path to folder containing DEM tiles.
            remove (bool): Whether to delete individual tiles after merge.
            output_format (str): "LZW" or "COG"
            tile_size (int): Tile size for COG output (default: 512)
        """

        dem_name = os.path.basename(raster_chunks_path)
        dem_files = glob.glob(os.path.join(raster_chunks_path, "*.tif"))

        if not dem_files:
            print(f"No DEM files found in the path: {raster_chunks_path}")
            return None
        
        # projections = set()
        # for tif in dem_files:
        #     ds = gdal.Open(tif)
        #     if ds:
        #         projections.add(ds.GetProjection())
        # if len(projections) > 1:
        #     print("Warning: DEM tiles have inconsistent projections.")


        # Build VRT
        vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')
        vrt_dataset = gdal.BuildVRT('/vsimem/mosaic.vrt', dem_files, options=vrt_options)
        if vrt_dataset is None or vrt_dataset.RasterCount == 0:
            print("VRT creation failed or resulted in empty dataset.")
            return None

        # Define output path
        suffix = ".cog.tif" if output_format.upper() == "COG" else ".tif"
        merged_dem_path = os.path.join(os.path.dirname(raster_chunks_path), f"{dem_name}_merged{suffix}")

        # Set creation options
        if output_format.upper() == "COG":
            creation_options = [
                "TILED=YES",
                "COMPRESS=DEFLATE",
                f"BLOCKXSIZE={tile_size}",
                f"BLOCKYSIZE={tile_size}",
                "COPY_SRC_OVERVIEWS=YES",
                "BIGTIFF=YES"
            ]
        elif output_format.upper() == "LZW":
            creation_options = ["COMPRESS=LZW", "BIGTIFF=YES"]
        else:
            raise ValueError("output_format must be 'COG' or 'LZW'")

        print(f"DEM files to merge: {dem_files}")
        print(f"Output merged DEM path: {merged_dem_path}")

        # Translate VRT to final output
        try:
            result = gdal.Translate(
                merged_dem_path,
                vrt_dataset,
                format='GTiff',
                creationOptions=creation_options,
                noData=-9999
            )
        except Exception as e:
            print(f"gdal.Translate threw an exception: {e}")
            return None

        # Clean up VRT
        vrt_dataset = None

        if result is None:
            print(f"gdal.Translate failed to create output: {merged_dem_path}")
            return None

        # Confirm output file exists
        if not os.path.isfile(merged_dem_path):
            print(f"Output file was not created: {merged_dem_path}")
            return None

        if remove:
            for file in dem_files:
                os.remove(file)
            print("Individual DEM tiles have been removed.")

        print(f"Merged DEM saved at: {merged_dem_path}")
        return merged_dem_path
    
    def build_cog_overviews(raster_path, levels=[2, 4, 8, 16, 32], resampling='nearest'):
        """
        Builds external overviews (.ovr) for a Cloud Optimized GeoTIFF (COG).

        Parameters:
            raster_path (str): Path to the input COG raster file.
            levels (list): List of overview levels to generate.
            resampling (str): Resampling method (e.g., 'average', 'nearest', 'mode').

        Returns:
            None
        """
        if not os.path.isfile(raster_path):
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        # Open in read-only mode to force external .ovr creation
        dataset = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if not dataset:
            raise RuntimeError(f"Failed to open raster: {raster_path}")

        # Build external overviews
        result = dataset.BuildOverviews(resampling, levels)
        if result != 0:
            raise RuntimeError(f"Overview generation failed for {raster_path}")

        print(f"External overviews (.ovr) created for {raster_path} using {resampling} resampling.")

        dataset = None

    @staticmethod
    def validate_rasters(raster_list):
        results = {}
        for tif in raster_list:
            try:
                ds = gdal.Open(tif)
                if ds is None:
                    results[tif] = "Unreadable"
                else:
                    # Optional: check for raster bands
                    if ds.RasterCount == 0:
                        results[tif] = "No raster bands"
                    else:
                        results[tif] = "Valid"
            except Exception as e:
                results[tif] = f"Error: {str(e)}"
        return results