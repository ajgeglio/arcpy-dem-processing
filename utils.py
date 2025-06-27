import os
import shutil
import datetime
import numpy as np
import pandas as pd
import pickle
import re
import glob
import rasterio
from osgeo import gdal
from scipy import ndimage

class ReturnTime:
    def __init__(self):
        pass

    @staticmethod
    def get_time_obj(time_s):
        if pd.notnull(time_s):
            return datetime.datetime.fromtimestamp(time_s)
        return np.nan

    @classmethod
    def get_Y(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%Y') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_m(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%m') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_d(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%d') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_t(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%H:%M:%S') if isinstance(dt, datetime.datetime) else np.nan

class WorkspaceCleaner:
    def __init__(self, class_instance):
        """
        Initializes the cleaner with paths to temporary workspaces.
        Args:
            temp_workspace_path (str): Path to the primary temporary workspace.
            local_temp_path (str): Path to the local temporary workspace.
            utils_instance: An instance of your Utils class for printing progress.
        """
        self.temp_workspace = class_instance.temp_workspace
        self.local_temp = class_instance.local_temp
        self.message_length = 0
        self._current_cleanup_problem_paths = [] # Stores paths that failed in the current rmtree op

    def _shutil_onerror_handler(self, func, path, exc_info):
        """
        Error handler called by shutil.rmtree when an error occurs.
        - func: The function that raised the exception (e.g., os.remove, os.rmdir).
        - path: The path name passed to func.
        - exc_info: Exception information (type, value, traceback) from sys.exc_info().
        """
        error_type, error_value, _ = exc_info
        # Construct a detailed message about the specific file/directory causing the issue.
        # error_value often contains useful OS-specific error codes like "[WinError 32]" for file in use.
        error_message = f"Problem during cleanup: Cannot {func.__name__} '{path}'. Reason: {error_value}"
        
        # Use a simple print for detailed errors to avoid interfering too much
        # with a potentially complex print_progress.
        self._current_cleanup_problem_paths.append(path)
        # Note: We don't re-raise the exception here; shutil.rmtree will continue if possible
        # and raise its own exception at the end if the directory couldn't be fully removed.

    def _delete_workspace_attempt(self, workspace_display_name, workspace_path):
        """
        Attempts to delete a single workspace directory and reports issues.
        Returns:
            bool: True if the workspace was successfully deleted or didn't exist, False otherwise.
        """
        self.message_length = Utils.print_progress(
            f"Processing {workspace_display_name} for cleanup at: {workspace_path}",
            self.message_length
        )

        if not os.path.exists(workspace_path):
            self.message_length = Utils.print_progress(
                f"  {workspace_display_name} not found (already deleted or never created).",
                self.message_length
            )
            print()
            return True

        self._current_cleanup_problem_paths = []  # Reset list for this attempt
        try:
            shutil.rmtree(workspace_path, onerror=self._shutil_onerror_handler)
            
            # After rmtree, check if the path still exists.
            # If it does, it means rmtree failed to remove the top-level directory.
            if os.path.exists(workspace_path):
                # _shutil_onerror_handler would have printed details about specific files.
                # final_message = (f"  Failed to completely delete {workspace_display_name} at '{workspace_path}'.\n"
                #                  f"  Problematic items encountered (see details above): {self._current_cleanup_problem_paths}")
                # self.message_length = Utils.print_progress(final_message, self.message_length)
                return False
            else:
                # Directory was successfully removed.
                if self._current_cleanup_problem_paths:
                    # This means some files/subdirs caused errors (and were reported by the handler),
                    # but shutil.rmtree eventually succeeded in removing the main directory.
                    final_message = (f"  Successfully deleted {workspace_display_name}: {workspace_path}.\n"
                                     f"  Note: Some internal items reported errors during deletion but were ultimately handled: {self._current_cleanup_problem_paths}")
                    self.message_length = Utils.print_progress(final_message, self.message_length)
                else:
                    final_message = f"  Successfully deleted {workspace_display_name}: {workspace_path}"
                    self.message_length = Utils.print_progress(final_message, self.message_length)
                print()
                return True

        except Exception as e:
            # This catches errors if rmtree itself fails critically (e.g., top-level directory
            # permission issues not handled by onerror, or if onerror itself raises an error).
            # The _current_cleanup_problem_paths might have been populated by the handler.
            error_details = f"Problematic items reported: {self._current_cleanup_problem_paths}" if self._current_cleanup_problem_paths else ""
            final_message = (f"  A critical error occurred while trying to delete {workspace_display_name} at '{workspace_path}': {e}.\n"
                             f"  {error_details}")
            self.message_length = Utils.print_progress(final_message, self.message_length)
            print()
            return False

    def clean_up(self):
        """
        Clean up all temporary files by deleting the specified temporary workspace directories.
        Reports specific files or directories that could not be deleted.
        Returns:
            bool: True if all cleanup operations were successful, False otherwise.
        """
        message = "Starting cleanup of temporary files..."
        self.message_length = Utils.print_progress(message, self.message_length)

        overall_success = True

        # Clean primary temporary workspace
        if self.temp_workspace: # Ensure path is provided
            if not self._delete_workspace_attempt("Temporary Workspace", self.temp_workspace):
                overall_success = False
        else:
            self.message_length = Utils.print_progress("Temporary Workspace path not configured, skipping.", self.message_length)

        # Clean local temporary workspace
        if self.local_temp: # Ensure path is provided
            if not self._delete_workspace_attempt("Local Temporary Workspace", self.local_temp):
                overall_success = False
        else:
            self.message_length = Utils.print_progress("Local Temporary Workspace path not configured, skipping.", self.message_length)
            
        if overall_success:
            self.message_length = Utils.print_progress(
                "Temporary file cleanup process completed successfully.",
                self.message_length
            )
        else:
            self.message_length = Utils.print_progress(
                "Temporary file cleanup process completed WITH ERRORS. Some files/directories may not have been deleted. Please check logs above.",
                self.message_length
            )
        return overall_success

class Utils:
    @staticmethod
    def time_convert(sec):
        mins = sec // 60
        sec = sec % 60
        hours = mins // 60
        mins = mins % 60
        print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

    @staticmethod
    def print_progress(message, previous_length=0):
        """
        Print a progress message in the terminal, overwriting the previous line.

        Parameters:
        - message (str): The message to display.
        - previous_length (int): The length of the previous message (optional).
        """
        # Get terminal width
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        # Pad the message to overwrite the previous one
        padded_message = message.ljust(max(previous_length, terminal_width))
        print(padded_message, end="\r")
        return len(message)

    def sanitize_path_to_name(self, path):
        """Replace invalid characters and get the base file name from a path."""
        name = os.path.splitext(os.path.basename(path))[0]
        return name.replace(".", "_").replace(" ", "_")

    def bagfile_from_image(filename, aluim_path, window=30):
        image_id = filename.split(".")[0]
        time_sec = float(image_id.split("_")[1] +"." + image_id.split("_")[2])
        print("Image Date and Time:")
        print(datetime.datetime.fromtimestamp(time_sec).strftime('%Y-%m-%d %H:%M:%S'))
        aluim = pd.read_csv(aluim_path, index_col=0, low_memory=False)
        aluim21_window = aluim[(aluim.Time_s >= time_sec-window/2) & (aluim.Time_s <= time_sec+window/2)]
        print(aluim21_window.BagFile.values)

    def copy_files(src_pths, dest_folder):
        l = len(src_pths)
        for i, img_pth in enumerate(src_pths):
            src, dst = img_pth, dest_folder
            # File copy is interrupted often due to network, added src/dest comparison
            if os.path.exists(src):
                if os.path.exists(dst):
                    if os.stat(src).st_size == os.stat(dst).st_size:
                        continue
                    else:
                        shutil.copy(src, dst)
                else:
                    shutil.copy(src, dst)
                print("Copying", i,"/",l, end='  \r')
            else: print(f"{src} not found")

    def load_pickle(pickle_pth): #unpickling
        with open(pickle_pth, "rb") as fp:   
            pf = pickle.load(fp)
        return pf
    def dump_pickle(pickle_pth, pf): #pickling
        with open(pickle_pth, "wb") as fp:
            pickle.dump(pf, fp)

    def list_collects(filepath):
        pat = '([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9]*_[a-z,A-Z]*[0-9]*[a-z,A-Z]*)'
        paths = glob.glob(os.path.join(filepath, "*"))
        collects = [p for p in paths if re.search(pat, p)]
        non_matching = [p for p in paths if not re.search(pat, p)]
        if non_matching:
            print("Non-matching paths:", non_matching)
        return collects

    def list_files(filepath, extensions=[]):
        """
        List all files in a directory (recursively) that match any of the given extensions.

        Args:
            filepath (str): The root directory to search.
            extensions (list): List of file extensions to match (e.g., ['.jpg', '.png']).

            list: List of matching file paths.
        """
        paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    paths.append(os.path.join(root, file))
        return paths

    def create_empty_txt_files(filename_list, ext=".txt"):
        for fil in filename_list:
            file = fil + f"{ext}"
            with open(file, 'w'):
                continue

    def make_datetime_folder():
        t = datetime.datetime.now()
        timestring = f"{t.year:02d}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"
        Ymmdd = timestring.split("-")[0]
        out_folder = f"2019-2023-{Ymmdd}"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        print(out_folder)
        return out_folder, Ymmdd
    
class demUtils:
    @staticmethod
    def is_empty(dem_data, nodata):
        """
        Function to check if the cleaned DEM data is empty.
        This function removes NaN values, -9999, 0, and infinite values from the DEM data."""
        # Remove NaN values, -9999, 0, and infinite values
        cleaned_data = dem_data[~(np.isnan(dem_data)) & ~(dem_data == nodata) & ~(dem_data == 0) & ~(np.isinf(dem_data))]
        return cleaned_data.size == 0

    @staticmethod
    def is_low_variance(dem_data, nodata, n=2):
        """
        Function to check if the number of unique values in the cleaned data array are less than or equal to 2.

        Returns:
        bool: True if the unique values array length is <= 2, False otherwise.
        """
        cleaned_data = dem_data[~(np.isnan(dem_data)) & ~(dem_data == nodata) & ~(dem_data == 0) & ~(np.isinf(dem_data))]
        return len(np.unique(cleaned_data)) <= n

    @staticmethod
    def merge_dem(dem_chunks_path, remove=True):
        """
        Function to merge DEM files in the specified path using GDAL.
        Removes the individual DEM tiles after a successful merge.
        Replaces NaN and infinite values with the nodata value (-9999) and updates the metadata.
        """
        dem_name = os.path.basename(dem_chunks_path)
        
        # Find all .tif files in the specified path
        dem_files = glob.glob(os.path.join(dem_chunks_path, "*.tif"))
        if not dem_files:
            print(f"No DEM files found in the path: {dem_chunks_path}")
            return None

        # Create a Virtual Raster (VRT) to mosaic DEM files
        vrt_options = gdal.BuildVRTOptions(resampleAlg='bilinear')  # Try 'bilinear' or 'cubic' for better alignment
        vrt_dataset = gdal.BuildVRT('/vsimem/mosaic.vrt', dem_files, options=vrt_options)

        # Define the output file path
        merged_dem_path = os.path.join(os.path.dirname(dem_chunks_path), f"{dem_name}_merged.tif")
        # Translate the VRT into a GeoTIFF
        gdal.Translate(
            merged_dem_path,
            vrt_dataset,
            format='GTiff',
            creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES']
        )
        # Close the VRT dataset
        vrt_dataset = None

        if remove:
            # Remove the individual DEM tiles
            for file in dem_files:
                os.remove(file)
            print("Individual DEM tiles have been removed.")
        print(f"Merged DEM saved at: {merged_dem_path}")
        return merged_dem_path

    @staticmethod
    def reduce_and_compress_dem(dem_path):
        """
        Function that replaces nodata values with -9999, compresses using GDAL BigTIFF, 
        and compresses the DEM file using LZW compression.

        Parameters:
        dem_path (str): Path to the input DEM file.

        Returns:
        None
        """
        # Open the input DEM file
        with rasterio.open(dem_path) as src:
            metadata = src.meta.copy()  # Copy metadata
            metadata.update({
                'dtype': 'float32',
                'compress': 'LZW',
                'nodata': np.nan,  # Set no-data value to NaN
                'driver': 'GTiff',
                'BIGTIFF': 'YES'
            })

            # Define output file path
            output_path = os.path.join(os.path.dirname(dem_path), os.path.basename(dem_path).split(".")[0] + "_compressed.tif")

            # Write the modified DEM to a new file in chunks
            with rasterio.open(output_path, 'w', **metadata) as dst:
                for ji, window in src.block_windows(1):  # Process by blocks
                    dem_data = src.read(1, window=window)
                    dem_data = np.where(np.isnan(dem_data) | np.isinf(dem_data) | (dem_data < -9999), np.nan, dem_data)
                    dst.write(dem_data.astype('float32'), 1, window=window)
        print(f"Reduced and compressed DEM saved to {output_path}")

    @staticmethod
    def convert_tiff_to_gdal_raster(input_dem_path, out_dem_folder, compress=True):
        """
        Convert a TIFF file to a GDAL raster TIFF using gdal_translate.
        
        :param input_dem_path: Path to the input TIFF file.
        :param output_tiff: Path to save the output raster TIFF file.
        """
        # Open the input TIFF file
        src_ds = gdal.Open(input_dem_path)
        dem_name = Utils().sanitize_path_to_name(input_dem_path)
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

    @staticmethod
    def replace_nodata_with_nan(dem_data, nodata=-9999):
        """
        Replace no-data values in the DEM data with NaN.
        """
        # Replace no-data values with NaN
        dem_data = np.where(dem_data == nodata, np.nan, dem_data)
        return dem_data

    @staticmethod
    def fill_nodata(dem_data, nodata=-9999, max_iterations=10, initial_window_size=3):
        """
        Fill no-data values in the DEM data with the mean of the surrounding values.

        Returns:
        numpy.ndarray: DEM data with no-data gaps filled.
        """
        # Create a mask for no-data values
        mask = np.where(dem_data == nodata, True, False)
        window_size = initial_window_size
        for iteration in range(max_iterations):
            if not np.any(mask):
                break  # Exit if no gaps remain

            # Fill no-data values with the mean of surrounding values
            filled_data = ndimage.generic_filter(
                dem_data, np.nanmean, size=window_size, mode='nearest', output=np.float32)
            dem_data[mask] = filled_data[mask]
            # Update the mask for remaining no-data values
            mask = np.where(dem_data == nodata, True, False)
            # Optionally increase the window size for subsequent iterations
            window_size += 2
        return dem_data