import os
import shutil
import datetime
import numpy as np
import pandas as pd
import pickle
import re
import glob
import sys

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
                "Temporary file cleanup process completed WITH ERRORS. Some files/directories may not have been deleted. Please check logs above.\n",
                self.message_length
            )
        return overall_success

class Utils:
    @staticmethod
    def remove_additional_files(directory, exts=[".ovr", ".cpg", ".dbf", ".tfw", ".xml"]):
        """Recursively remove additional files with specified extensions in directory and subdirectories."""
        removed_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(e) for e in exts):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        removed_files.append(file_path)
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")
        if removed_files:
            print(f"Removed files: {removed_files}")
        else:
            print("No additional files found to remove.")

    @staticmethod
    def time_convert(sec):
        mins = sec // 60
        sec = sec % 60
        hours = mins // 60
        mins = mins % 60
        print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

    @staticmethod
    def find_rasters(base_dir, abrv, resolutions, type_marker):
        """
        Helper to construct paths and return only those that exist.
        Assumes folder structure: base_dir \ resolution \ abrv_type_resolution.tif
        """
        found_files = []
        for res in resolutions:
            # Construct filename: e.g., MP_BY_1m.tif
            filename = f"{abrv}_{type_marker}_{res}.tif"
            full_path = os.path.join(base_dir, res, filename)
            
            if os.path.exists(full_path):
                found_files.append(full_path)
            else:
                # Optional: Print info if a resolution is expected but missing
                # print(f"Info: Skipping missing file: {filename}")
                pass
                
        return found_files

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
    
    @staticmethod
    def sanitize_path_to_name(path):
        """Replace invalid characters and get the base file name from a path."""
        name = os.path.splitext(os.path.basename(path))[0]
        return name.replace(".", "_").replace(" ", "_")
    
    @staticmethod
    def add_src_to_path():
        # Ensure src is on sys.path regardless of working directory
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "src"))
        if SRC_DIR not in sys.path:
            sys.path.insert(0, SRC_DIR)

    @staticmethod
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

    @staticmethod
    def chunk_and_move_files(file_list, source_dir, chunk_size_gb=10):
        dest_dir = source_dir  # Destination chunks live here too

        chunk_size_bytes = chunk_size_gb * 1024**3
        current_chunk = []
        current_size = 0
        chunk_count = 1

        for file_name in file_list:
            file_path = os.path.join(source_dir, file_name)
            file_size = os.path.getsize(file_path)

            if current_size + file_size > chunk_size_bytes and current_chunk:
                chunk_dir = os.path.join(dest_dir, f"chunk_{chunk_count}")
                os.makedirs(chunk_dir, exist_ok=True)

                for f in current_chunk:
                    shutil.move(os.path.join(source_dir, f), os.path.join(chunk_dir, f))

                chunk_count += 1
                current_chunk = []
                current_size = 0

            current_chunk.append(file_name)
            current_size += file_size

        if current_chunk:
            chunk_dir = os.path.join(dest_dir, f"chunk_{chunk_count}")
            os.makedirs(chunk_dir, exist_ok=True)
            for f in current_chunk:
                shutil.move(os.path.join(source_dir, f), os.path.join(chunk_dir, f))

    @staticmethod
    def load_pickle(pickle_pth): #unpickling
        with open(pickle_pth, "rb") as fp:   
            pf = pickle.load(fp)
        return pf
    
    @staticmethod
    def dump_pickle(pickle_pth, pf): #pickling
        with open(pickle_pth, "wb") as fp:
            pickle.dump(pf, fp)

    @staticmethod
    def list_files(filepath, extensions=[]):
        """
        List all files in a directory (recursively) that match any of the given extensions.
        """
        paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    paths.append(os.path.join(root, file))
        return paths
    
    @staticmethod
    def list_subfolders(directory):
        entries = glob.glob(os.path.join(directory, "*"))
        return [os.path.abspath(entry) for entry in entries if os.path.isdir(entry)]

    @staticmethod
    def create_empty_txt_files(filename_list, ext=".txt"):
        for fil in filename_list:
            file = fil + f"{ext}"
            with open(file, 'w'):
                continue

    @staticmethod
    def make_datetime_folder():
        t = datetime.datetime.now()
        timestring = f"{t.year:02d}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"
        Ymmdd = timestring.split("-")[0]
        out_folder = f"2019-2023-{Ymmdd}"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        print(out_folder)
        return out_folder, Ymmdd
    
    @staticmethod
    def natural_sort_key(item, filepath):
        is_dir = os.path.isdir(os.path.join(filepath, item)) # You'd need a real filepath here

        # Natural sort for the name
        def natural_sort_parts(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]

        # Prioritize folders (False for files, True for folders), then apply natural sort
        return (not is_dir, natural_sort_parts(item))
    
    @staticmethod
    def natural_sort(filepath):
        file_list = [
            f for f in os.listdir(filepath)
            if f.endswith('.tif') and not f.endswith('.tif.ovr') and not f.endswith('.tif.aux.xml')
        ]
        return sorted(file_list, key=lambda item: Utils.natural_sort_key(item, filepath))
    
    @staticmethod
    def sharepoint_exact_sort_key(filename):
        # Pattern to extract the two numbers and left-pad the second number with zeros
        # to ensure it is always two digits.
        # and lexographically sort by the combined key
        # Example filename: "tile-123-4.tif" should become "12340"
        match = re.search(r'tile+-(\d+)-(\d+)\.tif$', filename)
        if match:
            group1_int = int(match.group(1)) # Convert to int
            group2_int = int(match.group(2).ljust(2,'0')) # Convert to int
            padded_key = f"{group1_int}{group2_int}"
            return padded_key
    # example usage
    # file_list = os.listdir("path/to/your/directory")  # Replace with your directory path
    # sorted_files = sorted(file_list, key=sharepoint_exact_sort_key)
    # sorted_files