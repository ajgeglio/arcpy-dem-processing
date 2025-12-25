import rasterio
import os, re, time, json, glob, shutil, pickle, tempfile, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from joblib import Parallel, delayed
import pyproj
import arcpy
from arcpy.sa import *
from osgeo import gdal, osr

class ReturnTime:
    def __init__(self) -> None:
        pass
    def get_time_obj(self, time_s):
        # return datetime.datetime.fromtimestamp(time_s)
        return datetime.datetime.fromtimestamp(time_s) if pd.notnull(time_s) else np.nan
    def get_Y(self, time_s):
        return self.get_time_obj(time_s).strftime('%Y') if pd.notnull(time_s) else np.nan
    def get_m(self, time_s):
        return self.get_time_obj(time_s).strftime('%m') if pd.notnull(time_s) else np.nan
    def get_d(self, time_s):
        return self.get_time_obj(time_s).strftime('%d') if pd.notnull(time_s) else np.nan
    def get_t(self, time_s):
        return self.get_time_obj(time_s).strftime('%H:%M:%S') if pd.notnull(time_s) else np.nan

class Utils:
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

    def list_files(filepath, filetype):
        paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.lower().endswith(filetype.lower()):
                    paths.append(os.path.join(root, file))
        return(paths)

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

class GetCoordinates:
    def __init__(self) -> None:
        pass

    def reef_overlap(df_tiffs, OP_TABLE_path):
        OP_TABLE_df = pd.read_excel(OP_TABLE_path)
        OP_TABLE_df = OP_TABLE_df[["COLLECT_ID", "OP_DATE", "SURVEY123_NAME", "REEF_NAME", "LATITUDE", "LONGITUDE"]]
        df = OP_TABLE_df.copy()
        apx = (1/60)/2
        out_df = pd.DataFrame(columns=["COLLECT_ID", "OP_DATE", "SURVEY123_NAME", "REEF_NAME", "LATITUDE", "LONGITUDE", "REEF"])
        for i in range(len(df_tiffs)):
            reef = df_tiffs.loc[i]
            center_lat = reef.avr_lat
            center_lon = reef.avr_lon
            radius = np.sqrt((reef.max_lat - center_lat)**2 + (reef.max_lon - center_lon)**2) + apx
            distance = np.sqrt((df.LATITUDE - center_lat)**2 + (df.LONGITUDE - center_lon)**2)
            within_radius = distance <= radius
            idx = df[within_radius].index
            new_df = df.loc[idx]
            new_df['REEF'] = reef.Reef_Name
            new_df['DIST'] = distance
            out_df = pd.concat([out_df, new_df])
        return out_df[["COLLECT_ID", "OP_DATE", "SURVEY123_NAME", "REEF_NAME", "LATITUDE", "LONGITUDE", "REEF", "DIST"]]
    
    def get_projection(self, tif_file):
        ds = gdal.Open(tif_file)
        cs = osr.SpatialReference()
        return cs.ImportFromWkt(ds.GetProjectionRef())

    def get_min_max_xy(self, tif_file):
        ds = gdal.Open(tif_file)
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(ds.GetProjectionRef())
        # create the new coordinate system
        wgs84_wkt = """GEOGCS["WGS 84", DATUM["WGS_1984", SPHEROID["WGS 84", 6378137, 298.257223563, 
                        AUTHORITY["EPSG","7030"]], AUTHORITY["EPSG","6326"]], PRIMEM["Greenwich",0, 
                        AUTHORITY["EPSG","8901"]], UNIT["degree",0.01745329251994328, 
                        AUTHORITY["EPSG","9122"]], AUTHORITY["EPSG","4326"]]"""
        new_cs = osr.SpatialReference()
        new_cs .ImportFromWkt(wgs84_wkt)
        # create a transform object to convert between coordinate systems
        transform = osr.CoordinateTransformation(old_cs,new_cs) 
        #get the point to transform, pixel (0,0) in this case
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        miny = gt[0]
        minx = gt[3] + width*gt[4] + height*gt[5] 
        maxy = gt[0] + width*gt[1] + height*gt[2] 
        maxx = gt[3]
        #get the coordinates in lat long
        return transform.TransformPoint(miny,minx)[0:2], transform.TransformPoint(maxy,maxx)[0:2]
    
    def convert_tracklines_to_lat_lon(xyd_file, from_wkt, wgs84_wkt):
    
        with open(from_wkt) as f:
            from_wkt = f.read()
        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(from_wkt)
        
        with open(wgs84_wkt) as f:
            wgs84_wkt = f.read()
        new_cs = osr.SpatialReference()
        new_cs.ImportFromWkt(wgs84_wkt)
        
        # create a transform object to convert between coordinate systems
        # get the GeoTransform parameters from the WKT files
        transform = osr.CoordinateTransformation(old_cs, new_cs)
        gt = [0, 1, 0, 0, 0, -1]  # Assuming a simple identity transform for this example
        
        # convert the coordinates in xyd_file.X, xyd_file.Y to lat long
        lat_lon_coords = []
        for index, row in xyd_file.iterrows():
            x = row['X']
            y = row['Y']
            # Apply the GeoTransform to get the coordinates in the source projection
            x_geo = gt[0] + x * gt[1] + y * gt[2]
            y_geo = gt[3] + x * gt[4] + y * gt[5]
            # Transform the coordinates to the new projection
            lat, lon, _ = transform.TransformPoint(x_geo, y_geo)
            lat_lon_coords.append((lon, -lat))
        
        return lat_lon_coords
    
    # Function to get the coordinate system of the tif file
    def get_tif_coordinate_system(tif_file):
        ds = gdal.Open(tif_file)
        proj_wkt = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)
        return srs

    # Function to infer the coordinate system of the xyd_file
    def infer_xyd_coordinate_system(xyd_file):
        # Assuming the coordinates are in UTM zone 16N (you may need to adjust this based on your region)
        utm_zone = 16
        northern_hemisphere = True
        proj_str = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
        return pyproj.CRS(proj_str)

    def return_min_max_tif_df(self, tif_files=None):
        reef_dict = {"BH": "Bay Harbor", "CH": "Cathead Point", "ER": "Elk Rapids", "FI": "Fishermans Island", "IP":"Ingalls Point",
                "LR": "Lees Reef", "MS": "Manistique Shoal", "MP": "Mission Point", "MR":"Mudlake Reef", "NPNE": "Northport Bay NE",
                "NPNW":"Northport Bay NW", "NPS":"Northport Bay S", "NP": "Northport Point", "SP": "Suttons Point", 
                "TC":"Tannery Creek", "TS":"Traverse Shoal", "TP":"Tuckers Point", "WP": "Wiggins Point"}
        min_max = [self.get_min_max_xy(t) for t in tif_files]
        names = [os.path.basename(t) for t in tif_files]
        # reefs = [os.path.basename(os.path.dirname(t)) for t in tif_files]
        reefs = [n.split("_")[0] for n in names]
        min_y, min_x = [min_max[i][0][0] for i in range(len(min_max))], [min_max[i][0][1] for i in range(len(min_max))]
        max_y, max_x  = [min_max[i][1][0] for i in range(len(min_max))], [min_max[i][1][1] for i in range(len(min_max))]
        min_max_df = pd.DataFrame(np.c_[names, min_y, min_x, max_y, max_x], columns=["filename", "min_lat", "min_lon", "max_lat", "max_lon"])
        min_max_df.min_lat = min_max_df.min_lat.astype('float')
        min_max_df.min_lon = min_max_df.min_lon.astype('float')
        min_max_df.max_lat = min_max_df.max_lat.astype('float')
        min_max_df.max_lon = min_max_df.max_lon.astype('float')
        min_max_df["avr_lat"] = (min_max_df.min_lat + min_max_df.max_lat)/2
        min_max_df["avr_lon"] = (min_max_df.min_lon + min_max_df.max_lon)/2
        min_max_df["reef"] = reefs
        min_max_df["filepath"] = tif_files
        min_max_df = min_max_df.drop_duplicates(subset="reef").reset_index(drop=True)
        min_max_df["Reef_Name"] = min_max_df.reef.apply(lambda x: reef_dict[x])
        return min_max_df
    
    def log_col_lat_lon(self, root):
        out_df = pd.DataFrame()
        for collect_id in os.listdir(root):
            logpath = r"logs\*\Logs\*.log"
            folder = os.path.join(root,collect_id,logpath)
            logs = glob.glob(folder)
            log_df = pd.DataFrame()
            for log in logs:
                log_df_t = pd.read_csv(log, delimiter=";")
                log_df = pd.concat([log_df, log_df_t])
            try:
                med_lat, med_lon = log_df.Latitude.median(), log_df.Longitude.median()
            except:
                med_lat, med_lon = np.nan, np.nan
            df = pd.DataFrame([[collect_id, med_lat, med_lon]], columns=["collect_id", "Latitude", "Longitude"])
            out_df = pd.concat([out_df, df])
        return out_df
    
    def return_headers_min_max_coord(self, df):
        collects = df.collect_id.unique()
        coords = []
        for c in collects:
            df_tmp = df[df.collect_id == c]
            df_tmp = df_tmp[df_tmp.Usability == "Usable"]
            lats = df_tmp.Latitude.dropna()
            lons = df_tmp.Longitude.dropna()
            try:
                # list1 = [lats.min(), lons.min(), lats.max(), lons.max()]
                list = [np.percentile(lats, 10), np.percentile(lons, 10), np.percentile(lats, 90), np.percentile(lons, 90), np.percentile(lats, 50), np.percentile(lons, 50)]
            except IndexError:
                print("not lat lon in ", c)
                list = [np.nan]*6
            coords.append(list)
        collects_lat_lon = pd.DataFrame(np.c_[collects, coords], columns=["collect_id", "min_lat", "min_lon", "max_lat", "max_lon", "med_lat", "med_lon"])
        return collects_lat_lon
    
    def return_MissionLog_min_max_coord(self, collect):
        log_paths = glob.glob(os.path.join(collect,'logs','*','Logs','*.log'))
        try:
            dfs = [pd.read_csv(file, header=0, delimiter=';').dropna(axis=1) for file in log_paths]
            log_df = pd.concat(dfs, axis=0, ignore_index=True)
            lats = log_df['Latitude'].values
            lons = log_df['Longitude'].values
            lats = lats[(lats<50) & (lats>41)] # Sometimes there are erroneous lat lon values in the .log files
            lons = lons[(lons>-92.5) & (lons<-75.5)]
            min_lat, min_lon = lats.min(), lons.min()
            max_lat, max_lon = lats.max(), lons.max()
        except:
            min_lat, min_lon, max_lat, max_lon = np.nan, np.nan, np.nan, np.nan
        return min_lat, min_lon, max_lat, max_lon
    
    def return_MissionLog_min_max_df(self, collect_list):
        items = []
        for collect in collect_list:
            min_lat, min_lon, max_lat, max_lon = self.return_MissionLog_min_max_coord(collect)
            item = [collect, min_lat, min_lon, max_lat, max_lon]
            items.append(item)
        df = pd.DataFrame(items, columns=["collect_path", "min_lat", "min_lon", "max_lat", "max_lon"])
        cid = lambda x: os.path.basename(x)
        d = lambda x: x.split('_')[0]
        cn = lambda x: x.split('_')[1]
        iv = lambda x: x.split('_')[2]
        cs = lambda x: x.split('_')[3]
        df["collect_id"] = df.collect_path.map(cid)
        df["date"] = df.collect_id.apply(d)
        df["date"] = pd.to_datetime(df.date, format="%Y%m%d")
        df["collect"] = df.collect_id.apply(cn)
        df["AUV"] = df.collect_id.apply(iv)
        df["cam_sys"] = df.collect_id.apply(cs)
        df = df[["collect_path", "collect_id", "date", "collect", "AUV", "cam_sys", "min_lat", "min_lon", "max_lat", "max_lon"]]
        return df 

class HabitatDerivatives:
    def __init__(self, use_gdal=True, use_rasterio=False, chunk_size=None) -> None:
        """
        Initialize the HabitatDerivatives class.

        Parameters:
        use_gdal (bool): Flag to indicate whether to use GDAL for processing. Default is True.
        use_rasterio (bool): Flag to indicate whether to use Rasterio for processing. Default is False.
        chunk_size (int or None): Size of chunks for processing large DEMs. Default is None (no chunking).
        """
        self.chunk_size = chunk_size
        self.use_gdal = use_gdal
        self.use_rasterio = use_rasterio

    def calculate_lbp(self, dem_data, nodata, n_points, radius, method='default'):
        """
        Generate a Local Binary Pattern (LBP) representation from a DEM using a sliding window approach.

        Parameters:
        dem_data(numpy.ndarray): Input DEM as a 2D NumPy array. The DEM should be normalized to a specific range (e.g., 0-255) 
                             and must not contain NaN values. Ensure the data type is compatible with NumPy operations (e.g., float32 or int32).

        Parameters:
        dem_data(numpy.ndarray): Input DEM as a 2D NumPy array.
        radius (int): Radius of LBP neighborhood.
        n_points (int): Number of circularly symmetric neighbor points.
        method (str): Method to compute LBP ('default', 'uniform', etc.).

        Returns:
        numpy.ndarray: LBP-transformed DEM with the same shape.
        """
        # Mask no-data values
        if nodata is not None:
            dem_data = np.ma.masked_where((dem_data == nodata) | np.isnan(dem_data), dem_data)

        # Normalize the DEM data to the range 0-255
        min_val = dem_data.min()
        max_val = dem_data.max()
        if (max_val > min_val):  # Avoid division by zero
            normalized_data = ((dem_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            normalized_data = np.zeros_like(dem_data, dtype=np.uint8)

        # Compute LBP for the entire DEM
        lbp = local_binary_pattern(normalized_data, P=n_points, R=radius, method=method)

        return lbp
    
    def generate_hillshade_gdal(self, dem_file, hillshade_output):
        """
        Calculate hillshade using GDAL.DEMProcessing with multiple options.
        
        :param dem_file: Path to the input DEM file.
        :param hillshade_output: Path to save the hillshade output file.
        """
        # Set GDAL DEMProcessing options
        options = gdal.DEMProcessingOptions(
            computeEdges=True,          # Compute edges for smoother output
            azimuth=315,                # Direction of light source (in degrees)
            altitude=45,                # Elevation of the light source (in degrees)
            scale=1.0,                  # Scale factor for vertical exaggeration
            zFactor=1.0                 # Vertical exaggeration
        )
        
        # Perform hillshade processing
        gdal.DEMProcessing(
            destName=hillshade_output,  # Output file path
            srcDS=dem_file,             # Input DEM file
            processing="hillshade",     # Specify hillshade calculation
            options=options             # Pass multiple options
        )
        print(f"Hillshade file saved at: {hillshade_output}")

    def calculate_hillshade(self, dem_file, hillshade_output):
        """
        Calculate hillshade for a DEM file and save the output.

        Parameters:
        dem_file (str): Path to the input DEM file.
        hillshade_output (str): Path to save the hillshade output file.
        """
        if self.use_gdal:
            self.generate_hillshade_gdal(dem_file, hillshade_output)
        else:
            raise NotImplementedError("Custom hillshade calculation is not implemented for non-GDAL methods.")

    def calculate_slope_gdal(self, dem_data, transform):
        """ Calculate slope from a DEM using GDAL."""
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate slope
        slope_ds = gdal.DEMProcessing('', src_ds, 'slope', format='MEM', options=['-compute_edges'])# options = ['-zero_for_flat'] option can be added to calculate edges
        slope = slope_ds.GetRasterBand(1).ReadAsArray()
        return slope
    
    def calculate_aspect_gdal(self, dem_data, transform):
        """ Calculate slope and aspect from a DEM using GDAL. 
        This array contains the aspect values calculated from the Digital Elevation Model (DEM) using the GDAL library.
        Aspect values represent the compass direction that the slope faces. The values are typically in degrees, where:
        0° represents north,         180° represents south,        270° represents west."""
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate aspect
        aspect_ds = gdal.DEMProcessing('', src_ds, 'aspect', format='MEM', options=['-compute_edges'])
        aspect = aspect_ds.GetRasterBand(1).ReadAsArray()
        return aspect

    def generate_slope_gdal(self, input_dem, output_slope):
        """
        Calculate slope and aspect from a DEM using GDAL and save them as separate files.

        Parameters:
        input_dem (str): Path to the input DEM file.
        output_slope (str): Path to save the slope output file.
        output_aspect (str): Path to save the aspect output file.
        """
        # Calculate slope and save to output file
        gdal.DEMProcessing(
            destName=output_slope,
            srcDS=input_dem,
            processing="slope",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"Slope file saved at: {output_slope}")

    def calculate_slope(self, dem_data, transform, input_dem, output_slope):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_slope_gdal(dem_data, transform)
            else:
                self.generate_slope_gdal(input_dem, output_slope)
        else:
            return self.calculate_slope_aspect_skimage(dem_data)[0]
        
    def generate_aspect_gdal(self, input_dem, output_aspect):
        # Calculate aspect and save to output file
        gdal.DEMProcessing(
            destName=output_aspect,
            srcDS=input_dem,
            processing="aspect",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"Aspect file saved at: {output_aspect}")

    def calculate_slope_aspect_skimage(self, dem_data):
        """ Calculate slope and aspect from a DEM using skimage. """
        # Calculate gradient in x and y directions using sobel gradient
        gradient_x = ndimage.sobel(dem_data, 0)  # horizontal gradient
        gradient_y = ndimage.sobel(dem_data, 1)  # vertical gradient
        # Calculate slope
        cell_size = 9  # Replace with the actual cell size of your DEM
        slope = np.arctan(np.sqrt((gradient_x / cell_size)**2 + (gradient_y / cell_size)**2))
        slope = np.degrees(slope)  # convert radians to degrees
        # Calculate aspect
        aspect = np.arctan2(-gradient_y, gradient_x)
        # Convert aspect from radians to degrees
        aspect = np.degrees(aspect)
        # Convert aspect to compass direction in degrees azimuth
        aspect = (aspect) % 360
        return slope, aspect

    def calculate_aspect(self, dem_data, transform, input_dem, output_aspect):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_aspect_gdal(dem_data, transform)
            else:
                self.generate_aspect_gdal(input_dem, output_aspect)
        else:
            return self.calculate_slope_aspect_skimage(dem_data)[1]

    def calculate_roughness_gdal(self, dem_data, transform):
        """ Calculate terrain roughness using GDAL. """
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        # src_ds.GetRasterBand(1).SetNoDataValue(-9999)  # Set no-data value
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        
        # Calculate roughness
        roughness_ds = gdal.DEMProcessing('', src_ds, 'Roughness', format='MEM', options = ['-compute_edges'])
        roughness = roughness_ds.GetRasterBand(1).ReadAsArray()
        
        return roughness
    
    def generate_roughness_gdal(self, input_dem, output_roughness):
        """ Calculate terrain roughness using GDAL. """
        # Calculate roughness and save to output file
        gdal.DEMProcessing(
            destName=output_roughness,
            srcDS=input_dem,
            processing="roughness",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"Rougness file saved at: {output_roughness}")

    def calculate_roughness_skimage(self, dem_data):
        """ Calculate terrain roughness as the standard deviation. """
        return ndimage.generic_filter(dem_data, np.std, size=16)

    def calculate_roughness(self, dem_data, transform, input_dem, output_roughness):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_roughness_gdal(dem_data, transform)
            else:
                self.generate_roughness_gdal(input_dem, output_roughness)
        else:
            return self.calculate_roughness_skimage(dem_data)

    def calculate_tpi_gdal(self, dem_data, transform):
        """
        Calculate Topographic Position Index (TPI) using GDAL.
        TPI is a measure of the relative position of each cell in a digital elevation model (DEM) 
        compared to its surrounding cells. It is used to identify landforms such as ridges, valleys, 
        and flat areas. Positive TPI values indicate that the cell is higher than its surroundings 
        (e.g., ridges), negative values indicate that the cell is lower than its surroundings 
        (e.g., valleys), and values close to zero indicate flat areas.
        Parameters:
        dem_data(numpy.ndarray): Input DEM as a 2D NumPy array.
        
        Returns:
        numpy.ndarray: TPI-transformed DEM with the same shape.
        """
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        # src_ds.GetRasterBand(1).SetNoDataValue(-9999)  # Set no-data value-data value
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        
        # Calculate mean elevation of the neighbors
        tpi_ds = gdal.DEMProcessing('', src_ds, 'TPI', format='MEM', options = ['-compute_edges']) #
        tpi = tpi_ds.GetRasterBand(1).ReadAsArray()
        return tpi
    
    def generate_tpi_gdal(self, input_dem, output_tpi):
        """ Calculate terrain roughness using GDAL. """
        # Calculate roughness and save to output file
        gdal.DEMProcessing(
            destName=output_tpi,
            srcDS=input_dem,
            processing="tpi",
            options=gdal.DEMProcessingOptions(computeEdges=True)
        )
        print(f"TPI file saved at: {output_tpi}")

    def calculate_tpi_skimage(self, dem_data):
        """
        mean_neighbors = ndimage.generic_filter(dem, np.mean, size=self.window_size)  # Mean elevation of the neighbors
        Parameters:        dem_data(numpy.ndarray): Input DEM as a 2D NumPy array.

        Returns:
        numpy.ndarray: TPI-transformed DEM with the same shape.
        """
        mean_neighbors = ndimage.generic_filter(dem_data, np.mean, size=16)  # Mean elevation of the neighbors
        # mean_neighbors = ndimage.uniform_filter(dem, size=16)  # Mean elevation of the neighbors
        tpi = dem_data - mean_neighbors  # Calculate TPI
        return tpi

    def calculate_tpi(self, dem_data, transform, input_dem, output_tpi):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_tpi_gdal(dem_data, transform)
            else:
                self.generate_tpi_gdal(input_dem, output_tpi)
        else:
            return self.calculate_tpi_skimage(dem_data)

    def calculate_tri_gdal(self, dem_data, transform):
        """ Calculate Terrain Ruggedness Index (TRI) using GDAL. """
        # Create a temporary in-memory file for the DEM
        mem_drv = gdal.GetDriverByName('MEM')
        src_ds = mem_drv.Create('', dem_data.shape[1], dem_data.shape[0], 1, gdal.GDT_Float32)
        src_ds.SetGeoTransform(transform.to_gdal())  # Set geotransformation
        src_ds.GetRasterBand(1).WriteArray(dem_data)
        # Calculate TRI
        tri_ds = gdal.DEMProcessing('', src_ds, 'TRI', format='MEM', options=['-compute_edges'])
        tri = tri_ds.GetRasterBand(1).ReadAsArray()
        return tri
    
    def generate_tri_gdal(self, input_dem, output_tri):
        """
        Generate Terrain Ruggedness Index (TRI) from an input DEM using GDAL.

        :param input_dem: Path to the input DEM file.
        :param output_tri: Path to save the output TRI file.
        """
        # Use GDALDEMProcessing to generate TRI
        options = gdal.DEMProcessingOptions(creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'])
        gdal.DEMProcessing(output_tri, input_dem, 'TRI', options=['-compute_edges'], creationOptions=None)
        print(f"TRI file saved at: {output_tri}")

    def calculate_tri_skimage(self, dem_data, size=3):
        tri = np.empty_like(dem_data, dtype=np.float32)
        # Calculate the mean elevation of the neighbors
        tri = ndimage.generic_filter(dem_data, lambda window: np.sqrt(np.sum((window - window[4])**2)), size)
        return tri

    def calculate_tri(self, dem_data, transform, input_dem, output_tri):
        if self.use_gdal:
            if self.use_rasterio:
                return self.calculate_tri_gdal(dem_data, transform)
            else:
                self.generate_tri_gdal(input_dem, output_tri)
        else:
            return self.calculate_tri_skimage(dem_data)
    
    def calculate_flow_direction(self, dem_data, nodata=-9999):
        """
        Calculate the Flow Direction similar to ArcGIS (Spatial Analyst).
        https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-flow-direction-works.htm
        """

        # Define the direction encoding
        direction_encoding = np.array([[32, 64, 128],
                                        [16, 0, 1],
                                        [8, 4, 2]])

        # Initialize the flow direction array
        flow_direction = np.zeros(dem_data.shape, dtype=np.int32)

        def process_window(window):
            """Process a 3x3 window to calculate flow direction."""
            window = np.where(window == nodata, np.nan, window)  # Replace -9999 with NaN
             # Check if the window is empty (all NaN)
            if np.isnan(window).any():
                return 0  # No flow direction if all values are NaN or -9999
            diff = window[1, 1] - window  # Differences between center and neighbors
            max_diff = np.max(diff)
            return direction_encoding[diff == max_diff].sum() if max_diff > 0 else 0

        # Apply the sliding window approach
        flow_direction[1:-1, 1:-1] = np.array([
            process_window(dem_data[i - 1:i + 2, j - 1:j + 2])
            for i in range(1, dem_data.shape[0] - 1)
            for j in range(1, dem_data.shape[1] - 1)
        ]).reshape(dem_data.shape[0] - 2, dem_data.shape[1] - 2)

        return flow_direction

    def calculate_shannon_index_2d(self, dem_data, nodata, window_size=9):
        """
        Calculate the Shannon diversity index for each window in a 2D grid.

        Parameters:
        dem_data (numpy.ndarray): A 2D numpy array representing the DEM data.
        window_size (int): Size of the sliding window.

        Returns:
        numpy.ndarray: A 2D numpy array of Shannon diversity indices, with dimensions
                       (rows - window_size + 1, cols - window_size + 1).

        The Shannon diversity index is calculated for each window of size `window_size` x `window_size`
        in the input grid. The function slides the window across the grid and computes the index
        based on the unique values and their probabilities within each window.
        """
        def calculate_window_entropy(window):
            """Calculate entropy for a given window."""
            _, counts = np.unique(window, return_counts=True)
            probabilities = counts / counts.sum()
            return entropy(probabilities)

        # Calculate flow direction
        flow_direction = self.calculate_flow_direction(dem_data)
        rows, cols = flow_direction.shape
        shannon_indices = np.zeros((rows - window_size + 1, cols - window_size + 1))
        
        if self.is_low_variance(flow_direction, nodata, n=1):
            return shannon_indices
        else:
            # Use parallel processing to calculate entropy for each window
            results = Parallel(n_jobs=-1)(
                delayed(calculate_window_entropy)(
                    flow_direction[i:i + window_size, j:j + window_size]
                )
                for i in range(rows - window_size + 1)
                for j in range(cols - window_size + 1)
            )

            # Populate the Shannon indices array with the results
            for idx, value in enumerate(results):
                i = idx // (cols - window_size + 1)
                j = idx % (cols - window_size + 1)
                shannon_indices[i, j] = value
            # Trim the edges to avoid artifacts
            trim_size = window_size // 2
            trimmed_shannon_indices = shannon_indices[trim_size:-trim_size, trim_size:-trim_size]
            return trimmed_shannon_indices
    
    def return_dem_data(self, dem_data):
        return dem_data

    def is_empty(self, dem_data, nodata):
        """
        Function to check if the cleaned DEM data is empty.
        This function removes NaN values, -9999, 0, and infinite values from the DEM data."""
        # Remove NaN values, -9999, 0, and infinite values
        cleaned_data = dem_data[~(np.isnan(dem_data)) & ~(dem_data == nodata) & ~(dem_data == 0) & ~(np.isinf(dem_data))]
        return cleaned_data.size == 0
    
    def is_low_variance(self, dem_data, nodata, n=2):
        """
        Function to check if the number of unique values in the cleaned data array are less than or equal to 2.

        Returns:
        bool: True if the unique values array length is <= 2, False otherwise.
        """
        cleaned_data = dem_data[~(np.isnan(dem_data)) & ~(dem_data == nodata) & ~(dem_data == 0) & ~(np.isinf(dem_data))]
        return len(np.unique(cleaned_data)) <= n
    
    def merge_dem(self, dem_chunks_path, remove=True):
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

    def reduce_and_compress_dem(self, dem_path):
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
                'nodata': -9999,
                'BIGTIFF': 'YES'
            })

            # Define output file path
            output_path = dem_path.split(".")[0] + "_compressed.tif"

            # Write the modified DEM to a new file in chunks
            with rasterio.open(output_path, 'w', **metadata) as dst:
                for ji, window in src.block_windows(1):  # Process by blocks
                    dem_data = src.read(1, window=window)
                    dem_data = np.where(np.isnan(dem_data) | np.isinf(dem_data) | (dem_data < -9999), -9999, dem_data)
                    dst.write(dem_data.astype('float32'), 1, window=window)
        print(f"Reduced and compressed DEM saved to {output_path}")

    def convert_tiff_to_gdal_raster(self, input_tiff, compress=True):
        """
        Convert a TIFF file to a GDAL raster TIFF using gdal_translate.
        
        :param input_tiff: Path to the input TIFF file.
        :param output_tiff: Path to save the output raster TIFF file.
        """
        # Open the input TIFF file
        src_ds = gdal.Open(input_tiff)
        output_tiff = input_tiff.split(".")[0]+"_gdal.tif"
        # Use gdal_translate to convert the file
        if compress:
            gdal.Translate(output_tiff, src_ds, format='GTiff', creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES'], outputType=gdal.GDT_Float32)
        else:
            gdal.Translate(output_tiff, src_ds, format='GTiff', outputType=gdal.GDT_Float32)
        # Remove the original input TIFF file
        gdal.Dataset.__swig_destroy__(src_ds)  # Close the dataset
        del src_ds  # Delete the dataset reference
        os.remove(input_tiff)
        print(f"Converted {input_tiff} to {output_tiff} with compression={compress} as GDAL raster TIFF.")
        return output_tiff
    
    def fill_nodata(self, dem_data, nodata=-9999, max_iterations=10, initial_window_size=3):
        """
        Fill no-data values in the DEM data with the mean of the surrounding values.

        Parameters:
        dem_data (numpy.ndarray): Input DEM data as a 2D NumPy array.
        nodata (int): Value representing no-data in the DEM.
        max_iterations (int): Maximum number of iterations to fill gaps.
        initial_window_size (int): Initial size of the sliding window.

        Returns:
        numpy.ndarray: DEM data with no-data values filled.
        """
        # Create a mask for no-data values
        mask = np.where(dem_data == nodata, True, False)
        window_size = initial_window_size

        for iteration in range(max_iterations):
            if not np.any(mask):
                break  # Exit if no gaps remain

            # Fill no-data values with the mean of surrounding values
            filled_data = ndimage.generic_filter(
                dem_data, np.nanmean, size=window_size, mode='nearest', output=np.float32
            )
            dem_data[mask] = filled_data[mask]

            # Update the mask for remaining no-data values
            mask = np.where(dem_data == nodata, True, False)

            # Optionally increase the window size for subsequent iterations
            window_size += 2

        return dem_data

    def process_dem(self, 
                    input_dem,
                    shannon_window=9,
                    output_slope=None, 
                    output_aspect=None, 
                    output_roughness=None, 
                    output_tpi=None, 
                    output_tri=None,
                    output_hillshade=None, 
                    output_shannon_index=None, 
                    output_lbp_3_1=None, 
                    output_lbp_15_2=None, 
                    output_lbp_21_3=None,
                    output_dem=None):
        """ Read a DEM and compute slope, aspect, roughness, TPI, and TRI. Output each to TIFF files based on user input. """
        def generate_products():
            if output_slope:
                yield self.calculate_slope(dem_data, transform, input_dem, output_slope), output_slope   # Slope
            if output_aspect:
                yield self.calculate_aspect(dem_data, transform, input_dem, output_aspect), output_aspect  # Aspect
            if output_roughness:
                yield self.calculate_roughness(dem_data, transform, input_dem, output_roughness), output_roughness  # Roughness
            if output_tpi:
                yield self.calculate_tpi(dem_data, transform, input_dem, output_tpi), output_tpi  # TPI
            if output_tri:
                yield self.calculate_tri(dem_data, transform, input_dem, output_tri), output_tri  # TRI
            if output_hillshade:
                yield self.calculate_hillshade(input_dem, output_hillshade), output_hillshade # Hillshade
            if output_shannon_index:
                yield self.calculate_shannon_index_2d(dem_data, nodata, window_size=shannon_window), output_shannon_index  # Shannon Index
            if output_lbp_3_1:
                yield self.calculate_lbp(dem_data, nodata, 3, 1), output_lbp_3_1  # LBP
            if output_lbp_15_2:
                yield self.calculate_lbp(dem_data, nodata, 15, 2), output_lbp_15_2  # LBP
            if output_lbp_21_3:
                yield self.calculate_lbp(dem_data, nodata, 21, 3), output_lbp_21_3  # LBP
            if output_dem:
                yield self.return_dem_data(dem_data), output_dem  # used to chunk and merge data to get the same shape output as other products
        
        if not self.chunk_size:
            with rasterio.open(input_dem) as src:
                dem_data = src.read(1)  # Read the DEM data
                transform = src.transform  # Get the affine transform
                crs = src.crs  # Get the coordinate reference system
                metadata = src.meta  # Get metadata
                nodata = metadata['nodata']  # Get no-data value
                dem_data = self.fill_nodata(dem_data, nodata)  # Fill no-data values
                # Check if the DEM data is empty after filling no-data values
                if self.is_empty(dem_data, nodata):
                    print("The DEM data is empty after filling no-data values.")
                    return None

            # Process and write each product one at a time
            for data, output_file in generate_products():
                if data is not None:
                    with rasterio.open(
                        output_file,
                        'w',
                        driver='GTiff',
                        height=dem_data.shape[0],
                        width=dem_data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=crs,
                        transform=transform
                    ) as dst:
                        dst.write(data, 1)
                        # Write metadata from src into the chunk file
                        dst.update_tags(**src.tags())
                self.convert_tiff_to_gdal_raster(output_file, compress=False)
        
        else:
            with rasterio.open(input_dem) as src:
                transform = src.transform
                crs = src.crs
                metadata = src.meta  # Get metadata
                nodata = metadata['nodata']  # Get no-data value
                dem_data = self.fill_nodata(dem_data, nodata)  # Fill no-data values
                # Check if the DEM data is empty after filling no-data values
                if self.is_empty(dem_data, nodata):
                    print("The DEM data is empty after filling no-data values.")
                    return None
                tile_size = self.chunk_size
                width = src.width
                height = src.height

                for i in range(0, height, tile_size):
                    for j in range(0, width, tile_size):
                        # Read a chunk of the DEM
                        window = rasterio.windows.Window(j, i, min(tile_size, width - j), min(tile_size, height - i))
                        dem_data = src.read(1, window=window)

                        if not self.is_low_variance(dem_data, nodata, n=2):
                            print(f"Skipping chunk at ({i}, {j}) as it has near zero variance.", end="\r")
                            chunk_transform = rasterio.windows.transform(window, transform)

                            # Process and write each product for the chunk
                            for data, output_file in generate_products():
                                if data is None:
                                    print(f"Skipping chunk at ({i}, {j}) for {output_file} as data is None.", end="\r")
                                    continue

                                # Ensure output folder exists for the product
                                output_dir = os.path.dirname(output_file)
                                product_name = os.path.basename(output_file).split(".")[0]
                                product_folder = os.path.join(output_dir, product_name)
                                os.makedirs(product_folder, exist_ok=True)

                                # Define the chunk output file path
                                chunk_output_file = os.path.join(product_folder, f"{product_name}_chunk_{i}_{j}.tif")

                                # Skip processing if the chunk already exists
                                if os.path.exists(chunk_output_file):
                                    print(f"Skipping existing chunk at ({i}, {j}) for {output_file}.", end="\r")
                                    continue

                                # Save the chunk using Rasterio
                                with rasterio.open(
                                    chunk_output_file,
                                    'w',
                                    driver='GTiff',
                                    height=dem_data.shape[0],
                                    width=dem_data.shape[1],
                                    count=1,
                                    dtype=data.dtype,
                                    crs=crs,
                                    transform=chunk_transform
                                ) as dst:
                                    dst.write(data, 1)
                                    # Write metadata from src into the chunk file
                                    dst.update_tags(**src.tags())

                # Merge the tiles and clean up
                for _, output_file in generate_products():
                    product_folder = os.path.join(os.path.dirname(output_file), os.path.basename(output_file).split(".")[0])
                    merged_dem = self.merge_dem(product_folder)
                    self.reduce_and_compress_dem(merged_dem)

class landforms:
    def __init__(self, input_dem_path, geomorphons_directory="geomorphons", local_workspace="local_workspace"):
        self.geomorphons_directory = geomorphons_directory
        self.local_workspace = local_workspace
        self.input_dem = input_dem_path
        
        # Extract DEM name and define output raster paths
        self.input_dem_name = os.path.basename(input_dem_path).split(".")[0]
        self.geomorphons_folder = os.path.join(self.geomorphons_directory, self.input_dem_name)
        self.bathymorphons_raster = f"{self.input_dem_name}_bathymorphons.tif"
        
        # Ensure output and workspace directories exist
        os.makedirs(self.geomorphons_folder, exist_ok=True)
        os.makedirs(self.local_workspace, exist_ok=True)

    def calculate_geomorphon_landforms(self, angle_threshold=1, distance_units="METERS", search_distance=10, skip_distance=5, z_unit="METER"):
        """
        Calculates geomorphons over a specified search distance, skipping cells within a defined distance of the target cell.
        The calculated geomorphons are classified into landforms and saved as a raster.

        Parameters:
        - input_dem (str): Path to the input DEM file.
        - angle_threshold (int): Angle threshold for flat terrain classification.
        - distance_units (str): Units for distance (e.g., "METERS").
        - search_distance (int): Search distance for geomorphon calculation.
        - skip_distance (int): Skip distance for geomorphon calculation.
        - z_unit (str): Units for elevation (e.g., "METER").

        Returns:
        - str: Path to the saved geomorphons raster file.
        """
        # Set environment settings
        arcpy.env.workspace = self.local_workspace

        # Check out the ArcGIS Spatial Analyst extension license
        arcpy.CheckOutExtension("Spatial")

        out_raster = os.path.join(self.geomorphons_folder, f"{self.input_dem_name}_landforms_orig.tif")
        
        # Execute the GeomorphonLandforms tool
        out_geomorphon_landforms = GeomorphonLandforms(
            self.input_dem, self.bathymorphons_raster, angle_threshold, distance_units,
            search_distance, skip_distance, z_unit
        )
        # Save the output raster
        out_geomorphon_landforms.save(out_raster)

    def analyze_raster_data(raster):
        # Open the raster file
        with rasterio.open(raster) as src:
            # Read the raster data as a numpy array
            raster_data = src.read(1)  # Read the first band
            # Get raster metadata
            metadata = src.meta
            # Read the metadata tags
            tags = src.tags()
            print("Tags in the DEM file:")
            print(json.dumps(tags, indent=4))  # Pretty print the tags

        # Mask the nodata value
        masked_data = raster_data[raster_data != metadata['nodata']]
        # Analyze the raster values
        min_value = np.min(masked_data)
        max_value = np.max(masked_data)
        unique_values = len(np.unique(masked_data))

        print(f"Raster Metadata: {metadata}")
        print(f"Minimum Value: {min_value}")
        print(f"Maximum Value: {max_value}")
        print(f"Number of unique values ignoring nodata: {unique_values}")
        # Create the histogram
        plt.hist(masked_data, bins=np.arange(1, 12) - 0.5, color='blue', edgecolor='black')
        plt.xticks(range(1, 11))  # Ensure x-axis is 1-10 by 1
        plt.title("Histogram of Raster Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
    
    def classify_landform_from_bathymorphon(self, number, classes="10c"):
        """
        Function to calculate potential local ternary patterns for a given integer number.
        Then pass the ternary pattern to the lookup table to classify the landform.
        The lookup table is based on the number of minuses and pluses in the ternary pattern.

        Parameters:
        number (int): The input integer number.

        Returns:
        list: A list of potential local ternary patterns with a length of 8.
        """
        combined_lookup_table = {
            "10c": {
                (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "FS", (0, 4): "FS", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "PT",
                (1, 0): "FL", (1, 1): "FL", (1, 2): "FS", (1, 3): "FS", (1, 4): "FS", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
                (2, 0): "FL", (2, 1): "SH", (2, 2): "SL", (2, 3): "SL", (2, 4): "CN", (2, 5): "CN", (2, 6): "VL",
                (3, 0): "SH", (3, 1): "SH", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "CN",
                (4, 0): "SH", (4, 1): "SH", (4, 2): "CV", (4, 3): "SL", (4, 4): "SL",
                (5, 0): "RI", (5, 1): "RI", (5, 2): "CV", (5, 3): "CV",
                (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
                (7, 0): "RI", (7, 1): "RI",
                (8, 0): "PK"
            },
            "6c": {
                (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "FS", (0, 4): "FS", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "VL",
                (1, 0): "FL", (1, 1): "FL", (1, 2): "FS", (1, 3): "FS", (1, 4): "FS", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
                (2, 0): "FL", (2, 1): "SH", (2, 2): "SL", (2, 3): "SL", (2, 4): "SL", (2, 5): "VL", (2, 6): "VL",
                (3, 0): "SH", (3, 1): "SH", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "SL",
                (4, 0): "SH", (4, 1): "SH", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
                (5, 0): "RI", (5, 1): "RI", (5, 2): "RI", (5, 3): "SL",
                (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
                (7, 0): "RI", (7, 1): "RI",
                (8, 0): "RI"
            },
            "5c": {
                (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "SL", (0, 4): "VL", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "VL",
                (1, 0): "FL", (1, 1): "FL", (1, 2): "SL", (1, 3): "SL", (1, 4): "VL", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
                (2, 0): "FL", (2, 1): "SL", (2, 2): "SL", (2, 3): "SL", (2, 4): "SL", (2, 5): "VL", (2, 6): "VL",
                (3, 0): "SL", (3, 1): "SL", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "SL",
                (4, 0): "RI", (4, 1): "RI", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
                (5, 0): "RI", (5, 1): "RI", (5, 2): "RI", (5, 3): "SL",
                (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
                (7, 0): "RI", (7, 1): "RI",
                (8, 0): "PK"
            },
            "4c": {
                (0, 0): "FL", (0, 1): "FL", (0, 2): "FL", (0, 3): "SL", (0, 4): "VL", (0, 5): "VL", (0, 6): "VL", (0, 7): "VL", (0, 8): "VL",
                (1, 0): "FL", (1, 1): "FL", (1, 2): "SL", (1, 3): "SL", (1, 4): "VL", (1, 5): "VL", (1, 6): "VL", (1, 7): "VL",
                (2, 0): "FL", (2, 1): "SL", (2, 2): "SL", (2, 3): "SL", (2, 4): "SL", (2, 5): "VL", (2, 6): "VL",
                (3, 0): "SL", (3, 1): "SL", (3, 2): "SL", (3, 3): "SL", (3, 4): "SL", (3, 5): "SL",
                (4, 0): "RI", (4, 1): "RI", (4, 2): "SL", (4, 3): "SL", (4, 4): "SL",
                (5, 0): "RI", (5, 1): "RI", (5, 2): "RI", (5, 3): "SL",
                (6, 0): "RI", (6, 1): "RI", (6, 2): "RI",
                (7, 0): "RI", (7, 1): "RI",
                (8, 0): "RI"
            }
        }
        landform_abrv_to_int_dict = {
            "FL": 1, "PK": 2, "RI": 3, "SH": 4, "CV": 5, "SL": 6, "CN": 7, "FS": 8, "VL": 9, "PT": 10
        }
        if number < 0:
            raise ValueError("The input number must be a non-negative integer.")
        
        ternary_patterns = []
        while number > 0:
            ternary_patterns.append(number % 3)
            number //= 3
        
        # Reverse the list to get the ternary pattern in the correct order
        ternary_patterns.reverse()
        
        # Ensure the result has exactly 8 numbers by padding with zeros at the beginning
        while len(ternary_patterns) < 8:
            ternary_patterns.insert(0, 0)
        # count the minuses and plusses based on the local ternary patterns
        minuses, pluses = ternary_patterns.count(0), ternary_patterns.count(2)
        lookup_tuple = (minuses, pluses)

        land_abrv = combined_lookup_table[classes].get(lookup_tuple, "Unknown")
        land_int = landform_abrv_to_int_dict.get(land_abrv, "Unknown")
        return land_int

    def classify_bathymorphons(self, classes="10c"):
        """
        Process a raster file, classify its values using a given function, and save the modified raster.

        Parameters:
        input_raster_path (str): Path to the input raster file.
        value_to_label (dict): Dictionary mapping classified values to labels.
        classes (str): Classification scheme to use (default is "10c").

        Returns:
        str: Path to the saved output raster file.
        """
        value_to_label = {1:"Flat", 2:"Peak", 3:"Ridge", 4:"Shoulder", 5:"Spur (Convex slope)", 6:"Slope", 7:"Hollow (Concave slope)", 8:"Footslope", 9:"Valley", 10:"Pit"}
        bathymorphon_raster_path = os.path.join(self.local_workspace, self.bathymorphons_raster)
        with rasterio.open(bathymorphon_raster_path) as src:
            # Read the raster data as a numpy array
            raster_data = src.read(1)  # Read the first band
            # Get raster metadata
            metadata = src.meta

        raster_name = os.path.basename(bathymorphon_raster_path).split(".")[0]
        # Mask the nodata value
        masked_data = raster_data[raster_data != metadata['nodata']]

        # Apply the classification function to the masked data
        vectorized_classify = np.vectorize(lambda x: self.classify_landform_from_bathymorphon(x, classes=classes))
        masked_data = vectorized_classify(masked_data)

        # Update the raster data with the classified values
        raster_data[raster_data != metadata['nodata']] = masked_data

        # Define the output file path
        output_file = os.path.join(self.geomorphons_folder, f"{raster_name}_{classes}.tif")

        # Write the modified raster data to the output file
        with rasterio.open(output_file, 'w', **metadata) as dst:
            # Write the modified raster data to the output file
            dst.write(raster_data, 1)
            # Update metadata with value_to_label as a JSON string
            dst.update_tags(**{"value_to_label": json.dumps(value_to_label)})
        converted_tiff = HabitatDerivatives().convert_tiff_to_gdal_raster(output_file, compress=False)
        return converted_tiff
    
class inpainter:
    def __init__(self, input_dem_path, save_path="habitat_derivatives"):
        """Initialize the inpainter with the input DEM path and optional save path."""
        self.input_dem_path = input_dem_path
        self.temp_workspace = tempfile.mkdtemp()  # Use a temporary folder for intermediate outputs
        self.tif_name = Utils().sanitize_path_to_name(input_dem_path)
        self.save_path = os.path.join(save_path, self.tif_name)
        # create the save path if it doesn't exist
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)

    def fill_internal_gaps_arcpy(self, max_iterations=1, smoothing_radius=1):
        """Fill internal gaps in the DEM using smoothing and filling."""
        arcpy.env.overwriteOutput = True
        arcpy.env.workspace = self.temp_workspace
        arcpy.env.scratchWorkspace = self.temp_workspace

        input_tif = self.input_dem_path
        converted_input = os.path.join(self.temp_workspace, f"{self.tif_name}_converted.tif")

        try:
            # Step 1: Verify the input raster
            print("Verifying input raster...")
            if not arcpy.Exists(input_tif):
                raise FileNotFoundError(f"Input raster not found: {input_tif}")
            desc = arcpy.Describe(input_tif)
            print(f"Input raster properties: Format={desc.format}, SpatialReference={desc.spatialReference.name}")

            # Step 2: Convert the input raster to a supported format
            print("Converting input raster to supported format...")
            arcpy.CopyRaster_management(input_tif, converted_input, format="TIFF")
            print(f"Converted raster saved to: {converted_input}")

            # Step 4: Iteratively smooth and fill NoData values
            print("Filling NoData values iteratively with smoothing...")
            filled_raster = Raster(converted_input)
            for iteration in range(max_iterations):
                print(f"Pass {iteration + 1} of smoothing and filling...")
                # Apply smoothing using FocalStatistics
                smoothed_raster = FocalStatistics(filled_raster, NbrCircle(smoothing_radius, "CELL"), "MEAN", "DATA")
                # Fill gaps after smoothing
                filled_raster = Fill(smoothed_raster)
                smoothed_raster = None  # Release smoothed_raster object
            filled_raster_path = os.path.join(self.save_path, f"{self.tif_name}_filled.tif")
            filled_raster.save(filled_raster_path)
            print(f"Filled raster saved to: {filled_raster_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Clean up temporary files
            print("Cleaning up temporary files...")
            arcpy.Delete_management("in_memory")  # Clear in-memory workspace
            filled_raster = None
            arcpy.env.workspace = None
            arcpy.env.scratchWorkspace = None

            for file in os.listdir(self.temp_workspace):
                file_path = os.path.join(self.temp_workspace, file)
                for _ in range(5):  # Retry up to 5 times
                    try:
                        os.remove(file_path)
                        break
                    except PermissionError:
                        print(f"Retrying deletion of {file_path}...")
                        time.sleep(1)
                else:
                    print(f"Could not delete {file_path}")

            try:
                os.rmdir(self.temp_workspace)
                print("Temporary files cleaned up.")
            except Exception as cleanup_error:
                print(f"Could not remove temporary directory: {cleanup_error}")
        
        return filled_raster_path

    def create_binary_mask(self, input_raster, boundary_shapefile):
        """Create a binary mask using the input raster extents and a boundary shapefile."""
        print("Creating binary mask...")
        boundary_raster = os.path.join(self.save_path, f"{self.tif_name}_boundary.tif")
        binary_mask = os.path.join(self.save_path, f"{self.tif_name}_binary_mask.tif")

        # Set the environment to match the input raster
        print("Setting environment to match input raster...")
        arcpy.env.extent = input_raster
        arcpy.env.snapRaster = input_raster
        arcpy.env.cellSize = input_raster

        # Rasterize the boundary shapefile
        print("Rasterizing the boundary shapefile...")
        arcpy.PolygonToRaster_conversion(boundary_shapefile, "FID", boundary_raster, "CELL_CENTER", "", input_raster)
        print(f"Boundary raster created: {boundary_raster}")

        # Generate the binary mask (1 for boundary, 0 for everything else)
        print("Generating binary mask...")
        binary_mask_raster = Con(IsNull(Raster(boundary_raster)), 0, 1)
        binary_mask_raster.save(binary_mask)
        print(f"Binary mask saved to: {binary_mask}")
        return binary_mask

    def trim_raster(self, raster, binary_mask, remove_original=True):
        """Trim the raster using the binary mask and replace border zeros with NoData."""
        print("Trimming the raster using the binary mask...")
        name = os.path.splitext(os.path.basename(raster))[0]
        trimmed_raster_path = os.path.join(self.save_path, f"{name}_trimmed.tif")
        mask = Raster(binary_mask)
        
        # Convert binary mask to 1, NoData
        print("Converting binary mask to 1, NoData...")
        mask = Con(mask == 1, 1, None)

        # Apply the mask to the raster
        trimmed_raster = Raster(raster) * mask
        
        # Save the trimmed raster
        trimmed_raster.save(trimmed_raster_path)
        print(f"Trimmed raster saved to: {trimmed_raster_path}")
        if remove_original:
            # Remove the original raster if specified
            print("Removing the original raster...")
            arcpy.Delete_management(raster)
        return trimmed_raster_path

    def get_filled_data_boundary(self, min_area=1000, shrink_pixels=3, max_iterations=2, smoothing_radius=3):
        """Generate a shapefile of the data boundary, create a binary mask, and trim the filled raster."""
        filled_raster = self.fill_internal_gaps_arcpy(max_iterations=max_iterations, smoothing_radius=smoothing_radius)
        temp_integer_raster = None
        temp_polygon = None
        dissolved_polygon = None
        cleaned_polygon = None
        trimmed_raster = None

        try:
            print("Generating data boundary shapefile...")

            # Ensure the shapefile directory exists
            shapefile_dir = os.path.join(self.save_path, "shapefile")
            if not os.path.exists(shapefile_dir):
                os.makedirs(shapefile_dir)

            # Step 1: Convert the raster to integer type
            temp_integer_raster = os.path.join(self.save_path, f"{self.tif_name}_integer.tif")
            print("Converting raster to integer type...")
            integer_raster = Int(Raster(filled_raster))
            integer_raster.save(temp_integer_raster)
            print(f"Integer raster saved to: {temp_integer_raster}")

            # Step 2: Convert the integer raster to polygons
            temp_polygon = os.path.join(self.save_path, "temp_polygon.shp")
            print("Converting raster to polygons...")
            arcpy.RasterToPolygon_conversion(temp_integer_raster, temp_polygon, "NO_SIMPLIFY", "VALUE")
            print(f"Temporary polygon shapefile created: {temp_polygon}")

            # Step 3: Dissolve polygons to create a single boundary
            dissolved_polygon = os.path.join(self.save_path, "dissolved_polygon.shp")
            print("Dissolving polygons to create a single boundary...")
            arcpy.Dissolve_management(temp_polygon, dissolved_polygon)
            print(f"Dissolved polygon shapefile created: {dissolved_polygon}")

            # Step 4: Eliminate small polygons
            cleaned_polygon = os.path.join(self.save_path, "cleaned_polygon.shp")
            print(f"Removing small polygons smaller than {min_area} square units...")
            arcpy.EliminatePolygonPart_management(dissolved_polygon, cleaned_polygon, "AREA", min_area)
            print(f"Cleaned polygon shapefile created: {cleaned_polygon}")

            # Step 5: Shrink the boundary by n pixels
            output_shapefile = os.path.join(shapefile_dir, f"{self.tif_name}_boundary.shp")
            binary_mask = os.path.join(self.save_path, f"{self.tif_name}_binary_mask.tif")
            if shrink_pixels > 0:
                print(f"Shrinking the boundary by {shrink_pixels} pixels...")
                shrink_distance = -shrink_pixels  # Negative distance for shrinking
                arcpy.Buffer_analysis(cleaned_polygon, output_shapefile, shrink_distance, "FULL", "ROUND", "ALL")
                print(f"Shrunken boundary shapefile saved to: {output_shapefile}")
            else:
                # If no shrinking is needed, save the cleaned polygon as the final output
                arcpy.CopyFeatures_management(cleaned_polygon, output_shapefile)
                print(f"Boundary shapefile saved to: {output_shapefile}")

            # Create the binary mask
            binary_mask = self.create_binary_mask(filled_raster, output_shapefile)

            # Trim the filled raster using the binary mask
            self.trim_raster(filled_raster, binary_mask, remove_original=False)

        except Exception as e:
            print(f"An error occurred while generating the data boundary: {e}")

        finally:
            # Clean up temporary files
            if temp_integer_raster and arcpy.Exists(temp_integer_raster):
                arcpy.Delete_management(temp_integer_raster)
                print("Temporary integer raster deleted.")
            if temp_polygon and arcpy.Exists(temp_polygon):
                arcpy.Delete_management(temp_polygon)
                print("Temporary polygon shapefile deleted.")
            if dissolved_polygon and arcpy.Exists(dissolved_polygon):
                arcpy.Delete_management(dissolved_polygon)
                print("Temporary dissolved polygon shapefile deleted.")
            if cleaned_polygon and arcpy.Exists(cleaned_polygon):
                arcpy.Delete_management(cleaned_polygon)
                print("Temporary cleaned polygon shapefile deleted.")

        return binary_mask