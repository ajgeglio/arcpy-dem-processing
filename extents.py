import os, glob
import numpy as np
import pandas as pd
import pyproj
from arcpy.sa import *
from osgeo import gdal, osr
from utils import Utils

class GetExtents:
    def __init__(self) -> None:
        pass

    @staticmethod
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
    
    @staticmethod
    def get_projection(tif_file):
        ds = gdal.Open(tif_file)
        cs = osr.SpatialReference()
        return cs.ImportFromWkt(ds.GetProjectionRef())

    @staticmethod
    def get_min_max_xy(tif_file):
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
    
    @staticmethod
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
    
    @staticmethod
    def get_tif_coordinate_system(tif_file):
        # Function to get the coordinate system of the tif file
        ds = gdal.Open(tif_file)
        proj_wkt = ds.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)
        return srs

    @staticmethod
    def infer_xyz_coordinate_system(xyz_file, sample_size=1000):
        """
        Infers the coordinate system of an XYZ file by inspecting coordinate ranges.
        Assumes the file has three columns: X, Y, Z (no header).
        """
        try:
            # Read a sample of the file
            df = pd.read_csv(xyz_file, delim_whitespace=True, header=None, names=["X", "Y", "Z"], nrows=sample_size)
            x_vals = df["X"]
            y_vals = df["Y"]

            # Heuristic: if values look like lat/lon
            if x_vals.between(-180, 180).all() and y_vals.between(-90, 90).all():
                return pyproj.CRS("EPSG:4326")  # WGS84 Geographic

            # Heuristic: if values look like UTM (easting/northing in meters)
            if x_vals.min() > 100000 and x_vals.max() < 1000000 and y_vals.min() > 0:
                # Estimate UTM zone from mean longitude
                mean_lon = x_vals.mean()
                utm_zone = int((mean_lon + 180) / 6) + 1
                northern = y_vals.mean() > 0
                hemisphere = "+north" if northern else "+south"
                proj_str = f"+proj=utm +zone={utm_zone} {hemisphere} +datum=WGS84 +units=m +no_defs"
                return pyproj.CRS(proj_str)

            # Fallback
            return pyproj.CRS("EPSG:3857")  # Web Mercator as generic fallback

        except Exception as e:
            print(f"Could not infer CRS: {e}")
            return None

    @staticmethod
    def return_min_max_tif_df(tif_files=[]):
        reef_dict = {"BH": "Bay Harbor", "CH": "Cathead Point", "ER": "Elk Rapids", "FI": "Fishermans Island", "IP":"Ingalls Point",
                "LR": "Lees Reef", "MS": "Manistique Shoal", "MP": "Mission Point", "MR":"Mudlake Reef", "NPNE": "Northport Bay NE",
                "NPNW":"Northport Bay NW", "NPS":"Northport Bay S", "NP": "Northport Point", "SP": "Suttons Point", 
                "TC":"Tannery Creek", "TS":"Traverse Shoal", "TP":"Tuckers Point", "WP": "Wiggins Point", "GHR": "Good Harbor Reef", "TB":"Thunder Bay", "Ingalls": "IP", "Other": "Other"}
        min_max = [GetExtents.get_min_max_xy(t) for t in tif_files]
        names = [Utils().sanitize_path_to_name(t) for t in tif_files]
        # reefs = [os.path.basename(os.path.dirname(t)) for t in tif_files]
        reefs = [n.split("_")[0] for n in names]
        min_x, min_y = [min_max[i][0][0] for i in range(len(min_max))], [min_max[i][0][1] for i in range(len(min_max))]
        max_x, max_y  = [min_max[i][1][0] for i in range(len(min_max))], [min_max[i][1][1] for i in range(len(min_max))]
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
        try:
            min_max_df["Reef_Name"] = min_max_df.reef.apply(lambda x: reef_dict[x])
        except KeyError:
            # If the reef is not in the dictionary, assign "Other"
            min_max_df["Reef_Name"] = min_max_df.reef.apply(lambda x: reef_dict.get(x, "Other"))
        return min_max_df
    
    @staticmethod
    def log_col_lat_lon(root):
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
    
    @staticmethod
    def return_headers_min_max_coord(df):
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
    
    @staticmethod
    def return_MissionLog_min_max_coord(collect):
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
    
    @staticmethod
    def return_MissionLog_min_max_df(collect_list):
        items = []
        for collect in collect_list:
            min_lat, min_lon, max_lat, max_lon = GetExtents.return_MissionLog_min_max_coord(collect)
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