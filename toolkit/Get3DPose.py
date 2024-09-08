import cv2 as cv
import numpy as np
import pandas as pd
import json, csv

import pymap3d
import rasterio
from pyproj import Geod
from rasterio.warp import transform

class Tracker3D():
    def __init__(self, video_json, dsm_path, ort_path, ort_json) -> None:
        """
        Used to convert vehicle UV coordinates to physical coordinates.

        Args:
            video_json (path): The reference points in the video are derived from manual annotations.
            dsm_path (path): The DSM data path is located in the Physics directory.
            ort_path (str): The ORT data path is located in the Physics directory.
            ort_json (str): Reference point UV coordinate annotation.
        """
        self.cam = "dji_mavic_3t_sRGB"
        self.dsm_dir = '/usr/src/TrafficNight/Physics/dsm'
        self.video_dir = '/usr/src/TrafficNight/RawVideo'
        self.video_pose = None # save camera Matrix and vertor
        self.ortho_json = ort_json
        self.video_json = video_json
        self.ortho_path = ort_path
        self.dsm_path = dsm_path

        # pre-open dsm and ortho
        self.dsm = rasterio.open(self.dsm_path)
        self.ortho = rasterio.open(self.ortho_path)
        self.origin_lonlat = [None, None, None]
        self.cam_rotavet, self.cam_transvet = None, None

        # Trafficnight has two cameras: thermal imaging and RGB
        self.cam_matrix_ir = np.array(
            [[1.58239335e+03, 0.00000000e+00, 6.41746138e+02],
             [0.00000000e+00, 1.58180229e+03, 4.92388969e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
             )
        self.cam_distor_ir = np.array([[ 4.44085745e+01, -1.08401128e+02, -2.49682224e-03,-8.27697259e-03, 2.70290225e+00, 4.47484524e+01,
                                        -9.31348471e+01, -3.39309147e+01, 3.37146760e-02, -6.24796317e-04, 1.31348445e-02, 9.53964873e-03]])
        
        # rgbCam:
        self.cam_matrix_rgb = np.array([
            [2.75621420e+03, 0.00000000e+00, 1.91258769e+03],
            [0.00000000e+00, 2.75559857e+03, 1.05993147e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
            ])
        self.cam_distor_rgb = np.array([[2.82678353e-01, -9.93554910e-01, -1.59175511e-03,  7.67130325e-04, 
                                         1.03673362e+00]])


    def readJson(self,jsonPath):
        """
        Load spaces reference point from Json file

        Args:
            jsonPath (string): json file path
        Returns:
            dict: ref point information {point_name: [u,v]}
        """
        lutDic = {}
        try:
            with open(jsonPath, 'r') as file:
                data = json.load(file)
                for item in data['shapes']:
                    name = item.get('label')
                    coordinates = item.get('points')
                    lutDic[name] = coordinates[0]
            print("load success")
        except Exception as e:
            print(f"read JSON error: {e}")
        return lutDic
    
    def lutDsm(self, points):
        """
        Get the ground reference point physical coordinates of the 
        corresponding rows and columns on the DSM, 
        including longitude, latitude and ENU coordinates.

        Args:
            points (dic): The UV coordinates of the points to be queried come from the annotation tool.
        Returns:
            lonlatdic: The latitude and longitude coordinates of the 
            ground reference point in the WGS84 reference system.
            enudic: The coordinate dictionary of reference points in the ENU coordinate system, 
            with 0 as the origin of the ENU reference system.
        """
        lonlatdic = {}
        enudic = {}
        # build origin
        origin_uv = points['0']
        origin_lon, origin_lat = self.ortho.xy(origin_uv[1], origin_uv[0])
        row, col = self.dsm.index(origin_lon, origin_lat) # row and col in dsm
        origin_eva = self.dsm.read(1)[row, col] # get eva
        self.origin_lonlat = [origin_lon, origin_lat, origin_eva]
        
        for potName in points:
            [col,row] = points[potName]
            lon, lat = self.ortho.xy(row, col) # current point lon lat
            row_, col_ = self.dsm.index(lon, lat) # dsm row col
            elevation = self.dsm.read(1)[row_, col_]
            lonlatdic[potName]=[lon, lat, elevation]
            # convert current lon lat to enu
            e, n, u = pymap3d.geodetic2enu(lat, lon, elevation, origin_lat, origin_lon, origin_eva)
            enudic[potName] = [e * 1000, n* 1000, u* 1000]  # convert m to mm

        return lonlatdic, enudic

    def lonlat2enu(self, lonlat):
        """
        Convert the incoming latitude and longitude coordinates to ENU coordinates.
        The origin of the Enu coordinate system has been defined.

        Args:
            lonlat (dic): Input latitude and longitude dictionary.

        Returns:
            enudic: Returns the enu coordinates.
        """
        enu_dic = {}
        for potName in lonlat:
            [lon, lat, eva] = lonlat[potName]
            geod = Geod(ellps="WGS84")
            az, _, dist = geod.inv(self.origin_lonlat[0], self.origin_lonlat[1], lon, lat)
            angle_rad = np.deg2rad(az)
            east = dist * np.sin(angle_rad) *1000
            north = dist * np.cos(angle_rad) *1000
            up = (eva - self.origin_lonlat[2]) *1000
            enu_dic[potName] = [east, north, up]
        
        return enu_dic
    
    def enu2lonlat(self, enu_dict):
        """
        Convert local ENU coordinates back to longitude and latitude.

        Args:
            enu_dict (dic): A dictionary containing ENU coordinates for each point.

        Returns:
            dic: A dictionary with the corresponding longitude and latitude for each point.
        """
        lonlat_dict = {}
        geod = Geod(ellps="WGS84")
        for potName, enu in enu_dict.items():
            east, north, up = enu # convert m to mm
            east, north, up = east/1000, north/1000, up/1000
            az = np.rad2deg(np.arctan2(east, north))
            dist = np.sqrt(east**2 + north**2)
            lon, lat, _ = geod.fwd(self.origin_lonlat[0], self.origin_lonlat[1], az, dist)
            eva = up + self.origin_lonlat[2]
            lonlat_dict[potName] = [lon, lat, eva]
        return lonlat_dict
    
    
    def initCamPose(self):
        """
        Read the reference point information 
        and calculate the camera's extrinsic parameters.
        """
        # load match points information
        video_refuv = self.readJson(self.video_json) # video ref point (UV)
        ortho_refuv = self.readJson(self.ortho_json) # ort ref point (UV)
        ref3D84, ref3Denu = self.lutDsm(ortho_refuv) # ref point in 3D space and origin point
        ref3Denu_ = ref3Denu.copy()
        
        # caculate camera pose from pair
        new_ref3Denu = {}
        for name in ref3Denu:
            if name in video_refuv:
                new_ref3Denu[name] = ref3Denu[name].copy()
        ref3Denu = new_ref3Denu.copy()
        refpoints_enu = np.asarray([ref3Denu[name] for name in ref3Denu]).reshape(-1,1)
        refpoints_uv =np.asarray([video_refuv[name] for name in ref3Denu]).reshape(-1,1)

        refpoints_enu = np.array(refpoints_enu, dtype=np.float32).reshape(-1, 3) 
        refpoints_uv = np.array(refpoints_uv, dtype=np.float32).reshape(-1, 2) 

        # Use solvePnP to solve the camera extrinsics
        success, self.cam_rotavet, self.cam_transvet, inliers = cv.solvePnPRansac(refpoints_enu, 
                                                                                  refpoints_uv, 
                                                                                  self.cam_matrix_rgb, 
                                                                                  self.cam_distor_rgb)

        rotation_matrix, _ = cv.Rodrigues(self.cam_rotavet)
        cam_position = -rotation_matrix.T.dot(self.cam_transvet)
        self.cam_position = cam_position.flatten()
    
    def undistort_points(self, uv):
        """
        Remove the camera distortion effect of UV points on the image

        Args:
            uv (list): UV coordinate values on the image

        Returns:
            array : UV coordinates after removing distortion.
        """
        uv = np.array(uv, dtype=np.float32).reshape(-1, 1, 2)
    
        undistorted_uv = cv.undistortPoints(uv, self.cam_matrix_rgb, self.cam_distor_rgb, P=self.cam_matrix_rgb)

        return undistorted_uv.reshape(-1, 2)[0]
    
    
    def undistort_points_ir(self, uv):
        """
        Removing distortion effects from near-infrared camera images.

        Args:
            uv (list): UV coordinate values on the image

        Returns:
            array : UV coordinates after removing distortion.
        """

        self.cam_right_top_dist_coeffs = np.array([[4.40717806e+01, -1.06841333e+02,  3.63281275e-02, -2.24049689e-02,
                                                   2.82247064e+00,  4.53615275e+01, -9.50634932e+01, -3.40504832e+01,
                                                   9.42429256e-02, -1.20193157e-01,  2.71706211e-02,  6.83205393e-02]])
        
        self.cam_left_top_dist_coeffs = np.array([[ 4.31283664e+01, -1.06756138e+02, -4.89168106e-02, -2.17800982e-02,
                                                  2.67471469e+00,  4.59072600e+01, -9.41332793e+01, -3.38520760e+01,
                                                  1.34356373e-01, -7.81187186e-03, -2.60306265e-02, -6.05855175e-03]])

        uv = np.array(uv, dtype=np.float32).reshape(-1, 1, 2)

        u, v = uv[0, 0, 0], uv[0, 0, 1]
        if u > 640 and v < 512:
            # dist_coeffs = self.cam_right_top_dist_coeffs
            dist_coeffs = self.cam_distor_ir
        elif u <= 640 and v < 512:
            # dist_coeffs = self.cam_left_top_dist_coeffs
            dist_coeffs = self.cam_distor_ir
        else:
            dist_coeffs = self.cam_distor_ir

        undistorted_uv_ir = cv.undistortPoints(uv, self.cam_matrix_ir, dist_coeffs, P=self.cam_matrix_ir)

        return undistorted_uv_ir.reshape(-1, 2)[0]
    

    def construct_ray(self, uv, cam_type):
        """
        Construct a space vector from the camera position through the image UV coordinates 
        based on the camera's external parameter rotation vector and translation vector, 
        the internal parameter matrix and the distortion coefficient.

        Args:
            uv (lsit): Image UV coordinates

        Returns:
            ray_origin_enu: The position of the camera in the ENU coordinate system. 
                            Use camera extrinsics directly.
            ray_direction_enu: The direction vector of the ray passing through the UV pixels at the camera position.
        """

        ray_origin_enu = self.cam_position

        ## Convert the ray in the camera coordinate system to the ray in the world coordinate system
        rotation_matrix, _ = cv.Rodrigues(self.cam_rotavet)
        if cam_type == 'rgb':
            # Conventional RGB camera processing method
            uv_true = self.undistort_points(uv)
            ## Step 1: Convert image coordinates to normalized coordinates
            uv_homogeneous = np.array([uv_true[0], uv_true[1], 1.0])
            ## Step 2: Back-projection operation to obtain the ray in the camera coordinate system
            normalized_coordinates = np.linalg.inv(self.cam_matrix_rgb) @ uv_homogeneous 
            ray_camera_coordinates = np.array([normalized_coordinates[0], normalized_coordinates[1], 1], dtype=np.float32)
            ray_direction_enu = np.linalg.inv(rotation_matrix) @ ray_camera_coordinates # rgb camera
        
        else:
            # Infrared camera processing solution.
            uv_true_ir = self.undistort_points_ir(uv)
            uv_homogeneous_ir = np.array([uv_true_ir[0], uv_true_ir[1], 1.0]) 
            normalized_coordinates_ir = np.linalg.inv(self.cam_matrix_ir) @ uv_homogeneous_ir
            ray_camera_coordinates_ir = np.array([normalized_coordinates_ir[0], 
                                            normalized_coordinates_ir[1], 
                                            1], dtype=np.float32)
            ray_direction_enu = np.linalg.inv(rotation_matrix) @ ray_camera_coordinates_ir
        
        return ray_origin_enu, ray_direction_enu

    def rayTrack_dsm(self, uv, epsilon, camtype):
        """
        Solve the physical coordinates corresponding to the UV pixel coordinates. 
        Construct the ray of the UV pixel and use the binary method to find the ray landing point 
        that meets the requirements (minimum spacing tolerance).

        Args:
            uv (list): _description_
            epsilon (float): Spacing selection
            camtype (str): camera type ir or rgb

        Returns:
            _type_: _description_
        """

        # The camera's current UV ray
        ray_origin, ray_direction= self.construct_ray(uv, camtype)
        # Search starting conditions
        t_min, t_max = ray_origin[2], 3*ray_origin[2]

        dsm_data = self.dsm.read(1)
        dsm_height, dsm_width = dsm_data.shape
        it, max_it = 1, 50
        while t_max - t_min > epsilon:
            t_mid = (t_min + t_max) / 2.0
            point_mid = ray_origin + t_mid * ray_direction  # unit is mm
            en_point = point_mid[:2]  # unit is mm
            midlat, midlon, _ = pymap3d.enu2geodetic(
                en_point[0] / 1000, en_point[1] / 1000, self.origin_lonlat[2],
                self.origin_lonlat[1], self.origin_lonlat[0], self.origin_lonlat[2]
            )

            # Try to optimize the `index` method call to find the index of all intermediate points at once
            mid_row, mid_col = self.dsm.index(midlon, midlat)
            if 0 <= mid_col < dsm_width and 0 <= mid_row < dsm_height:
                dsm_eva = dsm_data[mid_row, mid_col]
                if dsm_eva is None:
                    t_max = t_mid
                    continue
                enu_eva = dsm_eva - self.origin_lonlat[2]
                if point_mid[2] < enu_eva * 1000:
                    t_max = t_mid # The intersection point is in the first half of the interval
                else:
                    t_min = t_mid # The intersection point is in the second half of the interval
            else:
                print("out Dsm Range") # If the ray is out of range of the DSM data
                t_max = t_mid
                continue
            it += 1
            if it > max_it:
                break

        potint_enu = ray_origin + (t_min + t_max) / 2.0 * ray_direction  # unit is mm        
        potint_lat, potint_lon, potint_eva = pymap3d.enu2geodetic(potint_enu[0]/1000, 
                                                         potint_enu[1]/1000, 
                                                         potint_enu[2]/1000 + self.origin_lonlat[2],
                                                         self.origin_lonlat[1], 
                                                         self.origin_lonlat[0], 
                                                         self.origin_lonlat[2])
        potint_lonlat = [potint_lon, potint_lat, potint_eva]
        pixel_v_, pixel_u_ = self.ortho.index(*[potint_lon, potint_lat]) 
        potint_uvortho = [pixel_u_, pixel_v_]

        return potint_enu, potint_lonlat, potint_uvortho
    
    
    def process_uv_point(self, row):
        """
        In multi-process mode, each UV point is processed individually, 
        the intersection with the DSM is calculated and an updated row with ENU and WGS84 coordinates is returned.

        Args:
            row (_type_): Read row data from csv

        Returns:
            _type_: _description_
        """
        u_center, v_center = round(float(row[3])), round(float(row[4]))

        # Call rayTrack_dsm to get ENU coordinates
        point_enu, point_lonlat, point_uvortho = self.rayTrack_dsm([u_center, v_center], 200, 'ir')  

        # Add ENU and WGS84 coordinates to the row data in mm
        new_row = row + [point_enu[0], point_enu[1], point_enu[2], point_lonlat[0], point_lonlat[1], point_lonlat[2]]
        return new_row


    def rayTrackAgent_mThread(self, csv_path):
        '''
        Use multiple processes to obtain the longitude and latitude coordinates of objects in the Track result.
        '''
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        data_rows = []
        with open(csv_path, mode='r') as file:
            reader = csv.reader(file)
            header = next(reader) 
            new_header = header + ['cent_x', 'cent_y', 'cent_z', 'cent_lon', 'cent_lat', 'cent_alt']
            
            with ThreadPoolExecutor(max_workers=8) as executor: 
                future_to_row = {executor.submit(self.process_uv_point, row): row for row in reader}
                
                for future in as_completed(future_to_row):
                    new_row = future.result()
                    print(new_row)
                    data_rows.append(new_row)

        data_rows.sort(key=lambda x: float(x[2]))
        new_csv_path = csv_path.replace('.csv', '_with_enu.csv') 
        with open(new_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_header) 
            writer.writerows(data_rows) 


    def DrawCameraView(self):
        '''
        Draw the camera field of view to ort
        '''
        camera_uvs = [
            [0,0],
            [3840,0],
            [3840, 2160],
            [0, 2160]
        ]
        ortho_image = self.ortho.read([3, 2, 1])  # Read RGB bands
        ortho_image = np.moveaxis(ortho_image, 0, -1)
        
        scale_factor = 0.3
        OrtView = cv.resize(ortho_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)

        # Draw Origin Point to Ort
        pixel_v, pixel_u = self.ortho.index(*[self.origin_lonlat[0], 
                                              self.origin_lonlat[1]])
        pixel_u_scaled, pixel_v_scaled = int(pixel_u * scale_factor), int(pixel_v * scale_factor)
        cv.circle(OrtView, (pixel_u_scaled, pixel_v_scaled), 5, (0, 255, 0), -1)
        cv.putText(OrtView, 'OriginPoint(0,0)', 
                   (pixel_u_scaled + 10, pixel_v_scaled - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 1, cv.LINE_AA)
        
        # Camera Estimates Pos
        cam_lat, cam_lon, cam_eva = pymap3d.enu2geodetic(self.cam_position[0]/1000, 
                                                         self.cam_position[1]/1000, 
                                                         self.cam_position[2]/1000 + self.origin_lonlat[2],
                                                         self.origin_lonlat[1], 
                                                         self.origin_lonlat[0], 
                                                         self.origin_lonlat[2])
        # draw cam estimates to ortview
        pixel_v, pixel_u = self.ortho.index(*[cam_lon, cam_lat])
        pixel_u_scaled, pixel_v_scaled = int(pixel_u * scale_factor), int(pixel_v * scale_factor)
        cv.circle(OrtView, (pixel_u_scaled, pixel_v_scaled), 5, (255, 255, 0), -1)
        cv.putText(OrtView, 'Camera estimates', 
                   (pixel_u_scaled + 10, pixel_v_scaled - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 0), 1, cv.LINE_AA)
        
        fov_points = []
        fov_points_nosc = []
        for uvpoint in camera_uvs:
            point_enu, point_lonlat, point_uvortho = self.rayTrack_dsm(uvpoint, 50, 'rgb')
            fov_points_nosc.append(point_uvortho)
            pixel_u_scaled, pixel_v_scaled = int(point_uvortho[0] * scale_factor), int(point_uvortho[1] * scale_factor)
            fov_points.append([pixel_u_scaled, pixel_v_scaled])
        
        fov_points = np.array(fov_points, dtype=np.int32)
        cv.polylines(OrtView, [fov_points], isClosed=True, color=(0, 255, 0), thickness=2)
        mask = np.zeros_like(OrtView)
        cv.fillPoly(mask, [fov_points], color=(255, 255, 255))
        black_background = np.zeros_like(OrtView)
        transparent_mask = cv.addWeighted(OrtView, 1 - 0.5, black_background, 0.5, 0)
        mask_inv = cv.bitwise_not(mask)
        OrtView_masked = cv.bitwise_and(OrtView, mask)
        OrtView_final = cv.add(OrtView_masked, cv.bitwise_and(transparent_mask, mask_inv))

        DrawCamView = self.video_json.replace('.json','_CamFov.jpg')
        cv.imwrite(DrawCamView, OrtView_final)
        return fov_points_nosc

    def drawAgents(self, csv_path, out_path):
        '''
        Generate agent video based on CSV file (with lon/lat).
        '''
        df = pd.read_csv(csv_path)
        ortho_image = self.ortho.read([3, 2, 1])  # Read RGB bands
        ortho_image = np.moveaxis(ortho_image, 0, -1)  # Convert from (channels, height, width) to (height, width, channels)
        
        scale_factor = 0.3
        small_ortho = cv.resize(ortho_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
        height, width = small_ortho.shape[:2]

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(out_path, fourcc, 24, (width, height))

        df.sort_values('scene_ts', inplace=True)
        unique_ts = df['scene_ts'].unique()

        for ts in unique_ts:
            frame = small_ortho.copy()
            ts_df = df[df['scene_ts'] == ts]
            for _, row in ts_df.iterrows():
                lon, lat = row['lon'], row['lat']
                pixel_v, pixel_u = self.ortho.index(*[lon, lat])
                pixel_u_scaled, pixel_v_scaled = int(pixel_u * scale_factor), int(pixel_v * scale_factor)
                if 0 <= pixel_u_scaled < width and 0 <= pixel_v_scaled < height:
                    cv.circle(frame, (pixel_u_scaled, pixel_v_scaled), 25, (0, 255, 0), -1)
                    print(f'Agent drawn at ({pixel_u_scaled}, {pixel_v_scaled}) for timestamp {ts}')
                else:
                    print(f"Warning: Agent out of bounds at time {ts} with coordinates ({pixel_u}, {pixel_v})")

            video_writer.write(frame)

        video_writer.release()
    

    def drawAgents_withfov(self, csv_path, out_path, fov_points_nosc):
        """
        Draw 3D coordinate tracking results taking FOV into account, which will reduce the display of unnecessary areas.

        Args:
            csv_path (path): Get the csv file path of the latitude and longitude information.
            out_path (path): Video output path
            fov_points_nosc (arr): The four corner boundary points of the image
        """
        class_colors = {
            0: (0, 255, 0),    
            1: (255, 0, 0),    
            2: (0, 0, 255),    #
            3: (255, 255, 0),  #
            4: (255, 0, 255),  # 
            5: (0, 255, 255)   # 
        }

        # Read tracking result data
        df = pd.read_csv(csv_path)

        # Reading an orthographic projection
        ortho_image = self.ortho.read([3, 2, 1]) 
        ortho_image = np.moveaxis(ortho_image, 0, -1)
        # Cropping ORT image fov_points_nosc [[u1,v1], [u2,v2],..]
        view_arr = np.asarray(fov_points_nosc)
        top_left_pixel_v, top_left_pixel_u = view_arr[:,1].min(), view_arr[:, 0].min()
        bottom_right_pixel_v, bottom_right_pixel_u = view_arr[:,1].max(), view_arr[:, 1].max()
        cropped_ortho = ortho_image[top_left_pixel_v:bottom_right_pixel_v, top_left_pixel_u:bottom_right_pixel_u]
        # Zoom
        scale_factor = 0.3
        small_ortho = cv.resize(cropped_ortho, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
        height, width = small_ortho.shape[:2]

        # Creating a Video Writer
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(out_path, fourcc, 24, (width, height))

        df.sort_values('scene_ts', inplace=True)
        unique_ts = df['scene_ts'].unique()
        for ts in unique_ts:
            frame = small_ortho.copy()
            ts_df = df[df['scene_ts'] == ts]
            for _, row in ts_df.iterrows():

                lon, lat = row['cent_lon'], row['cent_lat']
                pixel_v, pixel_u = self.ortho.index(lon, lat)
                pixel_u_cropped = pixel_u - top_left_pixel_u
                pixel_v_cropped = pixel_v - top_left_pixel_v
                pixel_u_scaled, pixel_v_scaled = int(pixel_u_cropped * scale_factor), int(pixel_v_cropped * scale_factor)
                
                if 0 <= pixel_u_scaled < width and 0 <= pixel_v_scaled < height:
                    cv.circle(frame, (pixel_u_scaled, pixel_v_scaled), 15, class_colors[row['class']], -1)
                    print(f'Agent drawn at ({pixel_u_scaled}, {pixel_v_scaled}) for timestamp {ts}')
                    agent_text = f"{row['class']} {row['agent_id']}"
                    text_position = (pixel_u_scaled + 20, pixel_v_scaled - 20)
                    cv.putText(frame, agent_text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    print(f"Warning: Agent out of bounds at time {ts} with coordinates ({pixel_u}, {pixel_v})")

            video_writer.write(frame)
        video_writer.release()
 
if __name__ == "__main__":
    dt = Tracker3D(video_json = '/usr/src/TrafficNight/TN_RawVedio/TN03/TN10281958.json',
                   dsm_path = '/usr/src/TrafficNight/Physics/TN03_dsm_20cm.tif',
                   ort_path = '/usr/src/TrafficNight/Physics/TN03_ort.tif',
                   ort_json = '/usr/src/TrafficNight/Physics/TN03_ort.json')
    dt.initCamPose()

    fov_points_nosc = dt.DrawCameraView()
    
    dt.rayTrackAgent_mThread('/usr/src/TrafficNight/trackRes/TN03_DJI_20231028195825_0001_T_24hz.csv')
    
    dt.drawAgents_withfov('/usr/src/TrafficNight/trackRes/TN03_DJI_20231028195825_0001_T_24hz_with_enu.csv', 
                          '/usr/src/TrafficNight/TN03_DJI_20231028195825_0001_T_24hz_TrackwithOrt.MP4', fov_points_nosc)
    