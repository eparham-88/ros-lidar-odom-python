#!/usr/bin/env python
import rospy
import numpy as np
import cv2


import numpy as np
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import tf.transformations as tr
from geometry_msgs.msg import Pose, Quaternion, Point
import cv_bridge
from cv_bridge import CvBridge
import message_filters


def blur(im):
    im_blur = cv2.GaussianBlur(im, (7,7), 0)
    return im_blur



class Lidar(object):
    def __init__(self):
        """
        beam_altitude_angles    -> len = 128, straight from page 86 of LiDAR manual (not sure if these are accurate, but a good starting point)
        beam_azimuth_angles     -> len = 128, straight from page 86 of LiDAR manual (not sure if these are accurate, but a good starting point)4
        beam_to_lidar_transform -> from page 87, converted to matrix
        lidar_to_beam_origin    -> just the top right element of the transform, but also provided on page 87
        scan_width              -> based on lidar config
        scan_height             -> based on lidar config
        n                       -> calculated form page 22 of manual
        theta_encoder           -> calculated from page 22 of manual, note that it is broadcast to calculate all encoder angles for the entire width so we can just index it later
        theta_azimuth           -> calcualted from page 22 of manual, note that it is broadcast to calculate all azimuth angles for the entire width so we can just index it later
        phi                     -> calcualted from page 22 of manual, note that it is broadcast to calculate all azimuth angles for the entire width so we can just index it later
        
        """
        self.beam_altitude_angles = np.array([44.47, 43.93, 43.22, 42.34, 41.53, 40.95, 40.23, 39.38, 38.58, 37.98, 37.27, 
                                              36.44, 35.65, 35.05, 34.33, 33.52, 32.74, 32.12, 31.41, 30.63, 29.85, 29.23, 
                                              28.52, 27.75, 26.97, 26.35, 25.65, 24.88, 24.12, 23.49, 22.79, 22.04, 21.29, 
                                              20.65, 19.96, 19.21, 18.48, 17.83, 17.12, 16.41, 15.66, 15.02, 14.32, 13.6, 
                                              12.88, 12.22, 11.53, 10.82, 10.1, 9.44, 8.74, 8.04, 7.33, 6.66, 5.97, 5.27, 
                                              4.56, 3.89, 3.2, 2.5, 1.8, 1.12, 0.43, -0.26, -0.96, -1.65, -2.33, -3.02, 
                                              -3.73, -4.42, -5.1, -5.79, -6.48, -7.2, -7.89, -8.56, -9.26, -9.98, -10.67, 
                                              -11.35, -12.04, -12.77, -13.45, -14.12, -14.83, -15.56, -16.26, -16.93, -17.63, 
                                              -18.37, -19.07, -19.73, -20.44, -21.19, -21.89, -22.55, -23.25, -24.02, -24.74, 
                                              -25.39, -26.09, -26.87, -27.59, -28.24, -28.95, -29.74, -30.46, -31.1, -31.81, 
                                              -32.62, -33.35, -33.99, -34.71, -35.54, -36.27, -36.91, -37.63, -38.47, -39.21, 
                                              -39.84, -40.57, -41.44, -42.2, -42.81, -43.55, -44.45, -45.21, -45.82])
        
        self.beam_azimuth_angles = np.array([11.01, 3.81, -3.25, -10.19, 10.57, 3.63, -3.17, -9.88, 10.18, 3.48, -3.1, -9.6, 
                                             9.84, 3.36, -3.04, -9.37, 9.56, 3.23, -2.99, -9.16, 9.3, 3.14, -2.95, -8.98, 9.08, 
                                             3.05, -2.91, -8.84, 8.9, 2.98, -2.88, -8.71, 8.74, 2.92, -2.85, -8.59, 8.6, 2.87, 
                                             -2.83, -8.51, 8.48, 2.82, -2.81, -8.43, 8.39, 2.78, -2.79, -8.37, 8.31, 2.75, -2.79, 
                                             -8.33, 8.25, 2.72, -2.78, -8.3, 8.22, 2.71, -2.79, -8.29, 8.19, 2.69, -2.79, -8.29, 
                                             8.18, 2.7, -2.8, -8.29, 8.2, 2.7, -2.81, -8.32, 8.22, 2.71, -2.82, -8.36, 8.27, 2.72, 
                                             -2.83, -8.41, 8.32, 2.74, -2.85, -8.48, 8.41, 2.78, -2.87, -8.55, 8.5, 2.81, -2.89, 
                                             -8.65, 8.63, 2.86, -2.92, -8.77, 8.76, 2.92, -2.97, -8.91, 8.93, 2.99, -3, -9.06, 
                                             9.12, 3.06, -3.06, -9.24, 9.34, 3.15, -3.12, -9.46, 9.6, 3.26, -3.18, -9.71, 9.91, 
                                             3.38, -3.25, -10, 10.26, 3.53, -3.34, -10.35, 10.68, 3.7, -3.43, -10.74])
        
        self.beam_to_lidar_transform = np.array([[1, 0, 0, 27.116], 
                                                 [0, 1, 0, 0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]])
        
        self.lidar_to_sensor_transform = np.array([[-1, 0, 0, 0],
                                                   [0, -1, 0, 0],
                                                   [0, 0, 1, 38.195],
                                                   [0, 0, 0, 1]])
        
        self.lidar_to_beam_origin_mm = 27.116
        
        self.scan_width = 1024
        self.scan_height = 128
        
        self.n = np.sqrt(np.square(self.beam_to_lidar_transform[0][3]) + np.square(self.beam_to_lidar_transform[2][3]))
        
        # self.theta_encoder = 2 * np.pi * (np.ones(self.scan_width) - np.arange(0,self.scan_width) / self.scan_width)
        
        # self.theta_azimuth = -2 * np.pi * self.beam_azimuth_angles / 360
        
        # self.phi = 2 * np.pi * self.beam_altitude_angles / 360
        
        # plt.plot(self.phi)
        # plt.show()
        
        # a = 0
        
    def getXYZCoords(self, u, v, r_16bit):
        """ 
        u is height (rows)
        v is width (cols)
        """
        # print(r)
        r = r_16bit * 4 # convert to mm
        
        theta_encoder = 2.0 * np.pi * (1.0 - v / self.scan_width)
        theta_azimuth = 0.0 * (-2.0 * np.pi * (self.beam_azimuth_angles[u] / 360.0))
        phi = 2.0 * np.pi * (self.beam_altitude_angles[u] / 360.0)
        
        x = (r - self.n) * np.cos(theta_encoder + theta_azimuth) * np.cos(phi) + self.beam_to_lidar_transform[0,3] * np.cos(theta_encoder)
        y = (r - self.n) * np.sin(theta_encoder + theta_azimuth) * np.cos(phi) + self.beam_to_lidar_transform[0,3] + np.sin(theta_encoder)
        z = (r - self.n) * np.sin(phi) + self.beam_to_lidar_transform[2,3]
        
        # x = (r - self.n)*np.cos(self.theta_encoder[measurement_id] + self.theta_azimuth[i])*np.cos(self.phi[i]) + (self.beam_to_lidar_transform[0][3])*np.cos(self.theta_encoder[measurement_id])
        # y = (r - self.n)*np.sin(self.theta_encoder[measurement_id] + self.theta_azimuth[i])*np.cos(self.phi[i]) + (self.beam_to_lidar_transform[0][3])*np.sin(self.theta_encoder[measurement_id])
        # z = (r - self.n)*np.sin(self.phi[i]) + (self.beam_to_lidar_transform[2,3])


        # Correct for lidar to sensor
        homogeneous = self.lidar_to_sensor_transform @ np.array([[x], [y], [z], [1]])
        homogeneous /= homogeneous[3,0]
        
        return homogeneous.T
    
    def setScanWidth(self, width):
        """ 
        in the event we change the width of the image, we need the encoder counts to change as well
        """
        self.scan_width = width
        self.theta_encoder = 2 * np.pi * (np.ones(self.scan_width) - np.arange(0,self.scan_width) / self.scan_width)



class Frame:
    def __init__(self, img, depth_img, br, lidar):
        self.depth_img = depth_img
        self.img = blur(img)
        self.kp, self.des = br.detectAndCompute(self.img, None)
        self.filter_keypoints()
        self.find_world_coordinates(lidar)


    def filter_keypoints(self):

        # find laplacian of depth_img to determine large deltas in depth

        laplacian = cv2.Laplacian(self.depth_img,cv2.CV_64F)
        laplacian = cv2.GaussianBlur(laplacian, (9,9), 0)

        # add areas that are within ~1m
        laplacian[self.depth_img<200] = 65536

        # dilate
        laplacian = cv2.dilate(laplacian, np.ones((5,5),np.uint8), iterations=4)

        # blur
        laplacian = cv2.GaussianBlur(laplacian, (19,19), 0)

        # Threshold
        self.mask = (laplacian>350.0)

        kp_f = []
        des_f = []

        for i in range(len(self.kp)):

            # pull u, v value from kp descriptor
            u = int(self.kp[i].pt[0])
            v = int(self.kp[i].pt[1])

            if self.mask[v,u] == True:
                continue
            else:
                kp_f.append(self.kp[i])
                des_f.append(self.des[i])

        self.kp = tuple(kp_f)
        self.des = np.uint8(np.asarray(des_f))

        return
    
    def find_match_coordinates(self, matches, query):

        # clear xyz from previous
        self.match_xyz = np.zeros((4, len(matches)))
        kp_match = []
        des_match = []
        

        for i, match in enumerate(matches):
            if query:
                i_m = match.queryIdx
            else:
                i_m = match.trainIdx

            self.match_xyz[:, i] = self.xyz[:, i_m]
            
            kp_match.append(self.kp[i_m])
            des_match.append(self.des[i_m])

        self.match_kp = tuple(kp_match)
        self.match_des = np.uint8(np.asarray(des_match))

        return
    
    def find_world_coordinates(self, lidar):

        self.xyz = np.zeros((4, len(self.kp)))

        for i, keypoint in enumerate(self.kp):
            v = int(keypoint.pt[0])
            u = int(keypoint.pt[1])

            point = lidar.getXYZCoords(u, v, self.depth_img[u,v])

            self.xyz[:, i] = point

        self.xyz[:3, :] *= 0.001

        return


    def filter_inliers(self, mask):
        self.match_xyz = self.match_xyz[:, mask]
        self.match_des = self.match_des[mask]
        self.match_kp = tuple(np.array(self.match_kp)[mask])






def fit_transformation(From, To, W=None):

    From_inliers = np.copy(From)
    To_inliers = np.copy(To)
    W_inliers = np.copy(W)

    M = None

    outlier_final_mask = From[3, :] > 0

    last_sum = 0
    for _ in range(20):

        # Find the transformation
        M = find_transform(From_inliers, To_inliers, W_inliers)
        

        # Find norm differences between points
        diff = np.linalg.norm(To - M @ From, axis=0)
        diff_ = np.linalg.norm(To_inliers - M @ From_inliers, axis=0)
        std = np.std(diff_)
        outlier_mask = diff < 0.8*std+np.average(diff)

        if np.sum(outlier_mask) == last_sum or np.sum(outlier_mask) < 10:
            # no outliers or too few remaining points for another round -> exit
            break

        # update outliers mask with new bad values
        outlier_final_mask = (outlier_final_mask) & (outlier_mask)

        From_inliers = From[:, outlier_final_mask]
        To_inliers = To[:, outlier_final_mask]
        if W.size != 0:
            W_inliers = W[outlier_final_mask]
        last_sum = np.sum(outlier_mask)

    return M, outlier_final_mask



def find_transform(From, To, W=None):
    # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf

    # Reconfigure data into 3xn
    P = From[:3, :]
    Q = To[:3, :]

    # Find centroids
    if W.size != 0:
        P_bar = np.sum(P*W, 1).reshape(-1,1) / np.sum(W)
        Q_bar = np.sum(Q*W, 1).reshape(-1,1) / np.sum(W)
    else:
        P_bar = np.mean(P, 1).reshape(-1,1)
        Q_bar = np.mean(Q, 1).reshape(-1,1)


    # Offset by centroids
    X = P - P_bar
    Y = Q - Q_bar

    # Calculate 3x3 covariance
    if W.size != 0:
        W_diag = np.diag(W)
    else:
        W_diag = np.eye(X.shape[1])
    cov = X @ W_diag @ Y.T

    # Use SVD to calculate the 3x3 Matrices U and V from coveriance
    U, _, V_T = np.linalg.svd(cov); V = V_T.T

    # Find rotation
    m = np.eye(U.shape[0]); m[-1,-1] = np.linalg.det(V @ U.T)
    R = V @ m @ U.T

    # Find translation
    T = Q_bar - R @ P_bar

    M = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1]) ))

    return M




class SubscriberAndPublisher:
    def __init__(self):

        sig_sub_ = message_filters.Subscriber('/ouster/signal_image', Image)
        rng_sub_ = message_filters.Subscriber('/ouster/range_image', Image)

        ts = message_filters.TimeSynchronizer([sig_sub_, rng_sub_], 10)

        self.pub_ = rospy.Publisher('lidar/odom', Odometry, queue_size=50)
        self.my_lidar = Lidar()
        self.br = cv2.BRISK_create(thresh=7, octaves=3, patternScale=1.0)
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.prev_frame = None

        self.seq_ = 0

        self.bridge = CvBridge()

        self.current_transform = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])
        
        ts.registerCallback(self.callback)

        # rospy.spin()

    def callback(self, signal_img_ptr, range_img_ptr):

        signal_img = self.bridge.imgmsg_to_cv2(signal_img_ptr, desired_encoding='bgr8')
        range_img = self.bridge.imgmsg_to_cv2(range_img_ptr, desired_encoding='passthrough')


        this_frame = Frame(signal_img,
                        range_img,
                        self.br,
                        self.my_lidar)

        if self.prev_frame != None:
            
            # Use Flann to identify matches between BRISK descriptions
            preliminary_matches = self.flann.knnMatch(self.prev_frame.des, this_frame.des, k=2)

            # Filter out bad matches below a theshold
            matches = []
            weights = []
            for match in preliminary_matches:
                if len(match) == 0:
                    continue
                m_d = match[0].distance
                if len(match) >= 2:
                    n_d = match[1].distance
                else:
                    n_d = 300

                if m_d < 0.7 * n_d:

                    matches.append(match[0])
                    if False:
                        # Weighted
                        w = max(200 - m_d, 0)
                        weights.append(w)
                    
            weights = np.array(weights)

            self.prev_frame.find_match_coordinates(matches, True)

            this_frame.find_match_coordinates(matches, False)

            M, _ = fit_transformation(this_frame.match_xyz, self.prev_frame.match_xyz, weights)

            # Store transformations and poses
            self.current_transform = self.current_transform @ M


        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = 'os_sensor'
        odom.header.seq = self.seq_
        q = tr.quaternion_from_matrix(self.current_transform)
        t = self.current_transform[:3,3]

        p = Pose()
        p.position.x = t[0]
        p.position.y = t[1]
        p.position.z = t[2]

        p.orientation.x = q[0]
        p.orientation.y = q[1]
        p.orientation.z = q[2]
        p.orientation.w = q[3]

        odom.pose.pose = p

        # # print(Pose(t, q))
        # odom.pose.pose = Pose(t, q)
        # print(odom.pose.pose)

        self.pub_.publish(odom)

        self.seq_ += 1
        self.prev_frame = this_frame
    

if __name__ == '__main__':
    rospy.init_node('lidar_odom_py_node')
    SubscriberAndPublisher()
    rospy.spin()
