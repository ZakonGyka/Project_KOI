import glob

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((25 * 17, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:25, 0:17].T.reshape(-1, 2)*15

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob('Right_calib/*.PNG')
        images_left = glob.glob('Left_calib/*.PNG')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (25, 17), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (25, 17), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (25, 17),
                                                  corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(1000)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (25, 17),
                                                  corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(1000)
            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('')

        camera_model = dict([('cam_matrix_1', M1), ('cam_matrix_2', M2),
                             ('dist_1', d1), ('dist_2', d2),
                             ('rvecs1', self.r1), ('rvecs2', self.r2),
                             ('R', R), ('T', T), ('E', E), ('F', F)])

        cv2.destroyAllWindows()
        return camera_model


cam_matrix_1 = []
cam_matrix_2 = []
dist_1 = []
dist_2 = []
R = []
T = []

if __name__ == '__main__':
    cal_data = StereoCalibration('/')
    print('end StereoCalibration')
    print('')
    cam_matrix_1 = cal_data.camera_model['cam_matrix_1']
    cam_matrix_2 = cal_data.camera_model['cam_matrix_2']
    dist_1 = cal_data.camera_model['dist_1']
    dist_2 = cal_data.camera_model['dist_2']
    T = cal_data.camera_model['T']
    R = cal_data.camera_model['R']


print('Intrinsic_mtx_1', cam_matrix_1)
print('dist_1', dist_1)
print('Intrinsic_mtx_2', cam_matrix_2)
print('dist_2', dist_2)
print('R', R)
print('T', T)
#print('E', E)
#print('F', F)

print('')
print('stereoRectify')
print('')


R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cam_matrix_1, dist_1,
    cam_matrix_2, dist_2,
    (1920, 1200),
    R, T
)

print('R1 = ', R1)
print('P1 = ', P1)
print('P2 = ', P2)
print('Q = ', Q)
print('')


left_for_matcher = cv2.imread('Box_left_cut.jpg', 0)
right_for_matcher = cv2.imread('Box_right_cut.jpg', 0)

# numDisparities должно быть кратно 16
left_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=7)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

left_disp = left_matcher.compute(left_for_matcher, right_for_matcher)
right_disp = right_matcher.compute(right_for_matcher, left_for_matcher)

'''lmbda - parameter defining the amount
 of regularization during filtering.  8000
'''
lmbda = 8000
'''sigma - parameter defining how sensitive the filtering process
 is to source image edges. from 0.8 to 2.0
 '''
sigma = 1.9

wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
left_disp_filtered = wls_filter.filter(left_disp,
                                       left_for_matcher,
                                       disparity_map_right=right_disp
                                       )

left_disp = np.uint8(left_disp)

# Вывод диспаритета
final_wide = 1200
r = float(final_wide) / left_disp.shape[1]
dim = (final_wide, int(left_disp.shape[0] * r))

# Уменьшаем изображение до подготовленных размеров
resized = cv2.resize(left_disp, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resize raw disparity", resized)

# Вывод отфильтрованого диспаритета
left_disp_filtered = cv2.normalize(src=left_disp_filtered,
                                   dst=left_disp_filtered,
                                   beta=0, alpha=255,
                                   norm_type=cv2.NORM_MINMAX
                                   )
left_disp_filtered = np.uint8(left_disp_filtered)

final_wide = 1200
r = float(final_wide) / left_disp_filtered.shape[1]
dim = (final_wide, int(left_disp_filtered.shape[0] * r))

# Уменьшаем изображение до подготовленных размеров
resized = cv2.resize(left_disp_filtered, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resize filtered_disparity", resized)

# Вывод матриц диспаритетов
print('raw_disparity_matrix = ', left_disp)
print('filtered_disparity_matrix = ', left_disp_filtered)
np.savetxt('filtered_disparity_matrix', left_disp_filtered)

# Вывод 3D точек с фильтрацией диспаритета
_3dImage = cv2.reprojectImageTo3D(left_disp_filtered, Q)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(_3dImage[::70, ::70, 0], _3dImage[::70, ::70, 1], _3dImage[::70, ::70, 2])
plt.show()

disparity_matix_array = np.asarray(left_disp_filtered)
plt.imshow(disparity_matix_array)
plt.show()
# cv2.imshow("filtered_disparity", disparity_matix_array)

# u1 left cam (IMG0092)
object_cam_points_cam_1 = np.array([[283, 775, 1283, 827, 280, 822, 1280],
                                    [718, 408, 555, 894, 982, 1165, 808]
                                    ], dtype=float)  # v1
# u2 right cam (IMG0091)
object_cam_points_cam_2 = np.array([[537, 1183, 1560, 899, 531, 891, 1559],
                                    [598, 407, 611, 883, 846, 1098, 854]
                                    ], dtype=float)  # v2

points4D = cv2.triangulatePoints(P1, P2,
                                 object_cam_points_cam_1,
                                 object_cam_points_cam_2
                                 )

print('points4D', points4D)

dst = cv2.convertPointsFromHomogeneous(points4D.T)
print('dst', dst)
print('')

res_x = []
res_y = []
res_z = []

for point in points4D.T:
    res_x.append(point[0] / point[3])
    res_y.append(point[1] / point[3])
    res_z.append(point[2] / point[3])
    print('res_x = ', res_x)
    print('res_y = ', res_y)
    print('res_z = ', res_z)


distance_length_up =  (((res_x[3] - res_x[2]) ** 2 + (res_y[3] - res_y[2]) ** 2 + (res_z[3] - res_z[2]) ** 2) ** 0.5)/1600
print('distance_length_up', distance_length_up)

distance_height_left =  (((res_x[3] - res_x[5]) ** 2 + (res_y[3] - res_y[5]) ** 2 + (res_z[3] - res_z[5]) ** 2) ** 0.5)/1600
print('distance_height_left', distance_height_left)

distance_width_up_left =  (((res_x[0] - res_x[3]) ** 2 + (res_y[0] - res_y[3]) ** 2 + (res_z[0] - res_z[3]) ** 2) ** 0.5)/1600
print('distance_width_up_left', distance_width_up_left)

cv2.waitKey()
#distance_length_far = (((res_x[0] - res_x[1]) ** 2 + (res_y[0] - res_y[1]) ** 2 + (res_z[0] - res_z[1]) ** 2) ** 0.5)/1600
#print('distance_length_far', distance_length_far)



#distance_length_down =  (((res_x[5] - res_x[6]) ** 2 + (res_y[5] - res_y[6]) ** 2 + (res_z[5] - res_z[6]) ** 2) ** 0.5)/1600
#print('distance_length_down', distance_length_down)



#distance_width_down_left =  (((res_x[4] - res_x[5]) ** 2 + (res_y[4] - res_y[5]) ** 2 + (res_z[4] - res_z[5]) ** 2) ** 0.5)/1600
#print('distance_width_down_left', distance_width_down_left)

#distance_width_right =  (((res_x[1] - res_x[2]) ** 2 + (res_y[1] - res_y[2]) ** 2 + (res_z[1] - res_z[2]) ** 2) ** 0.5)/1600
#print('distance_width_right', distance_width_right)

#distance_height_left_far =  (((res_x[0] - res_x[4]) ** 2 + (res_y[0] - res_y[4]) ** 2 + (res_z[0] - res_z[4]) ** 2) ** 0.5)/1600
#print('distance_height_left_far', distance_height_left_far)



#distance_height_right =  (((res_x[2] - res_x[6]) ** 2 + (res_y[2] - res_y[6]) ** 2 + (res_z[2] - res_z[6]) ** 2) ** 0.5)/1600
#print('distance_height_right', distance_height_right)

'''
distance_length_far = ((res_x[0] - res_x[1]) ** 2 + (res_y[0] - res_y[1]) ** 2 + (res_z[0] - res_z[1]) ** 2) ** 0.5
print('distance_length_far', distance_length_far)

distance_length_up =  ((res_x[3] - res_x[2]) ** 2 + (res_y[3] - res_y[2]) ** 2 + (res_z[3] - res_z[2]) ** 2) ** 0.5
print('distance_length_up', distance_length_up)

distance_length_down =  ((res_x[5] - res_x[6]) ** 2 + (res_y[5] - res_y[6]) ** 2 + (res_z[5] - res_z[6]) ** 2) ** 0.5
print('distance_length_down', distance_length_down)

distance_width_up_left =  ((res_x[0] - res_x[3]) ** 2 + (res_y[0] - res_y[3]) ** 2 + (res_z[0] - res_z[3]) ** 2) ** 0.5
print('distance_width_up_left', distance_width_up_left)

distance_width_down_left =  ((res_x[4] - res_x[5]) ** 2 + (res_y[4] - res_y[5]) ** 2 + (res_z[4] - res_z[5]) ** 2) ** 0.5
print('distance_width_down_left', distance_width_down_left)

distance_width_right =  ((res_x[1] - res_x[2]) ** 2 + (res_y[1] - res_y[2]) ** 2 + (res_z[1] - res_z[2]) ** 2) ** 0.5
print('distance_width_right', distance_width_right)

distance_height_left_far =  ((res_x[0] - res_x[4]) ** 2 + (res_y[0] - res_y[4]) ** 2 + (res_z[0] - res_z[4]) ** 2) ** 0.5
print('distance_height_left_far', distance_height_left_far)

distance_height_left =  ((res_x[3] - res_x[5]) ** 2 + (res_y[3] - res_y[5]) ** 2 + (res_z[3] - res_z[5]) ** 2) ** 0.5
print('distance_height_left', distance_height_left)

distance_height_right =  ((res_x[2] - res_x[6]) ** 2 + (res_y[2] - res_y[6]) ** 2 + (res_z[2] - res_z[6]) ** 2) ** 0.5
print('distance_height_right', distance_height_right)
'''
