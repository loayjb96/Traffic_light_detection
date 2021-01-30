import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_curr_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec

# transform pixels into normalized pixels using the focal length and principle point
# pts - point of 2-D (x,y)
# focal - distance between sensor and pinhole
# pp - principle point of 2-D (x,y)
def normalize(pts, focal, pp):
    focal_threshold = 10e-6
    points = []
    for i in range(len(pts)):
        if focal < focal_threshold:
            x = 0
            y = 0
        else:
            x = (pts[i, 0] - pp[0]) / focal
            y = (pts[i, 1] - pp[1]) / focal
        points.append([x, y])
    return np.array(points)

# transform normalized pixels into pixels using the focal length and principle point
# pts - normalized point of 2-D (x,y)
# focal - distance between sensor and pinhole
def unnormalize(pts, focal, pp):
    points = []
    for i in range(len(pts)):
        x = pts[i, 0] * focal + pp[0]
        y = pts[i, 1] * focal + pp[1]
        points.append([x, y])
    return np.array(points)

# extract R, foe and tZ from the Ego Motion
# [[Rx,Ry,Rz,tx]
#  [Rx,Ry,Rz,ty]
#  [Rx,Ry,Rz,tz]
#  [0, 0, 0,1]]
def decompose(EM):
    tz_threshold = 10e-6
    print(EM)
    R = EM[:3, :3]
    tZ = EM[2, 3]
    # if abs(tZ) < tz_threshold:
    #     foe = []
    # else:
    foe = np.array([EM[0, 3], EM[1, 3]]) / tZ
    return R, foe, tZ

# rotate the points - pts using R
# according to the equation given to us
# R - rotation matrix of 3X3
# pts - normalized point of 2-D (X,Y)
def rotate(pts, R):
    rotation_pts = []
    for point in pts:
        prev_x_y_vec = np.array([point[0], point[1], 1])
        a_b_c = R.dot(prev_x_y_vec)
        Xr_Yr = a_b_c / a_b_c[2]
        rotation_pts.append(Xr_Yr)
    return np.array(rotation_pts)


# compute the epipolar line between p and foe
# run over all norm_pts_rot and find the one closest to the epipolar line
# return the closest point and its index
# pp - principle point of 2-D (x,y)
# norm_pts_rot - list of normalized points od 2-D (x,y)
# foe - normalized point of 2-D (x,y)
def find_corresponding_points(p, norm_pts_rot, foe):
    closest_p = None
    closest_idx = None
    if norm_pts_rot == [] or foe == [] or p ==[]:
        return closest_idx, closest_p
    m = (foe[1] - p[1]) / (foe[0] - p[0])  # slope
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])  # intersection point
    sqrt_d = np.sqrt(m * m + 1)
    dist = {}
    for point in norm_pts_rot:
        distance = abs((m * point[0] + n - point[1]) / sqrt_d)
        dist[tuple(point)] = distance
    closest_p = list(min(dist, key=lambda k: dist[k]))
    closest_idx = np.where(norm_pts_rot == closest_p)
    return closest_idx, closest_p


# we calculate the distance (Z) according to the two axes,
# we'll get two different results,
# we'll go by the rule : as long as the distance is bigger then the result will be more accurate.
# therefore for each axis will give an appropriate weight in accordance.
# since in every data there might be some fault , if we detect a strange
# distances in any axes  (meaning huge differance between Z_x and Z_y)
# we'll return the average distance between the two.

def calc_dist(p_curr, p_rot, foe, tZ):
    if not p_rot:
        return 0
    noise_threshold = 5
    sum_dist_threshold = 10e-6

    Z_x = tZ * (foe[0]-p_rot[0]) / (p_curr[0]-p_rot[0])
    Z_y = tZ * (foe[1]-p_rot[1]) / (p_curr[1]-p_rot[1])
    dist_x = abs((p_curr[0]-p_rot[0]))
    dist_y = abs((p_curr[1]-p_rot[1]))
    sum_dist = dist_x + dist_y
    if sum_dist < sum_dist_threshold:
        return 0
    # if abs(dist_x-dist_y) > noise_threshold:
    #
    #     # we conclude there is some noise in the destination (Z) in one of the axes,
    #     # there for we return the average distention
    # return (Z_x + Z_y)/2

    return (Z_x * dist_x + Z_y * dist_y)/sum_dist
