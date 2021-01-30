import math

import cv2
from skimage import feature
from skimage.feature import peak_local_max

try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse
    from scipy import signal
    import skimage.feature

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

from data_test_prep import make_data_labels
print("All imports okay. Yay!")


def get_dist(point1, point2):
    dx = math.pow(point1[0] - point2[0], 2)
    dy = math.pow(point1[1] - point2[1], 2)
    return math.sqrt(dx + dy)


def get_max(point1, point2):
    if max(point1[0], point2[0]) > max(point1[1], point2[1]):
        return (point1[0], point1[1])
    else:
        return (point2[0], point2[1])


def filter_closer(points, threshold):
    processed_points = {}
    for pair in points:
        pair = tuple(pair)
        for point in processed_points:
            dist = get_dist(point, pair)
            if dist < threshold:
                del processed_points[point]
                new_point = get_max(point, pair)
                processed_points[new_point] = new_point
                break
        else:
            processed_points[pair] = pair
    return processed_points


def binarize(image_to_transform, threshold):
    # now, lets convert that image to a single greyscale image using convert()
    output_image = image_to_transform.copy()
    arr = np.array(output_image)
    arr[arr <= threshold] = 0
    arr[arr > threshold] = 255
    return arr


def find_tfl_lights(image, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """

    width, height = (2048, 1024)
    cropped_img = image.copy()
    cropped_img = adjust_gamma(image, 0.2)
    cropped_img = np.array(cropped_img)
    cropped_img[height // 2:height, 0:width] = 0
    red, green, blue = cv2.split(cropped_img)
    kernel = np.array(
        [[0, 1, 0],
         [1, 3, 1],
         [0, 1, 0]])

    kernel = kernel - kernel.mean()
    red, green, blue = cv2.split(cropped_img)
    ret1, otsu1 = cv2.threshold(red, 200, 255, cv2.THRESH_BINARY)
    ret3, otsu2 = cv2.threshold(green, 200, 255, cv2.THRESH_BINARY)
    ret3, otsu3 = cv2.threshold(blue, 200, 255, cv2.THRESH_BINARY)

    result = cv2.filter2D(otsu1, -1, kernel)
    result = ndimage.maximum_filter(result, size=1, mode='constant')
    red_points = peak_local_max(result, min_distance=50)
    result = cv2.filter2D(otsu2, -1, kernel)
    result = ndimage.maximum_filter(result, size=1, mode='constant')
    green_points = peak_local_max(result, min_distance=40)
    result = cv2.filter2D(otsu3, -1, kernel)
    result = ndimage.maximum_filter(result, size=1, mode='constant')
    blue_points = peak_local_max(result, min_distance=40)
    xs_red = []
    ys_red = []
    xs_green = []
    ys_green = []

    for point in red_points:
        curr = image[point[0], point[1]]
        if (curr[0] < 200 and curr[1] >= 230 and curr[2] >= 200):
            xs_green = np.append(xs_green, point[1])
            ys_green = np.append(ys_green, point[0])

        elif ((curr[1] <= 240 and curr[2] <= 230) and (curr[0] >= curr[1] and curr[0] >= 220)):
            xs_red = np.append(xs_red, point[1])
            ys_red = np.append(ys_red, point[0])

    for point in green_points:
        curr = image[point[0], point[1]]

        if (curr[0] >= curr[1] and curr[0] > 220):
            xs_red = np.append(xs_red, point[1])
            ys_red = np.append(ys_red, point[0])


        elif (curr[0] < 200 and curr[1] >= 230 and curr[2] >= 200):
            xs_green = np.append(xs_green, point[1])
            ys_green = np.append(ys_green, point[0])

    for point in blue_points:
        curr = image[point[0], point[1]]

        if (curr[0] < 200 and curr[1] >= 230 and curr[2] >= 200):
            xs_green = np.append(xs_green, point[1])
            ys_green = np.append(ys_green, point[0])

        if ((curr[1] <= 240 and curr[2] <= 230) and (curr[0] >= curr[1] and curr[0] >= 220)):
            xs_red = np.append(xs_red, point[1])
            ys_red = np.append(ys_red, point[0])

    return xs_red, ys_red, xs_green, ys_green


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()

    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, label_id_path=None, fig_num=None,step_folder=None):
    """
    Run the attention code
    """

    image = np.array(Image.open(image_path))
    # gf_fine_image = np.array(Image.open(label_id_path))
    gf_fine_image =None

    # show_image_and_gt(gf_fine_image, None, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)

    make_data_labels(image,red_x, red_y, green_x, green_y, step_folder)

    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    Dir_data_path = 'C:/Users/loay-/Desktop/Mobileye/leftImg8bit'
    # data_step_forlders = ["train", "val"]
    data_step_forlders = ["test"]
    print("start data preprations")
    for step_folder in data_step_forlders:

        cities = [x for x in os.listdir(Dir_data_path+"/"+step_folder)]
        for city in cities:
            print(f"start {city}")

            if args.dir is None:
                args.dir = Dir_data_path+"/"+step_folder+"/"+city
            flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
            for image in flist:
            #     json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
            #     label_id_fn = image.replace('_leftImg8bit.png', '_gtFine_labelIds.png')
            #     if not os.path.exists(json_fn):
            #         json_fn = None

                test_find_tfl_lights(image, json_path=None, label_id_path=None, fig_num=None, step_folder=step_folder)
            if len(flist):
                print("You should now see some images, with the ground truth marked on them. Close all to quit.")
            else:
                print("Bad configuration?? Didn't find any picture to show")
            # plt.show(block=True)
    print("finish data preprations")

print("start")

if __name__ == '__main__':
    main()
