import numpy as np
import matplotlib.pyplot as plt

def get_cropped_image(image, middle_pixel_x, middle_pixel_y, crop_limit):
    y_top = middle_pixel_y - crop_limit
    y_bottom = middle_pixel_y + (crop_limit+1)
    x_left = middle_pixel_x - crop_limit
    x_right = middle_pixel_x + (crop_limit+1)
    if y_top < 0 or y_bottom > image.shape[0]:
        return None
    if x_left < 0 or x_right > image.shape[1]:
        return None
    cropped_image = image[y_top:y_bottom, x_left:x_right]
    cropped_image = np.array(cropped_image).astype(np.uint8)
    return cropped_image


def got_tfl(image, tfl_px_code):
    return len(np.argwhere(image == tfl_px_code)) > 0


def get_tfl_data_labels(main_image, gf_fine_image, tfl_xs, tfl_ys, tfl_color_label):
    tfl_px_code = 19
    num_of_tfl = 0
    crop_limit = 40
    labels = []
    cropped_images = []
    for i in range(len(tfl_xs)):
        x = int(tfl_xs[i])
        y = int(tfl_ys[i])

        if gf_fine_image[y][x] == tfl_px_code:
            cropped_image = get_cropped_image(main_image, x, y, crop_limit)
            if cropped_image is not None:
                cropped_images.append(cropped_image)
                labels.append(tfl_color_label)
                num_of_tfl += 1
    return cropped_images, labels, num_of_tfl


def get_no_tfl_data_label(main_image, gf_fine_image, green_xs, green_ys, red_xs, red_ys, tfl_color_label, num_of_tfl):
    # plt.figure()

    # plt.imshow(gf_fine_image)

    tfl_px_code = 19
    crop_limit = 40
    labels = []
    cropped_images = []
    red_idx = 0
    got_red = False
    green_idx = 0
    got_green = False
    got_random = False

    while num_of_tfl > 0:
        while num_of_tfl > 0 and red_idx < len(red_xs) and not got_red:
            x = int(red_xs[red_idx])
            y = int(red_ys[red_idx])
            if gf_fine_image[y][x] != tfl_px_code:
                cropped_image = get_cropped_image(main_image, x, y, crop_limit)
                if cropped_image is not None:
                    cropped_images.append(cropped_image)
                    labels.append(tfl_color_label)
                    num_of_tfl -= 1
                    got_red = True
            red_idx += 1
        while num_of_tfl > 0 and green_idx < len(green_xs) and not got_green:
            x = int(green_xs[green_idx])
            y = int(green_ys[green_idx])
            if gf_fine_image[y][x] != tfl_px_code:
                cropped_image = get_cropped_image(main_image, x, y, crop_limit)
                if cropped_image is not None:
                    cropped_images.append(cropped_image)
                    labels.append(tfl_color_label)
                    num_of_tfl -= 1
                    got_green = True
            green_idx += 1
        while num_of_tfl > 0 and num_of_tfl != 0 and not got_random:
            y = np.random.randint((crop_limit+1), main_image.shape[0] - (crop_limit+1))
            x = np.random.randint((crop_limit+1), main_image.shape[1] - (crop_limit+1))
            if gf_fine_image[y][x] != tfl_px_code:
                cropped_image = get_cropped_image(main_image, x, y, crop_limit)
                if cropped_image is not None:
                    crop_image_fine = get_cropped_image(gf_fine_image, x, y, crop_limit)
                    if not got_tfl(crop_image_fine,tfl_px_code):
                        cropped_images.append(cropped_image)
                        labels.append(tfl_color_label)
                        num_of_tfl -= 1
                        got_random = True
        got_random = False
        got_red = False
        got_green = False

    return cropped_images, labels, num_of_tfl


def make_data_labels(main_image, gf_fine_image, red_xs, red_ys, green_xs, green_ys, step_folder):
    cropped_images = []
    labels = []
    num_of_tfl = 0
    crop_limit = 40
    tfl_px_code = 19
    red_tfl_label = '2'
    green_tfl_label = '1'
    no_tfl_label = '0'

    red_crop_images, red_labels, red_tfl_num = get_tfl_data_labels(main_image, gf_fine_image, red_xs, red_ys, red_tfl_label)
    cropped_images.extend(red_crop_images)
    labels.extend(red_labels)
    num_of_tfl += red_tfl_num

    green_crop_images, green_labels, green_tfl_num = get_tfl_data_labels(main_image, gf_fine_image, green_xs,green_ys, green_tfl_label)
    cropped_images.extend(green_crop_images)
    labels.extend(green_labels)
    num_of_tfl += green_tfl_num
    # print("num of tfl ,")

    no_tfl_crop_images, no_tfl_labels, tfl_num = get_no_tfl_data_label(main_image, gf_fine_image, green_xs, green_ys, red_xs, red_ys, no_tfl_label, num_of_tfl)
    cropped_images.extend(no_tfl_crop_images)
    labels.extend(no_tfl_labels)
    print("must be 0 = ", tfl_num)
    
    if len(cropped_images) != 0:
        f_images = open(step_folder+"/data.bin", "a")
        f_labels = open(step_folder+"/labels.bin", "a")
        np.array(np.array(cropped_images)).tofile(f_images)
        labels = np.array(labels).astype(np.uint8)
        np.array(labels).tofile(f_labels)
        f_images.close()
        f_labels.close()
