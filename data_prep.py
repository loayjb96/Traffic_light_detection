import numpy as np
import matplotlib.pyplot as plt


def get_cropped_image(image, middle_pixel_x, middle_pixel_y, crop_limit):
    y_top = middle_pixel_y - crop_limit
    y_bottom = middle_pixel_y + (crop_limit+1)
    x_left = middle_pixel_x - (crop_limit)
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


def make_data_labels(main_image, gf_fine_image, red_xs, red_ys, green_xs, green_ys, step_folder):
    cropped_images = []
    labels = []
    num_of_tfl = 0
    crop_limit = 40
    tfl_px_code = 19
    red_tfl = '2'
    green_tfl = '1'
    no_tfl = '0'


    for i in range(len(red_xs)):
        x = int(red_xs[i])
        y = int(red_ys[i])

        if gf_fine_image[y][x] == tfl_px_code:
            cropped_image = get_cropped_image(main_image, x, y, crop_limit)
            if cropped_image is not None:
                cropped_images.append(cropped_image)
                labels.append(red_tfl)
                num_of_tfl += 1
    for i in range(len(green_xs)):
        x = int(green_xs[i])
        y = int(green_ys[i])

        if gf_fine_image[y][x] == tfl_px_code:
            cropped_image = get_cropped_image(main_image, x, y, crop_limit)
            if cropped_image is not None:
                cropped_images.append(cropped_image)
                labels.append(green_tfl)
                num_of_tfl += 1
    while num_of_tfl != 0:

        y = np.random.randint((crop_limit+1), main_image.shape[0] - (crop_limit+1))
        x = np.random.randint((crop_limit+1), main_image.shape[1] - (crop_limit+1))
        if gf_fine_image[y][x] != tfl_px_code:
            cropped_image = get_cropped_image(main_image, x, y)
            if cropped_image is not None:
                crop_imag_fine = get_cropped_image(gf_fine_image, x, y)
                if not got_tfl(crop_imag_fine,tfl_px_code):
                    cropped_images.append(cropped_image)
                    labels.append(no_tfl)
                    num_of_tfl -= 1
    if len(cropped_images) != 0:
        f_images = open(step_folder+"/data.bin", "a")
        f_labels = open(step_folder+"/labels.bin", "a")
        np.array(np.array(cropped_images)).tofile(f_images)
        labels = np.array(labels).astype(np.uint8)
        np.array(labels).tofile(f_labels)
        f_images.close()
        f_labels.close()
