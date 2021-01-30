import numpy as np
import matplotlib.pyplot as plt

# the test data has no labels so we need to crop the image that came from pahe 1 (run_attention_as.py)
#even though its false possitive

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


def make_data_labels(main_image, red_xs, red_ys, green_xs, green_ys, step_folder,frame_num):
    cropped_images = []
    labels = []
    num_of_tfl = 0
    crop_limit = 40

    red_tfl = '2'
    green_tfl = '1'
    no_tfl = '0'

    for i in range(len(red_xs)):
        x = int(red_xs[i])
        y = int(red_ys[i])

        cropped_image = get_cropped_image(main_image, x, y, crop_limit)
        if cropped_image is not None:
            cropped_images.append(cropped_image)
            labels.append(red_tfl)
            num_of_tfl += 1

    for i in range(len(green_xs)):
        x = int(green_xs[i])
        y = int(green_ys[i])

        cropped_image = get_cropped_image(main_image, x, y, crop_limit)
        if cropped_image is not None:
            cropped_images.append(cropped_image)
            labels.append(green_tfl)
            num_of_tfl += 1

    if len(cropped_images) != 0:

        with open(step_folder+"/data_image"+frame_num+".bin", "ab") as f_images:
            print(step_folder+"/data_image"+frame_num+".bin")

            np.array(np.array(cropped_images)).tofile(f_images)
            labels = np.array(labels).astype(np.uint8)
            # f_images.close()