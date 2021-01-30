import numpy as np
from run_attention_as import find_tfl_lights, make_data_labels
import cv2
import keras
from SFM_standAlone import FrameContainer
from SFM import calc_TFL_dist
from stats import timer_decorator
from visualize import visualizer
import matplotlib.pyplot as plt
# from visualize import visualize_prediction

class TFL_manger():
    def __init__(self,focal, pp):
        self.img = None
        self.image_path = None
        self.traffic_light = []
        self.EM = []
        self.corresponding_ind = []
        self.focal = focal
        self.pp = pp
        self.points_data = {}
        self.prev_container = None
        self.curr_container = None

    def update_prev(self, frame, curr_data, img_path):
        self.prev_container.frame_id = frame
        self.points_data['points_' + self.curr_container.frame_id] = curr_data
        self.prev_container.frame_id = self.curr_container.frame_id
        prev_img_path = img_path
        prev_container = FrameContainer(prev_img_path)
        prev_container.traffic_light = np.array(self.points_data['points_' + self.prev_container.frame_id][0])
        return prev_container

    @timer_decorator
    def run_SFM(self, xs_red, ys_red, xs_green, ys_green,predictions,EM):
        curr_data = self.data_prep(xs_red, ys_red, xs_green, ys_green, predictions)
        self.points_data['points_' + self.curr_container.frame_id] = curr_data
        self.curr_container.traffic_light = np.array(self.points_data['points_' + self.curr_container.frame_id][0])

        if self.prev_container:
            self.curr_container.EM = EM
            self.curr_container = calc_TFL_dist(self.prev_container, self.curr_container, self.focal, self.pp)
        self.prev_container = FrameContainer(self.image_path)
        self.prev_container = self.update_prev(self.curr_container.frame_id, curr_data, self.image_path)
        return self.curr_container

    def run(self, image_path, EM):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.curr_container = FrameContainer(image_path)
        # part 1
        xs_red, ys_red, xs_green, ys_green = self.detect_tfl()
        # part 2
        predictions = self.run_CNN()
        # part 3
        curr_container = self.run_SFM(xs_red, ys_red, xs_green, ys_green, predictions,EM)
        # visualize
        vs = visualizer(self, (xs_red,ys_red),(xs_green,ys_green),curr_container,predictions)
        vs.run()



    @timer_decorator
    def detect_tfl(self):
        xs_red, ys_red, xs_green, ys_green = find_tfl_lights(self.img)
        make_data_labels(self.img, xs_red, ys_red, xs_green, ys_green, 'data_image',self.curr_container.frame_id)
        return xs_red, ys_red, xs_green, ys_green

    def load_test_data(self, data_dir, crop_shape=(81, 81)):
        images = np.memmap(data_dir, mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
        return {'images': images}

    @timer_decorator
    def run_CNN(self):
        with open("Model/model.json", 'r') as j:
            loaded_json = j.read()

        # load the model architecture:
        loaded_model = keras.models.model_from_json(loaded_json)
        # load the weights:
        loaded_model.load_weights('Model/model.h5')
        data_bin_file = "data_image"+"/data_image"+self.curr_container.frame_id+".bin"
        datasets = {
            'test': self.load_test_data(data_bin_file)
        }

        test = datasets['test']
        test_predictions = loaded_model.predict(test['images'])
        return test_predictions

    def data_prep(self, xs_red, ys_red, xs_green, ys_green,predictions):
        predict_threshold = 0.85
        data_positions = []
        data_tfl_color = []
        for i in range(len(xs_red)):
            if predictions[i][2] >= predict_threshold:
                data_positions.append([xs_red[i], ys_red[i]])
                data_tfl_color.append("red")
        for i in range(len(xs_green)):
            if predictions[i][1] >= predict_threshold:
                data_positions.append([xs_green[i], ys_green[i]])
                data_tfl_color.append("green")
        return (data_positions,data_tfl_color)





