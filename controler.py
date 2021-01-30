from tfl_manger import TFL_manger
import shutil
import os
import cv2
from SFM_standAlone import FrameContainer, visualize
import numpy as np
from SFM import calc_TFL_dist
import pickle


class Controler:
    def __init__(self, path):
        self.path = path
        self.pkl_path = ''
        self.image_list = []

    def read_files(self):
        with open(self.path) as file:
            lines = file.readlines()

            for index, line in enumerate(lines):
                line = line.strip('\n')
                if index == 0:
                    self.pkl_path = line
                else:
                    self.image_list.append(line)

    def get_pkl_info(self, pkl_path):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
            pkl_data = data
        focal = data['flx']
        pp = data['principle_point']
        return focal, pp, pkl_data

    def run(self):
        images_base = "Images"
        pkl_path = images_base+'/dusseldorf_000049.pkl'
        focal, pp, pkl_data = self.get_pkl_info(pkl_path)
        tfl = TFL_manger(focal, pp)
        EM = np.eye(4)
        for i in range(len(self.image_list)):
            image_path = self.image_list[i]
            EM = np.dot(pkl_data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            tfl.run(image_path, EM)



cnt = Controler('path.pls')
cnt.read_files()
cnt.run()
