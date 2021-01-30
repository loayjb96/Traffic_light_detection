import numpy as np
import matplotlib.pyplot as plt
import SFM

def visualize(tfl):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(tfl.prev_container, tfl.curr_container, tfl.focal, tfl.pp)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = SFM.unnormalize(norm_rot_pts, tfl.focal, tfl.pp)
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), tfl.focal, tfl.pp))
    
    fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12,6))
    prev_sec.set_title('prev(' + tfl.prev_container.frame_id + ')')
    prev_sec.imshow(tfl.prev_container.img)
    prev_p = tfl.prev_container.traffic_light
    prev_sec.plot(prev_p[:,0], prev_p[:,1], 'b+')

    curr_sec.set_title('curr(' + tfl.curr_container.frame_id + ')')
    curr_sec.imshow(tfl.curr_container.img)
    curr_p = tfl.curr_container.traffic_light
    curr_sec.plot(curr_p[:,0], curr_p[:,1], 'b+')

    for i in range(len(curr_p)):
        curr_sec.plot([curr_p[i,0], foe[0]], [curr_p[i,1], foe[1]], 'b')
        if tfl.curr_container.valid[i]:
            curr_sec.text(curr_p[i,0], curr_p[i,1], r'{0:.1f}'.format(tfl.curr_container.traffic_lights_3d_location[i, 2]), color='r')
    curr_sec.plot(foe[0], foe[1], 'r+')
    curr_sec.plot(rot_pts[:,0], rot_pts[:,1], 'g+')
    plt.show()    
    
class FrameContainer(object):
    def __init__(self, img_path):
        postfix_len = len("_leftImg8bit.png")
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
        frame = img_path[::-1][postfix_len:]
        num_idx = frame.find("_")
        frame = frame[0:num_idx][::-1]
        self.frame_id = frame
