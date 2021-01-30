from os.path import join
import matplotlib.pyplot as plt
import numpy as np


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Green TFL', 2: 'Red TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)


def get_labels(folder):
    with open(folder+'/labels.bin', 'rb') as fid:
        data = list(fid.read())
        print(len(data))
        no=0
        green=0
        red=0
        for val in data:
            if val == 0:
                no+=1
            if val == 1:
                green+=1
            if val == 2:
                red+=1
        print("green ",green)
        print("red ",red)
        print("no ",no)


def show_data(folder):
    data = load_tfl_data(folder+"/")
    print(data['images'].shape)
    viz_my_data(num=(6, 6), **data)
    get_labels(folder)
    plt.show()


#show_data("val")
show_data("val")