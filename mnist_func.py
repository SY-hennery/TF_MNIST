import numpy as np
import matplotlib.pyplot as plt

def parse_lables(labels):
    labels_list = labels.tolist()
    return labels_list[0].index(max(labels_list[0]))

def parse_image(images):
    curr_img = np.reshape(images[0],(28,28))
    plt.matshow(curr_img,cmap = plt.get_cmap('gray'))
    plt.savefig("image.png")
