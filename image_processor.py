from utils import *
from math import ceil, floor
import matplotlib.pyplot as plt


class ImageProcessor(object):
    def __init__(self, img):
        self.img = img
        self.selected_boxes = []

    def find_bounding_boxes(self, kernel_length, kernel_size, alpha, beta, sort_method):
        horizontal_lines_img, verticle_lines_img, kernel = detect_lines(
            self.img, kernel_length)

        img_final_bin = combine_lines(
            verticle_lines_img, horizontal_lines_img, kernel, alpha, beta)

        (contours, self.bounding_boxes) = find_sorted_contours(
            img_final_bin, sort_method)

        return self.bounding_boxes

    def select_boxes(self, box_selection_method):
        for (x, y, w, h) in self.bounding_boxes:
            if box_selection_method(x, y, w, h):
                self.selected_boxes.append((x, y, w, h))

        return self.selected_boxes

    def vizualize_selected_boxes(self, rows=None, columns=1, figsize=(100, 100), path=None):
        rows = rows or ceil(len(self.selected_boxes) / columns)

        fig, ax = plt.subplots(rows, columns, squeeze=False, figsize=figsize)

        for i, box in enumerate(self.selected_boxes):
            ax[floor(i / columns)][i %
                                   columns].imshow(crop_img(self.img, box))

        if path:
            plt.savefig(f"{path}/detected_boxes.png")
        plt.clf()
