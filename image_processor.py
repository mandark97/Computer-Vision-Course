from utils import *
from math import ceil, floor
import matplotlib.pyplot as plt


class ImageProcessor(object):
    def __init__(self, img):
        self.img = img
        self.selected_boxes = []

    # useless
    def find_bounding_boxes(self, kernel_length, kernel_size, alpha, beta, sort_method):
        # horizontal_lines_img, verticle_lines_img, kernel = detect_lines(
        #     self.img, kernel_length)
        # img_final_bin = combine_lines(
        #     verticle_lines_img, horizontal_lines_img, kernel, alpha, beta)
        (thresh, img_bin) = cv2.threshold(self.img, 0, 255,
                                          cv2.THRESH_BINARY)
        (contours, self.bounding_boxes) = find_sorted_contours(img_bin, sort_method)

        return self.bounding_boxes

    # useless
    def sort_bounding_boxes(self, sort_method):
        (contours, self.bounding_boxes) = find_sorted_contours(self.img, sort_method)
        return self.bounding_boxes

    def select_boxes(self, box_selection_method, sort_method="left-to_right"):
        (contours, self.bounding_boxes) = find_sorted_contours(self.img, sort_method)
        select_boxes = []
        for (x, y, w, h) in self.bounding_boxes:
            if box_selection_method(x, y, w, h):
                selected_boxes.append((x, y, w, h))

        return selected_boxes

    def vizualize_selected_boxes(self, select_boxes, rows=None, columns=1, figsize=(100, 100), path=None):
        rows = rows or ceil(len(selected_boxes) / columns)

        fig, ax = plt.subplots(rows, columns, squeeze=False, figsize=figsize)

        for i, box in enumerate(selected_boxes):
            ax[floor(i / columns)][i %
                                   columns].imshow(crop_img(self.img, box))

        if path:
            plt.savefig(f"{path}/detected_boxes.png")
        plt.clf()
