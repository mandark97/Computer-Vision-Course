
from collections import defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from utils import *

LABELS = ["A", "B", "C", "D"]


class QuestionBoxEvaluator(object):
    def __init__(self, question_box, check_boxes, logger=None):
        self.question_box = question_box
        self.check_boxes = check_boxes
        self.rows = defaultdict(list)
        self.logger = logger

    def evaluate(self, offset=0):
        self.__detect_rows()
        self.__detect_reference_header()
        # pprint(self.rows, self.logger)

        answers = {}

        for index, (row, values) in enumerate(self.rows.items()):
            correct_answer = self.__compute_answer(values)
            answers[offset + index + 1] = LABELS[correct_answer]

        return answers

    def __detect_rows(self):
        sorted_coords = sorted(self.check_boxes, key=lambda c: (c[1], c[0]))
        arr = np.array(self.check_boxes)[:, 1]
        arr = sorted(arr)
        distances = []
        for i in range(1, len(arr)):
            distances.append(arr[i] - arr[i-1])

        sorted_indexes = np.argsort(distances)
        sorted_indexes = sorted(sorted_indexes[-15:])

        old_index = 0
        group = {}
        for i in range(len(sorted_indexes)):
            group[i] = sorted_coords[old_index:sorted_indexes[i]+1]
            old_index = sorted_indexes[i] + 1

        group[len(sorted_indexes)] = sorted_coords[old_index: -1]

        for k, v in group.items():
            group[k] = sorted(v, key=lambda x: x[0])

        self.rows = group

    def __detect_reference_header(self):
        header = self.rows.pop(0)
        if len(header) != 4:
            for _, v in self.rows.items():
                if len(v) == 4:
                    header = v
                    break

        self.header = np.array(header)

    def __compute_answer(self, values):
        if len(values) != 4:
            translated_header = self.header.copy()
            translated_header[:, 1] = values[0][1]
            correct_answer, means = self.__marked_box(translated_header)

        else:
            correct_answer, means = self.__marked_box(values)

        # pprint(means, self.logger)
        return correct_answer

    def __marked_box(self, values):
        means = [crop_img(self.question_box, value).mean() for value in values]
        return np.argmax(means), means

    def vizualize_rows(self, title="", path=None):
        fig, ax = plt.subplots(len(self.rows))
        fig.suptitle(title)
        plt.subplots_adjust(hspace=1)

        index = 0
        for row, values in sorted(self.rows.items()):
            # if len(values) != 4:
            #     values = self.header.copy()
            #     values[:, 1] = values[0][1]

            row_img = self.__concatenate_row_imgs(values)
            ax[index].set_title(
                f"Row: {row}, Boxes: {len(values)}, {values}")
            ax[index].imshow(row_img)
            index = index + 1

        plt.show()

    def vizualize_checkboxes(self):
        img = cv2.cvtColor(self.question_box, cv2.COLOR_GRAY2BGR)

        for row, values in self.rows.items():
            for v in values:
                img = cv2.rectangle(
                    img, (v[0], v[1]), (v[0] + v[2], v[1] + v[3]), (255, 0, 0), 3)
                cv2.putText(img, f'{row}', (v[0]+5, v[1]+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        plt.imshow(img)
        plt.show()

    def __concatenate_row_imgs(self, values):
        row_img = []
        if len(values) == 0:
            return [[]]
        max_h = np.max(np.array(values)[:, 3])
        means = []
        for (x, y, w, h) in values:
            empty_arr = np.full((max_h, w), 0)
            empty_arr[0:h, 0:w] = crop_img(self.question_box, (x, y, w, h))

            means.append(empty_arr.mean())
            row_img.append(empty_arr)
            row_img.append(np.full((max_h, 20), 255))

        return np.concatenate(row_img, axis=1)
