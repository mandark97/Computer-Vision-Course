import pdb
from utils import *
import os
import shutil
from collections import defaultdict
from math import ceil, floor
from pprint import pprint

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import matplotlib
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from math import floor, ceil
matplotlib.rcParams['image.cmap'] = 'gray'

LABELS = ["A", "B", "C", "D"]


class ImageProcessor(object):
    def __init__(self, img):
        self.img = img
        self.selected_boxes = []

    def find_bounding_boxes(self, kernel_length, kernel_size, alpha, beta, sort_method):
        horizontal_lines_img, verticle_lines_img, kernel = detect_lines(
            self.img, kernel_length)
        # fig, ax = plt.subplots(ncols=2)
        # ax[0].imshow(horizontal_lines_img)
        # ax[1].imshow(verticle_lines_img)

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

    def vizualize_selected_boxes(self, rows=None, columns=1, **kwargs):
        rows = rows or ceil(len(self.selected_boxes) / columns)

        fig, ax = plt.subplots(rows, columns, squeeze=False, **kwargs)

        for i, box in enumerate(self.selected_boxes):
            ax[floor(i / columns)][i %
                                   columns].imshow(crop_img(self.img, box))


class ExamEvaluator(object):
    def __init__(self, question_box, check_boxes):
        self.question_box = question_box
        self.check_boxes = check_boxes
        self.rows = defaultdict(list)

    def detect_rows(self):
        self.rows = test(self.check_boxes)

    def evaluate(self, offset=0):
        # import pdb; pdb.set_trace()
        header = self.rows.pop(0)
        if len(header) != 4:
            for _, v in self.rows.items():
                if len(v) == 4:
                    header = v
                    break

        header = np.array(header)
        x_coords = header[:, 0]

        answers = {}
        # pprint(self.rows)
        for index, (row, values) in enumerate(self.rows.items()):
            answer, means = self.marked_box(values)
            max_mean = np.max(means)
            # if offset + index + 1 == 18:
            #     import pdb; pdb.set_trace()
            # not stupid if it works
            if len(values) != 4:
                smecherie = header.copy()
                smecherie[:, 1] = values[0][1]
                correct_answer, _ = self.marked_box(smecherie)
            else:
                correct_answer = answer
            # if max_mean < 15:
            #     if len(values) == 3:
            #         detected_x = np.array(values)[:, 0]
            #         manevra = x_coords.copy()

            #         for x in detected_x:
            #             ceva = np.argmin([abs(x_coord - x)
            #                               for x_coord in manevra])
            #             manevra = np.delete(manevra, ceva)

            #         correct_answer = list(x_coords).index(manevra[0])

            #     else:
            #         # print(
            #         #     f"question {offset + index + 1} meh meh {len(values)}")
            #         correct_answer = answer

            # else:
            #     answer_x = values[answer][0]
            #     correct_answer = np.argmin(
            #         [abs(x_coord - answer_x) for x_coord in x_coords])

            answers[offset + index + 1] = LABELS[correct_answer]
            # print(
            #     f"Question {index + 1} answer: {LABELS[correct_answer]}, boxes_detected: {len(values)} first answer {answer}  row: {row}")
        # plt.show()

        return answers

    def vizualize_rows(self, title=""):
        fig, ax = plt.subplots(len(self.rows), figsize=(100, 100))
        fig.suptitle(title)
        plt.subplots_adjust(hspace=1)

        index = 0
        for row, values in sorted(self.rows.items()):
            row_img = self.concatenate_row_imgs(values)
            ax[index].set_title(
                f"Row: {row}, Boxes: {len(values)}")
            ax[index].imshow(row_img)
            index = index + 1

    def concatenate_row_imgs(self, values):
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

        return np.concatenate(row_img, axis=1)

    def marked_box(self, values):
        means = [crop_img(self.question_box, value).mean() for value in values]
        return np.argmax(means), means


def evaluate_exam(img_path):
    img = cv2.imread(img_path)
    img_bin = img_to_bin(img)
    img_processor = ImageProcessor(img_bin)
    img_processor.find_bounding_boxes(kernel_length=70,
                                      kernel_size=(3, 3),
                                      alpha=0.5,
                                      beta=0.5,
                                      sort_method="left-to-right")
    question_boxes = img_processor.select_boxes(
        lambda x, y, w, h: w > 900 and h > 1500 and h < 3000)
    img_processor.vizualize_selected_boxes(columns=2, figsize=(100, 100))

    final_answers = {}
    for i, q in enumerate(question_boxes):
        question_box = crop_img(img_processor.img, q)
        question_box_processor = ImageProcessor(question_box)
        question_box_processor.find_bounding_boxes(kernel_length=700,
                                                   kernel_size=(3, 3),
                                                   alpha=0.5,
                                                   beta=0.5,
                                                   sort_method="left-to-right")
        check_boxes = question_box_processor.select_boxes(
            lambda x, y, w, h: w > 90 and h > 90 and h < 150 and w < 200)
        exam_eval = ExamEvaluator(question_box, check_boxes)
        exam_eval.detect_rows()
        exam_eval.vizualize_rows(str(i))
        answers = exam_eval.evaluate(i * 15)
        final_answers.update(answers)

    return final_answers


def load_answers(file):
    correct_answers = {}
    with open(file) as ans:
        ceva = ans.read()
        answers = ceva.splitlines()
        answers.pop(0)
        answers.pop(-1)

        for ans in answers:
            key, val = ans.split(' ')
            correct_answers[int(key)] = val

    return correct_answers

# correct_answers = load_answers(ANSWERS)


def evaluate_answers(final_answers, correct_answers):
    correct = 0
    for k, v in correct_answers.items():
        if final_answers[k] == v:
            correct = correct + 1
        else:
            print(f"Question {k} Detected {final_answers[k]} Correct: {v}")

    print(f"{correct}/{len(correct_answers)}")

    if correct == len(correct_answers):
        return True
    else:
        return False

# FILENAME = "data/exemple_corecte/image_1.jpg"
# ANSWERS = "data/exemple_corecte/image_1.txt"


def run_for_img(i, debug=True):
    try:
        img_path = f"data/exemple_corecte/image_{i}.jpg"
        ans_path = f"data/exemple_corecte/image_{i}.txt"
        print("Running for ", i)

        exam_answers = evaluate_exam(img_path)
        correct_answers = load_answers(ans_path)
        if debug:
            # pprint(exam_answers)
            # pprint(correct_answers)
            plt.show()
        return evaluate_answers(exam_answers, correct_answers)
    except:
        print(f"Failed at {i}")
        return False


results = [run_for_img(i, debug=False) for i in range(1, 151)]
print(sum(results), "/", len(results))
# run_for_img(17, debug=False)
