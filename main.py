import os
import pdb
from collections import defaultdict
from pprint import pprint

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from exam_evaluator import ExamEvaluator
from image_processor import ImageProcessor
from utils import *

matplotlib.rcParams['image.cmap'] = 'gray'


def evaluate_exam(img_path, logger=None, debug_path=None):
    # 5847,4132, 3
    img = cv2.imread(img_path)
    img_bin = img_to_bin(img)

    img_processor = ImageProcessor(img_bin)
    question_boxes = img_processor.select_boxes(
        box_selection_method=lambda x, y, w, h: w > 900 and h > 1500 and h < 3000, sort_method="left-to-right")
    option_boxes = img_processor.select_boxes(
        box_selection_method=lambda x, y, w, h: x > 2500 and y < 3000 and y > 2500 and w < 200 and h > 90 and h < 150,
        sort_method="top-to-bottom")
    img_processor.vizualize_selected_boxes(question_boxes,
        columns=2, figsize=(100, 100), path=debug_path)

    final_answers = {}
    for i, q in enumerate(question_boxes):
        question_box = crop_img(img_processor.img, q)
        question_box_processor = ImageProcessor(question_box)
        check_boxes = question_box_processor.select_boxes(
            box_selection_method=lambda x, y, w, h: w > 90 and h > 90 and h < 150 and w < 200, sort_method="left_to_right")
        exam_eval = ExamEvaluator(question_box, check_boxes, logger=logger)
        answers = exam_eval.evaluate(i * 15)
        final_answers.update(answers)

        if debug_path:
            exam_eval.vizualize_rows(str(i), debug_path)

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


def evaluate_answers(final_answers, correct_answers, logger=None):
    correct = 0
    for k, v in correct_answers.items():
        if final_answers[k] == v:
            correct = correct + 1
        else:
            pprint(
                f"Question {k} Detected {final_answers[k]} Correct: {v}", logger)

    pprint(f"{correct}/{len(correct_answers)}", logger)
    print(f"{correct}/{len(correct_answers)}")
    if correct == len(correct_answers):
        return True
    else:
        return False


def run_for_img(i, debug=False):
    try:
        img_path = f"data/exemple_corecte/image_{i}.jpg"
        ans_path = f"data/exemple_corecte/image_{i}.txt"
        print("Running for ", i)

        if debug:
            os.makedirs(f"debug/{i}", exist_ok=True)
            logger = open(f"debug/{i}/output.txt", "w")
            debug_path = f"debug/{i}"
        else:
            logger = None
            debug_path = None

        exam_answers = evaluate_exam(
            img_path, logger=logger, debug_path=debug_path)
        correct_answers = load_answers(ans_path)

        return evaluate_answers(exam_answers, correct_answers, logger)
    except Exception as error:
        print(error)
        print(f"Failed at {i}")
        return False
    finally:
        if logger:
            logger.close()


if __name__ == "__main__":
    run_for_img(13, debug=True)
    # results = [run_for_img(i, debug=False) for i in range(1, 151)]
    # print(f"{sum(results)}/{len(results)}")
