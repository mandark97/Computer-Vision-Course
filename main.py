from pprint import pprint

import matplotlib

from image_processor import ImageProcessor
from number_detection import OptionDetection
from question_box_evaluator import QuestionBoxEvaluator
from utils import *

matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['figure.figsize'] = 20, 20
IMAGE = "image"
PERSPECTIVE = "perspective"
ROTATION = "rotation"

template = cv2.imread("data/template.jpg", 0)


def evaluate_exam(img_path, explain=False):
    # 5847,4132, 3
    img = cv2.imread(img_path)
    img, _ = homography(img, template)

    img_bin = img_to_bin(img)

    img_processor = ImageProcessor(img_bin)
    question_boxes = img_processor.select_boxes(
        box_selection_method=lambda x, y, w, h: w > 900 and h > 1500 and h < 3000,
        sort_method="left-to-right")

    option_boxes = img_processor.select_boxes(
        box_selection_method=lambda x, y, w, h: x > 2500 and y < 3000 and y > 2500 and w < 200 and h > 90 and h < 150,
        sort_method="top-to-bottom")

    final_answers = {}
    if explain:
        img_processor.vizualize_selected_boxes(question_boxes,
                                               columns=2)
        img_processor.vizualize_selected_boxes(
            option_boxes, columns=2, border=18)

    exam_number = OptionDetection(img_bin, option_boxes).evaluate()
    final_answers.update(exam_number)
    for i, q in enumerate(question_boxes):
        question_box = crop_img(img_processor.img, q)
        question_box_processor = ImageProcessor(question_box)
        check_boxes = question_box_processor.select_boxes(
            box_selection_method=lambda x, y, w, h: w > 90 and h > 90 and h < 150 and w < 200,
            sort_method="left_to_right")
        exam_eval = QuestionBoxEvaluator(
            question_box, check_boxes)
        answers = exam_eval.evaluate(i * 15)
        final_answers.update(answers)
        if explain:
            exam_eval.vizualize_checkboxes()
            exam_eval.vizualize_rows(str(i))

    return final_answers


def load_answers(file):
    correct_answers = {}
    with open(file) as ans:
        ceva = ans.read()
        answers = ceva.splitlines()
        subject, subject_number = answers.pop(0).split(" ")
        correct_answers["subject"] = subject
        correct_answers["subject_number"] = int(subject_number)
        answers.pop(-1)

        for ans in answers:
            key, val = ans.split(' ')
            correct_answers[int(key)] = val

    return correct_answers


def evaluate_answers(final_answers, correct_answers):
    correct = 0
    if final_answers["subject"] != correct_answers["subject"] or final_answers["subject_number"] != correct_answers["subject_number"]:
        print(
            f"detected: {final_answers['subject']} {final_answers['subject_number']}, correct_answers: {correct_answers['subject']} {correct_answers['subject_number']}")
    for k, v in correct_answers.items():
        if final_answers[k] == v:
            correct = correct + 1
        else:
            pprint(f"Question {k} Detected {final_answers[k]} Correct: {v}")

    pprint(f"{correct}/{len(correct_answers)}")
    print(f"{correct}/{len(correct_answers)}")
    if correct == len(correct_answers):
        return True
    else:
        return False


def run_for_img(i, image_type=IMAGE, explain=False):
    try:
        img_path = f"data/exemple_corecte/{image_type}_{i}.jpg"
        ans_path = f"data/exemple_corecte/image_{i}.txt"
        print("Running for ", i)

        exam_answers = evaluate_exam(
            img_path, explain)
        correct_answers = load_answers(ans_path)

        return evaluate_answers(exam_answers, correct_answers)
    except Exception as error:
        print(error)
        print(f"Failed at {i}")
        return False


if __name__ == "__main__":
    run_for_img(2, explain=True)
    # results = [run_for_img(i, debug=False) for i in range(1, 151)]
    # print(f"{sum(results)}/{len(results)}")
