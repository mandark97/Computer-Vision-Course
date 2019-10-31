import os
import pdb
from collections import defaultdict
from pprint import pprint

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from imutils.object_detection import non_max_suppression

from exam_evaluator import ExamEvaluator
from image_processor import ImageProcessor
from utils import *

matplotlib.rcParams['image.cmap'] = 'gray'
# PATH = "data/template_test_grila.jpg"
PATH = "data/exemple_corecte/image_150.jpg"

img = cv2.imread(PATH)
print(img.shape)
