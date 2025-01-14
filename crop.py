import os
import cv2
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

IMAGE = '8-6P1-8-7r-2k5-R7-2p2K2-8.png'
INPUT_SIZE = 512

# returns the angle between two vectors ab and ac
def get_angle(ab, ac):
    x = np.clip((ab @ ac) / np.linalg.norm(ab) / np.linalg.norm(ac), -1, 1)
    return np.degrees(np.arccos(x)) if not np.isnan(x) else 0

# check if the set of points form a square
def check_square(points):
    if len(points) != 4:
        return 4 * 90 ** 2, 0
    a, b, c, d = np.squeeze(points)
    bcd = get_angle(b - c, d - c)
    cda = get_angle(c - d, a - d)
    dab = get_angle(d - a, b - a)
    abc = get_angle(a - b, c - b)

    return np.sum((np.array([bcd, cda, dab, abc]) - 90) ** 2), np.mean(np.abs([a - b, b - c, c - d, d - a]))

# count how many squares are inside the square of index i
def child_count(i, hierarchy, is_square):
    j = hierarchy[0, i, 2]
    if j < 0:
        return 0

    total = 0
    while hierarchy[0, j, 0] > 0:
        j = hierarchy[0, j, 0]
    while j > 0:
        if is_square[j]:
            total += 1
        j = hierarchy[0, j, 1]
    return total

def get_contours(image, blur_radius):
    edge_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (blur_radius, blur_radius), 2)
    edge_image = cv2.Canny(edge_image, 20, 200)
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)),
        iterations=1
    )

    # find the contours
    contours, hierarchy = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Convert contours to a list
    contours = list(contours)
    # aproximates every contour group to a simpler polygon
    for i in range(len(contours)):
        contours[i] = cv2.approxPolyDP(contours[i], 0.04 * cv2.arcLength(contours[i], True), True)

    return contours, hierarchy

def get_candidate_boards(contours, hierarchy, max_error, min_side, min_child):
    square_info = list(map(lambda contour: check_square(contour), contours))
    is_square = list(map(lambda info: info[0] < max_error and info[1] > min_side, square_info))

    squares = []
    for i, square_flag, (error, side), contour in zip(range(len(contours)), is_square, square_info, contours):
        if square_flag:
            cnt = child_count(i, hierarchy, is_square)
            if cnt > min_child:
                squares.append(contour)

    return squares

def preprocess(image):
    image2 = np.copy(image)
    all_boards = []

    for blur in (3, 5, 7, 9, 11):
        contours, hierarchy = get_contours(image, blur)
        boards = get_candidate_boards(contours, hierarchy, 4 * 10 ** 2, 10, 10)
        all_boards.extend(boards)
        for board in boards:
            error, side = check_square(board)
    all_boards = list(map(lambda board: (board, check_square(board)[1]), all_boards))
    all_boards = sorted(all_boards, key=lambda board_side: board_side[1], reverse=True)
    best_board = (None, 0)
    for board, side in all_boards:
      if side >= best_board[1] * 0.9:
        best_board = (board, side)

    return best_board[0]
# rotate and crop the image, given the points of the new border
def crop_image(image, points, rotate=False):
    points = np.squeeze(points)
    if rotate:
      low, high = points[np.lexsort((points[:, 1], points[:, 0]))[:2]]
      angle = -get_angle(np.array([0, 1]), high - low)
      if angle < -90:
          angle, low, high = angle + 180, high, low

      rot = cv2.getRotationMatrix2D((int(low[0]), int(low[1])), angle, 1.0)
      h, w = image.shape[:2]
      image = cv2.warpAffine(image, rot, (w, h))
      points = np.concatenate((points, np.ones(points.shape[0]).reshape((-1, 1))), axis=1)
      points = (rot @ points.T).T.astype(np.int32)

    image = image[points[:, 1].min():points[:, 1].max(), points[:, 0].min():points[:, 0].max()]
    return image

# read and resize the image
image = imageio.imread(IMAGE)
board = preprocess(image)
cropped = crop_image(image, board)
cropped = cv2.resize(cropped, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
plt.figure(figsize=(15, 15))
plt.imshow(cropped)
