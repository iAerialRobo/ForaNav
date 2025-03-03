import cv2
import numpy as np
import time
from skimage.feature import hog
from joblib import dump, load
from img_prep import img_prep

start_time = time.time()

model = load('YOUR_FILE_PATH')

n = 0.08  
m = 1 / n - 0.01

imorg = cv2.imread('YOUR_FILE_PATH')

imorg_RGB = cv2.cvtColor(imorg, cv2.COLOR_BGR2RGB)
img, img_res, adjusted_img = img_prep(imorg_RGB, n)

img_ref = cv2.imread('YOUR_FILE_PATH')
img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
HSV_ref = cv2.cvtColor(img_ref, cv2.COLOR_RGB2HSV)

window_size = (24, 24)
step_size = 3

distance_threshold = 15
hsv_threshold = 0.5
score_threshold = 25 * 10 ** (-4)

c = 6
b = 3
O = 2
cellSize = (c, c)
blockSize = (b, b)
numBins = 9
blockOverlap = (O, O)

c2 = 12
b2 = 5
O2 = 2
cellSize2 = (c2, c2)
blockSize2 = (b2, b2)
blockOverlap2 = (O2, O2)

img_height, img_width = img.shape

result_list = []
similarity_list = []
position_list = []
vis_list = []

for row in range(0, img_height - window_size[0] + 1, step_size):
    for col in range(0, img_width - window_size[1] + 1, step_size):

        window = img[row:row + window_size[0], col:col + window_size[1]]
        window_hog = hog(window, orientations=numBins, pixels_per_cell=cellSize, cells_per_block=blockSize,
                         block_norm='L2-Hys')

        score = model.decision_function([window_hog])[0]

        position_list.append((col, row))
        similarity_list.append(score)
        vis_list.append(None)

remove_idx = np.array(similarity_list) < score_threshold

similarity_list = np.delete(similarity_list, np.where(remove_idx))

position_list = np.delete(position_list, np.where(remove_idx), axis=0)

vis_list = np.delete(vis_list, np.where(remove_idx), axis=0)

final_result_list = []

sorted_idx = np.argsort(similarity_list)[::-1]
similarity_list = similarity_list[sorted_idx]
position_list = position_list[sorted_idx]
vis_list = vis_list[sorted_idx]

num_bins = 256
reference_H_hist = cv2.calcHist([HSV_ref], [0], None, [num_bins], [0, 256]) / np.prod(HSV_ref.shape[:2])
reference_S_hist = cv2.calcHist([HSV_ref], [1], None, [num_bins], [0, 256]) / np.prod(HSV_ref.shape[:2])

for i in range(position_list.shape[0]):
    current_position = position_list[i]

    distance_flag = True
    for j in range(len(final_result_list)):
        existing_position = final_result_list[j][:2]
        distance = np.sqrt((current_position[0] - existing_position[0]) ** 2 +
                           (current_position[1] - existing_position[1]) ** 2)
        if distance < distance_threshold:
            distance_flag = False
            break

    if distance_flag:
        window = imorg[int(current_position[1] * m):int((current_position[1] + window_size[0]) * m),
                 int(current_position[0] * m):int((current_position[0] + window_size[1]) * m), :]
        window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
        HSV_current = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)
        current_H_hist = cv2.calcHist([HSV_current], [0], None, [num_bins], [0, 256]) / np.prod(HSV_current.shape[:2])
        current_S_hist = cv2.calcHist([HSV_current], [1], None, [num_bins], [0, 256]) / np.prod(HSV_current.shape[:2])
        H_correlations = np.sum((reference_H_hist - np.mean(reference_H_hist)) *
                                (current_H_hist - np.mean(current_H_hist))) / \
                         (np.std(reference_H_hist) * np.std(current_H_hist)) / 255
        S_correlations = np.sum((reference_S_hist - np.mean(reference_S_hist)) *
                                (current_S_hist - np.mean(current_S_hist))) / \
                         (np.std(reference_S_hist) * np.std(current_S_hist)) / 255
        H_S_average = (H_correlations + S_correlations) / 2
        if H_S_average > hsv_threshold:
            window2 = imorg[int(current_position[1] * m):int((current_position[1] + window_size[0]) * m),
                      int(current_position[0] * m):int((current_position[0] + window_size[1]) * m), :]
            test_img, _, _ = img_prep(window2, 1)
            test_feature = hog(test_img, orientations=numBins, pixels_per_cell=cellSize2, cells_per_block=blockSize2,
                               block_norm='L2-Hys')
            X = numBins
            L = test_feature.size // X
            L = int(L)
            test_feature = test_feature.reshape(X, L)
            row_sums = np.sum(test_feature, axis=1)
            variance = np.var(row_sums)

            if variance > 120:
                final_result_list.append(np.append(current_position, similarity_list[i]))

end_time = time.time()

run_time = end_time - start_time

for final_result in final_result_list:
    x = final_result[0]
    y = final_result[1]
    score = final_result[2]
    cv2.rectangle(imorg, (int(x * m), int(y * m)),
                  (int((x + window_size[1]) * m), int((y + window_size[0]) * m)), (0, 0, 255), 10)
    cv2.putText(imorg, f'Score: {score * 10 ** 4:.2f}', (int(x * m) + 15, int(y * m) + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

window_name = 'Final Detection'
window_width = 1344
window_height = 756
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)

resized_image = cv2.resize(imorg, (window_width, window_height))

cv2.imshow('Final Detection', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
