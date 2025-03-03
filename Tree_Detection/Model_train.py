import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from skimage.feature import hog
from joblib import dump, load
from sklearn.metrics import precision_score, recall_score
from img_prep import img_prep

image_folder = 'YOUR_FILE_PATH'
positive_test_folder = 'YOUR_FILE_PATH'
negative_test_folder = 'YOUR_FILE_PATH'
image_extension = '.jpg'

c = 6
b = 3
O = 2
cellSize = (c, c)
blockSize = (b, b)
numBins = 9
blockOverlap = (O, O)

n = 0.1
feature_matrix = []

score_threshold = 0

image_files = glob.glob(image_folder + '*' + image_extension)
for image_file in image_files:
    imorg = cv2.imread(image_file)
    v_channel, img_res, adjusted_img = img_prep(imorg, n)
    feature = hog(v_channel, orientations=numBins, pixels_per_cell=cellSize, cells_per_block=blockSize,
                  block_norm='L2-Hys')
    feature_matrix.append(feature)

kernel_scale = 0.1
nu = 0.00020
model = OneClassSVM(kernel='rbf', gamma=kernel_scale, nu=nu)
model.fit(feature_matrix)

positive_test_files = glob.glob(positive_test_folder + '*' + image_extension)
negative_test_files = glob.glob(negative_test_folder + '*' + image_extension)

num_positive_files = len(positive_test_files)
num_negative_files = len(negative_test_files)

predictions = np.zeros(num_positive_files + num_negative_files)
scores = np.zeros(num_positive_files + num_negative_files)

for i, test_file in enumerate(positive_test_files):
    test_img = cv2.imread(test_file)
    v_channel, img_res, adjusted_img = img_prep(test_img, n)
    test_feature = hog(v_channel, orientations=numBins, pixels_per_cell=cellSize, cells_per_block=blockSize,
                       block_norm='L2-Hys')

    label = model.predict([test_feature])[0]
    score = model.decision_function([test_feature])[0]

    if score <= score_threshold:
        label = -1

    predictions[i] = label
    scores[i] = score

for i, test_file in enumerate(negative_test_files):
    test_img = cv2.imread(test_file)
    v_channel, img_res, adjusted_img = img_prep(test_img, n)
    test_feature = hog(v_channel, orientations=numBins, pixels_per_cell=cellSize, cells_per_block=blockSize,
                       block_norm='L2-Hys')

    label = model.predict([test_feature])[0]
    score = model.decision_function([test_feature])[0]

    if score <= score_threshold:
        label = -1

    predictions[num_positive_files + i] = label
    scores[num_positive_files + i] = score

ground_truth = np.concatenate((np.ones(num_positive_files), -np.ones(num_negative_files)))
accuracy = np.sum(predictions == ground_truth) / len(ground_truth)

precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)

print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}%'.format(precision * 100))
print('Recall: {:.2f}%'.format(recall * 100))
print('kernel_scale: {:.6f}'.format(kernel_scale))
print('nu: {:.6f}'.format(nu))

save_folder = ''YOUR_FILE_PATH''
save_path = os.path.join(save_folder, 'test.joblib')

dump(model, save_path)

plt.scatter(range(len(scores)), scores)
plt.xlabel('Num')
plt.ylabel('Value')
plt.title('1000scatter')
plt.show()
