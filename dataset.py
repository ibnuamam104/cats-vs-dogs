import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pickle

DATA_DIR = "./cats_vs_dogs/"
categories = ["dog", "cat"]

for category in categories:
    path = os.path.join(DATA_DIR, category)
    for image in os.listdir(path):
        image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        # plt.imshow(image_array, cmap="gray")
        # plt.show()
        break
    break
# print(image_array.shape)

image_size = 75
# new_array = cv2.resize(image_array, (image_size, image_size))
# plt.imshow(new_array, cmap='gray')
# plt.show()

training_data = []
def create_training_data():
    for category in categories:
        path = os.path.join(DATA_DIR, category)
        class_category = categories.index(category)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(image_array, (image_size, image_size))
                training_data.append([resized_image, class_category])
                print("Done")
            except Exception as e:
                pass
create_training_data()
print(len(training_data))
random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, image_size, image_size, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

