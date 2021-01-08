from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import os

#entering the image path
imagePaths = ['results/test3.jpg']

# load our object detector and label binarizer from disk
print("[INFO] loading object detector...")
model = load_model('weights/bboxweights.h5')
lb = pickle.loads(open('weights/lb.pickle', "rb").read())

# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:
	# load the input image (in Keras format) from disk and preprocess
	# it, scaling the pixel intensities to the range [0, 1]
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image) / 255.0
	image = np.expand_dims(image, axis=0)

	# predict the bounding box of the object along with the class
	# label
	(boxPreds, labelPreds) = model.predict(image)
	(startX, startY, endX, endY) = boxPreds[0]

	# determine the class label with the largest predicted
	# probability
	i = np.argmax(labelPreds, axis=1)
	label = lb.classes_[i][0]

	# load the input image (in OpenCV format), resize it such that it
	# fits on our screen, and grab its dimensions
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(600,600))
	(h, w) = image.shape[:2]

	# scale the predicted bounding box coordinates based on the image
	# dimensions
	startX = int(startX * w)
	startY = int(startY * h)
	endX = int(endX * w)
	endY = int(endY * h)

	# draw the predicted bounding box and class label on the image
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 255, 0), 2)
	cv2.rectangle(image, (startX, startY), (endX, endY),
		(0, 255, 0), 2)

	# show the output image
	cv2.imwrite("results/Output3.jpg", image)
	cv2.waitKey(0)