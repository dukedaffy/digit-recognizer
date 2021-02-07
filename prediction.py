from keras.models import load_model
import matplotlib.pyplot  as plt
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

#load the saved model
model = load_model('D:\\duke\\project\\sudoku\\model1.h5')

model.compile(optimizer=Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

#loading images from your system
path='D:\\duke\\project\\sudoku\\predict\\3.jpg' 
img1 = cv2.imread(path)

#converting the image to grayscale
img= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#converting to binary image
et, img1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 

#resizing the image
img = cv2.resize(img1,(28,28))
img = np.reshape(img,[1,28,28,1])

#predicting the model
pre = model.predict(img)
prediction = np.array([np.argmax(x) for x in pre])
print("predicted value = ",prediction[0])

#plotting input image and processed image
plt.subplot(1,2,1)
plt.imshow(img1)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(img[0,:,:,:], cmap="gray")
plt.axis('off')
plt.show()


