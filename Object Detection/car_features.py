import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def color_hist(img, nbins=32, bins_range=(0,256)):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # compute the histogram of the HSV channels separately
    h_hist = np.histogram(hsv_img[:, :, 0], bins=nbins, range=bins_range)
    s_hist = np.histogram(hsv_img[:, :, 1], bins=nbins, range=bins_range)
    v_hist = np.histogram(hsv_img[:, :, 2], bins=nbins, range=bins_range)
    #Concatenat the histogram into a single feature vector
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    #Normalize the result
    norm_features = hist_features/ np.sum(hist_features)
    return norm_features

# Define a function to extract features from a list of images
# Have this function call color_hist()
def extract_features(imgs, hist_bins=32, hist_range=(0,256)):
    features = []
    #Iterate through the list of images
    for file in imgs:
        image = mpimg.imread(file)
        hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
        features.append(hist_features)
    return features

#Read in car and non-car images
images = glob.glob('dataset/*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

#See how your classifier performs under different binning scenarios
histbin = 32

car_features = extract_features(cars, hist_bins=histbin, hist_range=(0,256))
notcar_features = extract_features(notcars, hist_bins=histbin, hist_range=(0,256))

#Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
#Fit a per column scalar
X_scalar = StandardScaler().fit(X)
#Apply the scaler to X
scaled_X = X_scalar.transform(X)

#Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

#Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2,random_state = rand_state)


print('Dataset includes', len(cars), 'cars and', len(notcars), 'not-cars')
print('Using', histbin, 'histogram bins')
print('Feature vector length:', len(X_train[0]))

#use a linear SVC
svc = SVC(kernel='linear')
#Check training time for SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t,2), 'Seconds to train SVC....')
#Check the score of the SVC
print('Test Accuracy of SVC =', round(svc.score(X_test, y_test), 4))
#Check the prediction time for single sample
t = time.time()
n_predict = 10
print('My SVC predicts:', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')




