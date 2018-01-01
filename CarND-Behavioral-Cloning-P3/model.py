import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import MaxPooling2D, merge, Dense, Activation, Conv2D, Lambda, AveragePooling2D, Flatten, Dropout, Input, BatchNormalization, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

csv_dir = "C:\\Users\\DavidCrompton\\Documents\\data\\driving_log.csv"
img_dir = "C:\\Users\\DavidCrompton\\Documents\\data\\IMG\\"

samples = []

with open(csv_dir) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples=samples[1:] 
        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)      
        
def data_generator(samples, batch_size=32, aug=False): 
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            center_images = []
            left_images = []
            right_images = []
            center_angles = []
            left_angles = []
            right_angles = []
            for batch_sample in batch_samples:
                center_image = cv2.imread(img_dir + batch_sample[0].split("/")[-1])[...,::-1]  # [...,::-1] Converts BGR to RGB
                left_image = cv2.imread(img_dir + batch_sample[1].split("/")[-1])[...,::-1]
                right_image = cv2.imread(img_dir + batch_sample[2].split("/")[-1])[...,::-1]
                center_angle = float(batch_sample[3])
                
                center_images.append(center_image)
                left_images.append(left_image)
                right_images.append(right_image)
                
                correction = 0.3 # this is a parameter to tune
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                
                center_angles.append(center_angle)
                left_angles.append(steering_left)
                right_angles.append(steering_right)

            x_train = np.array(center_images)
            y_train = np.array(center_angles)
            yield x_train, y_train
            
            x_train = np.array(left_images)
            y_train = np.array(left_angles)
            yield x_train, y_train
            
            x_train = np.array(right_images)
            y_train = np.array(right_angles)
            yield x_train, y_train
            
            # For the next batch yield augmented data
            if aug == True:
                images = [np.fliplr(image) for image in center_images]
                angles = [-angle for angle in center_angles]  # Reverse steering angle for flipped images
                x_train = np.array(images)
                y_train = np.array(angles)
                yield x_train, y_train
				
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='same', activation="relu"))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='same', activation="relu"))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='same', activation="relu"))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
model.add(Conv2D(64, 3, 3, subsample=(2, 2), border_mode='same', activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1164, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1))

opt = optimizers.Adam(lr=0.0001)

# Load weights for model if they already exist
try:
    model.load_weights('model.h5')
except Exception:
    pass
model.compile(loss="mse", optimizer=opt)

# Set the model to save the best weights, and stop if the weights do not imporve after 5 epochs
checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, 
                               save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)


model.fit_generator(
    data_generator(train_samples, aug=False),
    samples_per_epoch = len(train_samples)*3,  # Times three as we use the left, center and right camera
    validation_data=data_generator(validation_samples),
    nb_val_samples=len(validation_samples),
    nb_epoch=2000,
    callbacks=[checkpointer, earlystop] 
)