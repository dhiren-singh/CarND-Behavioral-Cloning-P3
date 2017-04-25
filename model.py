import csv
import numpy as np
from keras.models import (
        Sequential,
        Model,
)
from keras.layers import (
        Flatten,
        Dense,
        Lambda,
        Activation,
        Dropout,
)
from keras.layers.convolutional import (
        Convolution2D,
        Cropping2D,
)
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split


def gen_model():
        # Split data into 80-20 split.
        train_samples, validation_samples = train_test_split(load_samples(), test_size=0.2)
        print("Length of train samples ", len(train_samples))
        print("Length of validation samples ", len(validation_samples))

        train_generator = preprocess_and_load(train_samples, batch_size=16)
        validation_generator = preprocess_and_load(validation_samples, batch_size=8)
        train_and_save(
                train_samples=train_samples,
                train_generator=train_generator,
                validation_samples=validation_samples,
                validation_generator=validation_generator
        )


def load_samples():
        lines = []
        with open('data/driving_log.csv') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                        lines.append(line)
        return lines


def preprocess_and_load(samples, batch_size=32):
    num_samples = len(samples)
    # Loop forever so the generator never terminates
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size//4]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                image = Image.open(name)
                center_image = np.asarray(image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # Flip the image and angle becomes negative. Add this to image array.
                image_flipped = np.fliplr(image)
                center_image_flipped = np.asarray(image_flipped)
                center_angle_flipped = -center_angle
                images.append(center_image_flipped)
                angles.append(center_angle_flipped)

                # Align the image to center by +0.5
                name = './data/IMG/' + batch_sample[1].split('/')[-1]
                image = Image.open(name)
                left_image = np.asarray(image)
                left_angle = float(batch_sample[3]) + 0.5
                images.append(left_image)
                angles.append(left_angle)

                # Align the image to center by -0.5
                name = './data/IMG/' + batch_sample[2].split('/')[-1]
                image = Image.open(name)
                right_image = np.asarray(image)
                right_angle = float(batch_sample[3]) - 0.5
                images.append(right_image)
                angles.append(right_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            # shuffle & return generator we would use this as input.
            yield sklearn.utils.shuffle(X_train, y_train)


def train_and_save(
        train_samples,
        train_generator,
        validation_samples,
        validation_generator
):
        model = training_model()
        model.compile(loss='mse', optimizer='adam')
        # fit using generator object.
        model.fit_generator(
                train_generator,
                samples_per_epoch=len(train_samples * 4),
                validation_data=validation_generator,
                nb_val_samples=len(validation_samples),
                nb_epoch=50
        )
        model.save('model.h5')


def training_model():
        model = Sequential()
        # Normalization the data for better results
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
        # Crop the image to remove unwanted features
        model.add(Cropping2D(cropping=((60, 20), (0, 0))))
        # Convolutional 24@79x159
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Convolutional 36@39x79
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Convolutional 48@19x39
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Convolutional 64@9x19
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Convolutional 64@5x10
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Connected network.
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))
        return model


gen_model()
