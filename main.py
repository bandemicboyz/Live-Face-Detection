import time
import uuid
import os
import cv2
import tensorflow as tf
import albumentations as alb
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

IMAGES_PATH = os.path.join('data','images')
number_images = 30
vid = cv2.VideoCapture(0)

#collect images
for img in range(number_images):
    print(f'collecting image number {img}')
    ret,frame = vid.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname,frame)
    cv2.imshow('frame',frame)
    time.sleep(.5)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()

#limit gpu memory growth to avoid out of memory error
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

#load images into pipeline
images = tf.data.Dataset.list_files('data\\images\\*.jpg',shuffle = False )
print(images.as_numpy_iterator().next())

#define function to load images
def load_image(filepath):
    byte_img = tf.io.read_file(filepath)
    img = tf.io.decode_jpeg(byte_img)
    return img
#use map function to map through and load each image
images = images.map(load_image)

#MANUALLY SPLIT DATA IN TEST TRAIN AND VALIDATION

#move matching labels into train,test,val folder
for folder in ['TRAIN','TEST','VAL']:
    for file in os.listdir(os.path.join('data',folder,'images')):
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data','labels',filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath,new_filepath)


augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                         bbox_params=alb.BboxParams(
                         format='albumentations',
                         label_fields=['class_labels']))

#loop through each folder and every image in said folder
for partition in ['train','test','val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))
        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')

        # check to see if image has a corresponding label
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            #map coordinate to simple vector
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]

            # transform coordinates into albumentations format by dividing by width and height of image
            coords = list(np.divide(coords, [640,480,640,480]))

        try:

            # will create 60 augmented images for each image that was collected
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)

#load,resize, and scale images down. This makes it easier for the model to learn.
train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg',shuffle = False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x,(120,120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg',shuffle = False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x,(120,120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg',shuffle = False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x,(120,120)))
val_images = val_images.map(lambda x: x/255)

#define function to load labels
def load_label(label_path):
    with open(label_path.numpy(),'r',encoding= 'utf-8') as f:
        label = json.load(f)
    return [label['class'],label['bbox']]

#load labels
train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16]))

# print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))

#create a dataset that contains images and the corresponding labels to be fed into the model for training, testing, and validation
train = tf.data.Dataset.zip((train_images,train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images,test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images,val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val.prefetch(4)


def build_model():
    input_layer = Input(shape = (120,120,3))
    vgg = VGG16(include_top = False)(input_layer)

    #classification model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048,activation = 'relu')(f1)
    class2 = Dense(1,activation = 'sigmoid')(class1)

    #bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048,activation = 'relu')(f2)
    regress2 = Dense(4,activation = 'sigmoid')(regress1)
    facetracker = Model(inputs = input_layer, outputs = [class2,regress2])
    return facetracker

#initialize facetracker model
facetracker = build_model()
batches_per_epoch = len(train)

#as the model learns the learning rate will decay this assits in optimization and generalization
lr_decay = (1./0.75-1)/batches_per_epoch

#added legacy to optimizer to avoid error
opt = tf.keras.optimizers.legacy.Adam(learning_rate = 0.0001, decay = lr_decay)

#create localization loss
def localization_loss(y_true,yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))

    h_true = y_true[:,3] - y_true[:,1]

    w_true = y_true[:,2] - y_true[:,0]

    h_pred = yhat[:,3] - y_true[:,1]

    w_pred = yhat[:,2] - y_true[:,0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size

#define class loss and regression loss methods
classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

#build machine learning pipeline
class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)

        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

logdir = 'logs'
checkpoint_path = logdir
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#create a checkpoint after each epoch to act as a failsafe
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= os.path.join(checkpoint_path,'checkpoints'),
    verbose=1,
    save_weights_only=True)

# #train model
hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback,cp_callback],)

#visualize model performance
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()

# #save model
facetracker.save('facetracker.h5')
facetracker1 = load_model('facetracker.h5')

#real time detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = frame[50:500, 50:500, :]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120, 120))

    yhat = facetracker1.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]

    if yhat[0] > 0.5:
        # Controls the main rectangle
        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255, 0, 0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame,
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [0, -30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                   [80, 0])),
                      (255, 0, 0), -1)

        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                [0, -5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('EyeTrack', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()