import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

import os
import scipy
import h5py
import numpy             as np
from random                               import shuffle
from shutil                               import copyfile
from tensorflow.keras.models              import load_model
from sklearn.metrics                      import confusion_matrix, classification_report
from tensorflow.keras.callbacks           import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

SOURCEDIR = 'train/'
DATADIR = 'data/'
MODELDIR = 'checkpoint.h5'

def modelVGG16():

  model = Sequential()
  # BLOCK 1
  model.add(Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', input_shape=(224, 224, 3)))
  model.add(Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  # BLOCK 2
  model.add(Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  # BLOCK 3
  model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  # BLOCK 4
  model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  # BLOCK 5
  model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform'))
  model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
  
  # BLOCK 6
  model.add(Flatten())
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(4096, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # Dense 1000 causes error

  return model



def processData() -> None:

    ### SEPARATE DOGS AND CATS
    
    print('Separating dogs and cats...')
    cats = []
    dogs = []
    
    for img in os.listdir(SOURCEDIR):
        # elif to prevent unknown files
        if   img[:3] == 'cat' : cats.append(img)
        elif img[:3] == 'dog' : dogs.append(img)
    
    
    ### SPLIT USAGE RATIO
    
    print('Splitting for training, validating and testing...')
    def datasplit(data: list, pertrain: int, pervalid: int, seed: int):
    
        shuffle(data)
        train = int(len(data) * pertrain)
        valid = int(len(data) * pervalid + train)
    
        return list(data[:train]),           \
               list(data[train:valid]),      \
               list(data[valid:])
    
    traincats, valcats, testcats = datasplit(cats, .6, .2, 10)
    traindogs, valdogs, testdogs = datasplit(dogs, .6, .2, 10) 
    
    
    ### SAVE TO FOLDER
    
    print('Saving to folder... (less than 1 minute)')
    def saveto(SUBDIR: str, data: list) -> None:
        copy = lambda img: copyfile(SOURCEDIR + img, DATADIR + SUBDIR + img)
        list(map(copy, data))
        return None
    
    saveto('train/cats/', traincats)
    saveto('train/dogs/', traindogs)
    saveto('val/cats/', valcats)
    saveto('val/dogs/', valdogs)
    saveto('test/cats/', testcats)
    saveto('test/dogs/', testdogs)

    print('Processed Data!')



def train(model):

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_it = train_datagen.flow_from_directory(DATADIR + 'train/', class_mode='binary', batch_size=20, target_size=(224, 224))
    val_it = val_datagen.flow_from_directory(DATADIR + 'val/'    , class_mode='binary', batch_size=20, target_size=(224, 224))



    def optimizer_init_fn(): 
        learning_rate = 0.001
        momentum = 0.9
        return tf.keras.optimizers.SGD(learning_rate, momentum)

    model.compile(optimizer_init_fn(), 'binary_crossentropy', ['accuracy'])
    filepath = MODELDIR
    callback = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    return None



def test():

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_it = test_datagen.flow_from_directory(DATADIR + 'test/'  , class_mode='binary', batch_size=20, target_size=(224, 224))

    model = load_model(MODELDIR)
    print(1)
    results = model.evaluate(test_it, verbose=0)
    
    print("    Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))
    
    return None



def customizedtest():
    
    model = load_model(MODELDIR)

    #dictionary to label all traffic signs class.

    classes = { 
        0:'its a cat',
        1:'its a dog',
    }

    #initialise GUI
    top = Tk()
    top.geometry('800x600')
    top.title('CatsVSDogs Classification')
    top.configure(background = '#CDCDCD')
    label = Label(top,background = '#CDCDCD', font = ('arial',15,'bold'))
    sign_image = Label(top)

    def classify(file_path):
        global label_packed
        image = Image.open(file_path)
        image = image.resize((224,224))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        image = image / 255
        pred = model.predict(image)
        if pred[0][0] > .5: 
            pred = 1
        else: 
            pred = 0
        sign = classes[pred]
        label.configure(foreground='#011638', text=sign) 

    def show_classify_button(file_path):
        classify_b = Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
        classify_b.configure(background='#364156', foreground='white',
    font=('arial',10,'bold'))
        classify_b.place(relx=0.79,rely=0.46)

    def upload_image():
        try:
            file_path = filedialog.askopenfilename()
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width() / 2.25),
        (top.winfo_height() / 2.25)))
            im = ImageTk.PhotoImage(uploaded)
            sign_image.configure(image=im)
            sign_image.image = im
            label.configure(text='')
            show_classify_button(file_path)
        except:
            pass

    upload = Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
    upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    upload.pack(side=BOTTOM,pady=50)
    sign_image.pack(side=BOTTOM,expand=True)
    label.pack(side=BOTTOM,expand=True)
    heading = Label(top, text="CatsVSDogs Classification",pady=20, font=('arial',20,'bold'))
    heading.configure(background='#CDCDCD',foreground='#364156')
    heading.pack()
    top.mainloop()

while True:
    print("1. Process Data")
    print("2. Train (New)")
    print("3. Retrain")
    print("4. Test (test all in 'test' dataset)")
    print("5. Test (test using customized image)")
    print("6. Exit")
    t = int(input("Input: "))

    if t == 1:
        print("PROCESSING..")
        processData()
    elif t == 2:
        print("TRAINING...")
        train(modelVGG16())
    elif t == 3:
        print("RETRAINING...")
        train(load_model(MODELDIR))
    elif t == 4:
        print("TESTING...")
        test()
    elif t == 5:
        print("TESTING...")
        customizedtest()
    elif t == 6:
        print("BYE BYE")
        break
    else:
        print("INVALID INPUT")