import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from time import perf_counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from IPython.display import Markdown, display

# Create a list with the filepaths for training and testing
dir_ = Path('C:\\Users\elliscll\Documents\Lego_Project\\archive\dataset')
file_paths = list(dir_.glob(r'**/*.png'))
file_paths = [str(x) for x in file_paths]
df = pd.DataFrame({'Filepath':file_paths})

def get_label(string):
    string  = ' '.join(string.split('/')[-1].replace('.png', '').split(' ')[1:-1])
    string = string.lower()
    return string

# Retrieve the label from the path of the pictures
df['Label'] = df['Filepath'].apply(lambda x: get_label(x))

# Load the paths of the validation set
validation = pd.read_csv('C:\\Users\elliscll\Documents\Lego_Project\\archive\\validation.txt', names = ['Filepath'])
validation['Filepath'] = validation['Filepath'].apply(lambda x: 'C:\\Users\elliscll\Documents\Lego_Project\\archive\dataset\\' + x)
#print(validation.head())
print(validation['Filepath'][0])

# The paths of the validation set is already in the DataFrame df
# Create a new column "validation_set" to indicate which
# paths of picture should be in the training and validation set
df['validation_set'] = df['Filepath'].isin(validation['Filepath'])
#print(df.head())
print(df['Filepath'][0])

# Create a DataFrame for the training set
# and for the test set. What is called "validation set"
# will be used as test set in the workbook
# Use only 30% of the data to speed up the model tests
train_df = df[df['validation_set'] == False].sample(frac = 0.3)
test_df = df[df['validation_set'] == True].sample(frac = 0.3)

a = 0
print(f'### Number of pictures in the train set: {train_df.shape[0]}')
print(f'### Number of pictures in the test set: {test_df.shape[0]}')

def create_gen():
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='training',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        subset='validation',
        rotation_range=30, # Uncomment to use data augmentation
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    
    return train_generator,test_generator,train_images,val_images,test_images

def get_model(model):
# Load the pretained model
    kwargs =    {'input_shape':(224, 224, 3),
                'include_top':False,
                'weights':'imagenet',
                'pooling':'avg'}
    
    pretrained_model = model(**kwargs)
    pretrained_model.trainable = False
    
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(46, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Dictionary with the models
models = {
    "MobileNet": {"model":tf.keras.applications.MobileNet, "perf":0},
}

# Create the generators
train_generator,test_generator,train_images,val_images,test_images=create_gen()
print('\n')

# Fit the models
for name, model in models.items():
    
    # Get the model
    m = get_model(model['model'])
    models[name]['model'] = m
    
    start = perf_counter()
    
    # Fit the model
    history = m.fit(train_images,validation_data=val_images,epochs=1,verbose=1)
    
    # Sav the duration and the val_accuracy
    duration = perf_counter() - start
    duration = round(duration,2)
    models[name]['perf'] = duration
    print(f"{name:20} trained in {duration} sec")
    
    val_acc = history.history['val_accuracy']
    models[name]['val_acc'] = [round(v,4) for v in val_acc]

# Create a DataFrame with the results
models_result = []

for name, v in models.items():
    models_result.append([ name, models[name]['val_acc'][-1], 
                          models[name]['perf']])
    
df_results = pd.DataFrame(models_result, 
                          columns = ['model','val_accuracy','Training time (sec)'])
df_results.sort_values(by='val_accuracy', ascending=False, inplace=True)
df_results.reset_index(inplace=True,drop=True)
df_results

# Use the whole data which is split into training and test datasets
# Create a DataFrame for the training set
# and for the test set. What is called "validation set"
# will be used as test set in the workbook
train_df = df[df['validation_set'] == False]
test_df = df[df['validation_set'] == True]

# Create the generators
train_generator,test_generator,train_images,val_images,test_images=create_gen()

# Get the model with the highest validation score
best_model = df_results.iloc[0]

# Create a new model
model = get_model( eval("tf.keras.applications."+ best_model[0]) )

# Train the model
history = model.fit(train_images,
                    validation_data=val_images,
                    epochs=1,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=3,
                            restore_best_weights=True)]
                    )
model.save('C:\\Users\elliscll\Documents\Lego_Project\src\models')

# Predict the label of the test_images
pred = model.predict(test_images)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]
# Get the accuracy on the test set
y_test = list(test_df.Label)
acc = accuracy_score(y_test,pred)

# Display the results
print(f'## Best Model: {best_model[0]} with {acc*100:.2f}% accuracy on the test set')

def generate_video_from_images(test_df):
    #print(len(test_images))
    import glob
    import cv2 
    import random
    imgs = []
    for row in test_df['Filepath']:
        print(row)
        img = cv2.imread(row)
        h,w,l = img.shape
        size = (w,h)
        imgs.append(img)

    print(size)
    print(np.array(imgs).shape)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('Legos1.mp4', fourcc, 20, (400,400))
        
    blank_frames_index = random.sample(range(len(imgs) + 2000), 2000)
    print(blank_frames_index[0:10])
    blank_image = np.zeros((400,400,3), np.uint8)
    j=0
    for i in range(len(imgs)+2000):
        if i in blank_frames_index:
            for x in range(20):
                out.write(blank_image)
    else:
        for x in range(20):
            out.write(imgs[j])
        j+=1
    out.release()

def create_label_csv():
    train_generator,test_generator,train_images,val_images,test_images=create_gen()
    print('\n')
    labels = (train_images.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    import csv
    with open('labels.csv', 'w') as f:
        for key in labels.keys():
            f.write("%s,%s\n"%(key,labels[key]))