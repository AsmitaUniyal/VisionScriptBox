import os
import numpy as np
import cv2
import time
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import load_model,Model
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.applications.imagenet_utils import preprocess_input


# image specification
img_rows= 224
img_cols= 224
img_channels=3
num_classes = 3
frame=15
#%%
#model = load_model('custom_model62.h5')
weight_file='custom_model_weights_iter_62.h5'
model1= load_model('custom_model62.h5')

model1.load_weights(weight_file)
model1.layers.pop()
         
path = os.getcwd()
dataset_path = 'used_data/' 
dataset_list = os.listdir(dataset_path)
labels_name= {'Drinking' : 0, 'Reading': 1, 'Talking': 2}
labels_list= []
num_classes = 3


new_input = model1.input
last_layer = model1.get_layer('fc1').output
model2 = Model(new_input,last_layer)

def extract_features(list1,ds_name):  
    feature_path ='Saved_Features/'  
#    img = cv2.imread('frame_0.jpg')
#    img = cv2.resize(img, (224, 224))
    #x = image.img_to_array(img)
    x = np.array(list1)
    #x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #model2.summary()
    features = (model2.predict(x))
    #features_train = model.predict(img_array)
    
    #print("features_train",features,len(features),features.shape)
    feature_path = os.path.join(feature_path+ds_name+str(time.time()) + ".npy")
    print("feature_path",feature_path,len(features))
    np.save(feature_path, features)
    #np.save(open('features_train.npy', 'w'), features_train)



for dataset in dataset_list:
    data_list = os.path.join(dataset_path, dataset)
    img_list = os.listdir(data_list)   
    #print("Loading images of: ", data_list)
    label = labels_name[dataset]
    for db_list in img_list:
        data2_list = os.path.join(data_list, db_list)
        db2_list = os.listdir(data2_list)
        print("Loading images of: ", data2_list)
        count=0
        image_list = []
        #img_path= os.path.join(data_list,img)
        #image = cv2.imread(img_path,1)
        for img in db2_list:
            image = cv2.imread(data2_list + '/'+ img)
            image = cv2.resize(image,(img_rows, img_cols))
            print("Image loaded: ", img)
            #cv2.imshow('Test Picture', image)
            #cv2.waitKey(1)
            if(count<frame):
                image_list.append(image)
                labels_list.append(label)
                count=count+1
            elif(count==frame):
                count =0
                extract_features(image_list,dataset)   
                image_list = []
        cv2.destroyAllWindows()

