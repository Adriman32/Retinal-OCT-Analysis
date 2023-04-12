import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from Models import *
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import time

def  load_data(path):
    load_start = time.time()
    img_list = []
    label_list = []
    for curr_class in os.listdir(path):
        for patient_num in os.listdir(path+curr_class+'/'):
            for filename in os.listdir(path+curr_class+'/'+patient_num+'/'):
                parsed_name = filename.split('_')
                if parsed_name[0] == 'OD' or parsed_name[0] == 'OS':
                    for os_od_filename in os.listdir(path+curr_class+'/'+patient_num+'/'+filename+'/'):
                        parsed_name = os_od_filename.split('_')
                        img_label = parsed_name[1].split('.')[0].lower()
                        img = Image.open(path+curr_class+'/'+patient_num+'/'+filename+'/'+os_od_filename).resize((224,224))
                        img_aug = aug_image(img,preprocess=False)
                        img_list.append(img_aug)
                        label_list.append(img_label.lower())

                        img_proc = aug_image(img,preprocess=True)
                        img_list.append(img_proc)
                        label_list.append(img_label.lower())

                        
                else:
                    img_label = parsed_name[1].split('.')[0]
                    img = Image.open(path+str(curr_class)+'/'+str(patient_num)+'/'+filename).resize((224,224))
                    img_arr = np.asarray(img)
                    img_arr = img_arr.astype('float32') / 255.0
                    img_list.append(img_arr)
                    label_list.append(img_label.lower())

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    label_list = label_encoder.fit_transform(label_list)
    label_list = label_list.reshape(len(label_list), 1)
    label_list = onehot_encoder.fit_transform(label_list)
    print('Data loaded in %s minutes'%(round((time.time()-load_start)/60,4)))
    return np.asarray(img_list),label_list

def aug_image(img,preprocess=True):
    img_arr = np.asarray(img)
    
    if preprocess:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                shear_range=5,
                                                                brightness_range=[0.8,1.2],
                                                                zoom_range=0.2,
                                                                horizontal_flip=True)
        it = datagen.flow(np.expand_dims(img_arr,0), batch_size=1)
        img_arr = it.next()[0].astype('float32')

    img_norm = img_arr.astype('float32') / 255.0

    img_zero_mean = img_norm - np.mean(img_norm)
    img_out = img_zero_mean / np.std(img_zero_mean)

    return img_out

if __name__ == '__main__':
    start_time = time.time()

    path = "/path/to/dataset/"

    img_array_list, img_labels = load_data(path)
    x_train, x_test, y_train, y_test = train_test_split(img_array_list, img_labels,random_state=42)

    my_model = FPN_VGG16(num_classes = len(y_train[0]),verbose=True)
    my_model.compile()

    history = my_model.train(x_train,y_train,validation_data=(x_test,y_test),epochs=200,batch_size=16,verbose=1)
    print("--------------------------------------------------------------")
    my_model.save()
    print('Total Run Time: %s minutes'%(round((time.time()-start_time)/60,4)))
