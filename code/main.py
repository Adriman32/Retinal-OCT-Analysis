import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from Models import VGG16
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
                        img_arr = np.asarray(img)
                        img_arr = img_arr.astype('float32') / 255.0
                        img_list.append(img_arr)
                        label_list.append(img_label)
                else:
                    img_label = parsed_name[1].split('.')[0]
                    img = Image.open(path+str(curr_class)+'/'+str(patient_num)+'/'+filename).resize((224,224))
                    img_arr = np.asarray(img)
                    img_arr = img_arr.astype('float32') / 255.0
                    img_list.append(img_arr)
                    label_list.append(img_label)

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    label_list = label_encoder.fit_transform(label_list)
    label_list = label_list.reshape(len(label_list), 1)
    label_list = onehot_encoder.fit_transform(label_list)
    print('Data loaded in %s minutes'%(round((time.time()-load_start)/60,4)))
    return img_list,label_list


if __name__ == '__main__': 
    start_time = time.time()
    path = 'path/to/dataset/'
    img_array_list, img_labels = load_data(path)
    x_train, x_test, y_train, y_test = train_test_split(img_array_list, img_labels,random_state=42)

    x_train = tf.stack(x_train)
    x_test = tf.stack(x_test)

    my_model = VGG16(num_classes = len(y_train[0]))
    my_model.compile()
    history = my_model.train(x_train,y_train)
    model_loss, model_accuracy = my_model.evaluate(x_test, y_test)
    print("--------------------------------------------------------------")
    print('Loss: %s\tAccuracy: %s'%(round(model_loss,4),round(model_accuracy,4)))
    print("--------------------------------------------------------------")
    my_model.save()
    print('Total Run Time: %s minutes'%(round((time.time()-start_time)/60,4)))
