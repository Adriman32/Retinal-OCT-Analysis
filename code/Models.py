import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout, Concatenate, Input, Add, UpSampling2D, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow import keras
import os.path
import time


class VGG16:
	def __init__(self,num_classes, verbose=False):
		self.verbose = verbose
		self.history = None
		self.feature_blocks = []
		self.model = self.create_model(num_classes)

		self.tb_cb = self.setup_cbs()

	def create_model(self, num_classes, verbose = False):
		if self.verbose or verbose:
			print("Creating VGG16 Model...")
		input_layer = Input(shape=(224,224,3))
		conv_layer_1 = Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding='same')(input_layer)
		act_layer_1 = Activation(activation='relu')(conv_layer_1)
		conv_layer_2 = Conv2D(filters=64, kernel_size=(3,3),padding='same')(act_layer_1)
		act_layer_2 = Activation(activation='relu')(conv_layer_2)
		max_pool_layer_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_2)

		conv_layer_3 = Conv2D(filters=128,kernel_size=(3,3),padding='same')(max_pool_layer_1)
		act_layer_3 = Activation(activation='relu')(conv_layer_3)
		conv_layer_4 = Conv2D(filters=128,kernel_size=(3,3),padding='same')(act_layer_3)
		act_layer_4 = Activation(activation='relu')(conv_layer_4)
		max_pool_layer_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_4)

		conv_layer_5 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(max_pool_layer_2)
		act_layer_5 = Activation(activation='relu')(conv_layer_5)
		conv_layer_6 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_5)
		act_layer_6 = Activation(activation='relu')(conv_layer_6)
		conv_layer_7 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_6)
		act_layer_7 = Activation(activation='relu')(conv_layer_7)
		max_pool_layer_3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_7)
		
		conv_layer_8 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(max_pool_layer_3)
		act_layer_8 = Activation(activation='relu')(conv_layer_8)
		conv_layer_9 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_8)
		act_layer_9 = Activation(activation='relu')(conv_layer_9)
		conv_layer_10 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_9)
		act_layer_10 = Activation(activation='relu')(conv_layer_10)
		max_pool_layer_4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_10)

		conv_layer_11 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(max_pool_layer_4)
		act_layer_11 = Activation(activation='relu')(conv_layer_11)
		conv_layer_12 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_11)
		act_layer_12 = Activation(activation='relu')(conv_layer_12)
		conv_layer_13 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_12)
		act_layer_13 = Activation(activation='relu')(conv_layer_13)
		max_pool_layer_5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_13)

		flatten_layer = Flatten()(max_pool_layer_5)
		dense_layer_1 = Dense(units=4096)(flatten_layer)
		act_layer_14 = Activation(activation='relu')(dense_layer_1)
		dense_layer_2 = Dense(units=4096)(act_layer_14)
		act_layer_15 = Activation(activation='relu')(dense_layer_2)
		dense_layer_3 = Dense(units=num_classes)(act_layer_15)
		act_layer_16 = Activation(activation='softmax')(dense_layer_3)

		self.model = Model(inputs=input_layer,outputs=act_layer_16)
		if self.verbose or verbose:
			print("Model Successfully Created!")
		return self.model

	def compile(self,optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'],verbose=False):
		if self.verbose or verbose:
			print("Compiling Model...")

		self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
		if self.verbose or verbose:
			print("Compilation Success!")
		return self.model 

	def train(self,x_train, y_train, epochs=10, batch_size=32, validation_data=None,verbose = False):
		if self.verbose or verbose:
			print('Now Training Model...')
		self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data,callbacks=[self.cb],verbose=verbose)
		if self.verbose or verbose:
			print('Training Complete!')
		return self.history


	def evaluate(self,x_test,y_test, verbose = True):
		if self.verbose or verbose:
			print("Evaluating Model...")
		loss, accuracy = self.model.evaluate(x_test,y_test,verbose=verbose)
		if self.verbose or verbose:
			print("Evaluation Complete!")
		return loss, accuracy 

	def predict(self,sample):
		output = self.model.predict(sample)
		return output

	def summary(self):
		model_summary = self.model.summary()
		return model_summary

	def save(self,model_name="",verbose=True):
		if(model_name == ""):
			model_name = 'output_models/VGG16_' + str(round(self.history.history['accuracy'][0], 3)) + '.h5' 
		self.model.save(model_name)
		if self.verbose or verbose:
			print("Model saved to " + model_name)

	def setup_cbs(self):
		cb_list = []
		# cb_list.append(self.setup_early_stopping_cb())
		cb_list.append(self.setup_tensorboard_cb())
		return cb_list

	def setup_tensorboard_cb(self):
		model_name = 'VGG16_' + time.strftime("run_%Y_%m_%d_-%H_%M_%S")
		root_logdir = os.path.join(os.curdir, 'logs/')
		log_path = os.path.join(root_logdir, model_name)
		return keras.callbacks.TensorBoard(log_path)

	def setup_early_stopping_cb(self):
		monitor = 'val_loss'
		patience = 10
		best_weights = True
		return keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=best_weights)

class FPN_VGG16:
	def __init__(self,num_classes, verbose=False):
		self.verbose = verbose
		self.history = None
		self.feature_blocks = []
		self.model = self.create_model(num_classes)

		self.cb = self.setup_cbs()

	def create_model(self, num_classes, verbose = False):
		if self.verbose or verbose:
			print("Creating FPN-VGG16 Model...")

		# VGG16 Based Encoder
		input_layer = Input(shape=(224,224,3))
		conv_layer_1 = Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding='same')(input_layer)
		act_layer_1 = Activation(activation='relu')(conv_layer_1)
		
		max_pool_layer_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_1)

		conv_layer_3 = Conv2D(filters=128,kernel_size=(3,3),padding='same')(max_pool_layer_1)
		act_layer_3 = Activation(activation='relu')(conv_layer_3)
		conv_layer_4 = Conv2D(filters=128,kernel_size=(3,3),padding='same')(act_layer_3)
		act_layer_4 = Activation(activation='relu')(conv_layer_4)
		max_pool_layer_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_4)

		conv_layer_5 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(max_pool_layer_2)
		act_layer_5 = Activation(activation='relu')(conv_layer_5)
		conv_layer_6 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_5)
		act_layer_6 = Activation(activation='relu')(conv_layer_6)
		conv_layer_7 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_6)
		act_layer_7 = Activation(activation='relu')(conv_layer_7)
		max_pool_layer_3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_7)
		
		conv_layer_8 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(max_pool_layer_3)
		act_layer_8 = Activation(activation='relu')(conv_layer_8)
		conv_layer_9 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_8)
		act_layer_9 = Activation(activation='relu')(conv_layer_9)
		conv_layer_10 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_9)
		act_layer_10 = Activation(activation='relu')(conv_layer_10)
		max_pool_layer_4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_10)

		conv_layer_11 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(max_pool_layer_4)
		act_layer_11 = Activation(activation='relu')(conv_layer_11)
		conv_layer_12 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_11)
		act_layer_12 = Activation(activation='relu')(conv_layer_12)
		conv_layer_13 = Conv2D(filters=512,kernel_size=(3,3),padding='same')(act_layer_12)
		act_layer_13 = Activation(activation='relu')(conv_layer_13)
		max_pool_layer_5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(act_layer_13)


		# Feature Fusion using FPN
		ff_conv_layer_1 = Conv2D(filters=256,kernel_size=(1,1),padding='same')(max_pool_layer_1)
		act_layer_14 = Activation(activation='relu')(ff_conv_layer_1)
		ff_conv_layer_2 = Conv2D(filters=256,kernel_size=(1,1),padding='same')(max_pool_layer_2)
		act_layer_15 = Activation(activation='relu')(ff_conv_layer_2)
		ff_upsample_layer_1 = UpSampling2D(size=(2,2))(act_layer_15)
		ff_add_layer_1 = Add()([act_layer_14,ff_upsample_layer_1])

		ff_conv_layer_3 = Conv2D(filters=256,kernel_size=(1,1),padding='same')(max_pool_layer_3)
		act_layer_16 = Activation(activation='relu')(ff_conv_layer_3)
		ff_upsample_layer_2 = UpSampling2D(size=(2,2))(act_layer_16)
		ff_add_layer_2 = Add()([act_layer_15,ff_upsample_layer_2])

		ff_conv_layer_4 = Conv2D(filters=256,kernel_size=(1,1),padding='same')(max_pool_layer_4)
		act_layer_17 = Activation(activation='relu')(ff_conv_layer_4)
		ff_upsample_layer_3 = UpSampling2D(size=(2,2))(act_layer_17)
		ff_add_layer_3 = Add()([act_layer_16,ff_upsample_layer_3])

		ff_conv_layer_5 = Conv2D(filters=256,kernel_size=(1,1),padding='same')(max_pool_layer_5)
		act_layer_18 = Activation(activation='relu')(ff_conv_layer_5)
		ff_upsample_layer_4 = UpSampling2D(size=(2,2))(act_layer_18)
		ff_add_layer_4 = Add()([act_layer_17,ff_upsample_layer_4])


		# Forming Scale-Representative Feature Maps
		ff_conv_layer_6 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(ff_add_layer_1)
		act_layer_19 = Activation(activation='relu')(ff_conv_layer_6)
		ff_conv_layer_7 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_19)
		act_layer_20 = Activation(activation='relu')(ff_conv_layer_7)
		avg_pool_layer_1 = GlobalAveragePooling2D()(act_layer_20)

		ff_conv_layer_8 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(ff_add_layer_2)
		act_layer_21 = Activation(activation='relu')(ff_conv_layer_8)
		ff_conv_layer_9 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_21)
		act_layer_22 = Activation(activation='relu')(ff_conv_layer_9)
		avg_pool_layer_2 = GlobalAveragePooling2D()(act_layer_22)

		ff_conv_layer_10 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(ff_add_layer_3)
		act_layer_23 = Activation(activation='relu')(ff_conv_layer_10)
		ff_conv_layer_11 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_23)
		act_layer_24 = Activation(activation='relu')(ff_conv_layer_11)
		avg_pool_layer_3 = GlobalAveragePooling2D()(act_layer_24)

		ff_conv_layer_12 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(ff_add_layer_4)
		act_layer_25 = Activation(activation='relu')(ff_conv_layer_12)
		ff_conv_layer_13 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_25)
		act_layer_26 = Activation(activation='relu')(ff_conv_layer_13)
		avg_pool_layer_4 = GlobalAveragePooling2D()(act_layer_26)

		ff_conv_layer_14 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_18)
		act_layer_27 = Activation(activation='relu')(ff_conv_layer_14)
		ff_conv_layer_15 = Conv2D(filters=256,kernel_size=(3,3),padding='same')(act_layer_27)
		act_layer_28 = Activation(activation='relu')(ff_conv_layer_15)
		avg_pool_layer_5 = GlobalAveragePooling2D()(act_layer_28)

		# Concatenating the Scale-Representative Feature Maps
		concat_layer_1 = Concatenate()([avg_pool_layer_1,avg_pool_layer_2,avg_pool_layer_3,avg_pool_layer_4,avg_pool_layer_5])

		# Classification Layer
		dense_layer_1 = Dense(units=1280)(concat_layer_1)
		act_layer_29 = Activation(activation='relu')(dense_layer_1)
		dense_layer_2 = Dense(units=512)(act_layer_29)
		act_layer_30 = Activation(activation='relu')(dense_layer_2)
		dropout_layer_1 = Dropout(rate=0.5)(act_layer_30)
		dense_layer_3 = Dense(units=num_classes,activation='softmax')(dropout_layer_1)

		# Creating Model
		self.model = Model(inputs=input_layer,outputs=dense_layer_3)
		if self.verbose or verbose:
			print("Model Successfully Created!")
		return self.model

	def compile(self,optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'],verbose=False):
		if self.verbose or verbose:
			print("Compiling Model...")

		
		# loss = keras.losses.CategoricalCrossentropy(from_logits=False,class_weights)

		self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
		if self.verbose or verbose:
			print("Compilation Success!")
		return self.model 

	def train(self,x_train, y_train, epochs=10, batch_size=16, validation_data=None,verbose = False):
		if self.verbose or verbose:
			print('Now Training Model...')
		class_weights = {0: 0.26, 1: 0.29, 2: 0.45}
		self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data,callbacks=[self.cb],verbose=verbose,class_weight=class_weights)
		if self.verbose or verbose:
			print('Training Complete!')
		return self.history

	def evaluate(self,x_test,y_test, verbose = True):
		if self.verbose or verbose:
			print("Evaluating Model...")
		loss, accuracy = self.model.evaluate(x_test,y_test,verbose=verbose)
		if self.verbose or verbose:
			print("Evaluation Complete!")
		return loss, accuracy 

	def predict(self,sample):
		output = self.model.predict(sample)
		return output

	def summary(self):
		model_summary = self.model.summary()
		return model_summary

	def save(self,model_name="",verbose=True):
		if(model_name == ""):
			model_name = 'output_models/FPN-VGG16_' + str(round(self.history.history['accuracy'][-1], 3)) + '.h5' 
		self.model.save(model_name)
		if self.verbose or verbose:
			print("Model saved to " + model_name)

	def setup_cbs(self):
		cb_list = []
		cb_list.append(self.setup_early_stopping_cb())
		cb_list.append(self.setup_tensorboard_cb())
		return cb_list

	def setup_tensorboard_cb(self):
		model_name = 'FPN-VGG16_' + time.strftime("run_%Y_%m_%d_-%H_%M_%S")
		root_logdir = os.path.join(os.curdir, 'logs/')
		log_path = os.path.join(root_logdir, model_name)
		return TensorBoard(log_path)

	def setup_early_stopping_cb(self):
		monitor = 'val_loss'
		patience = 10
		best_weights = True
		return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=best_weights)
	
	def setup_learning_rate_cb(self):
		# TODO: Create learning rate callback that reduces each epoch that validation accuracy does not improve
		pass
