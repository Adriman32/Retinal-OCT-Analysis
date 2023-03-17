import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten, Dropout, Concatenate, Input

class VGG16:
	def __init__(self,verbose=False):
		self.verbose = verbose
		self.history = None
		self.feature_blocks = []
		self.model = self.create_model()

	def create_model(self, verbose = False):
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
		dense_layer_3 = Dense(units=4)(act_layer_15)
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

	def train(self,x_train,y_train,verbose = True):
		if self.verbose or verbose:
			print('Now Training Model...')
		self.history = self.model.fit(x_train,y_train,verbose=verbose)
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
