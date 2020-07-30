def define_model():
    
    # 7 Convolutional Layers 
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", input_shape=(32,128,1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(2048, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
	
	
    # 4 Recurrent Layers
    model.add(LSTM(128, input_shape(32,128,1), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128,  activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128,  activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(80,  activation='softmax', return_sequences=True))
    model.add(Dropout(0.2))

	
	
    # 1 CTC Encoder and Decoder
	
	
	
    # Flattern
    model.add(Flatten())




    # Model Compile
    model_optimizer = Adam(lr=0.001)
    model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
