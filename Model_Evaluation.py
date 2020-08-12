import HWTRModel as ht



def eval_model(x_train, y_train, x_test, y_test):

    model = model = ht.HWTRModel(input_dims=(32,128,1), universe_of_discourse=80)
    model.compile(learning_rate=0.001)
    model.fit(x_train, y_train)
    pred = model.predict(x_test, decode_using_ctc=True)

    
