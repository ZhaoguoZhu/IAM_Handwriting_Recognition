import HWTRModel as ht



def eval_model(x_train, y_train, x_test, y_test):

    model =  ht.HWTRModel(input_dims=(32,128,1), universe_of_discourse=80)
    model.compile(learning_rate=0.001)
    model.fit(x_train, y_train)
    model.save("final_model.h5")

    
