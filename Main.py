from Data_Preprocessing import prepare_data
from Loading_and_Normalize import normalize_pixels
from Model_Evaluation import eval_model
from Data_Preprocessing import prepare_data


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from numpy import asarray

def exe():
    TrainX, TrainY, TestX, TestY = prepare_data()

    '''
    TrainX, TrainY, TestX, TestY= list(), list(), list(), list()    
    image = load_img("words/a01/a01-000u/a01-000u-00-00.png", target_size=(32, 128), color_mode="grayscale")
    image = img_to_array(image)    
    TrainX.append(image)
    TestX.append(image)
    arr = [1,2]
    TrainY.append(arr)
    TestY.append(arr)
    TrainX = asarray(TrainX)
    print(type(TrainX))
    print(TrainX)
    TrainY = asarray(TrainY)
    print(type(TrainY))
    print(TrainY)
    TestX = asarray(TestX)
    TestY = asarray(TestY)
    '''
     
    TrainX, TestX = normalize_pixels(TrainX, TestX)
    print("did i get here")
    eval_model(TrainX, TrainY, TestX, TestY)

exe()
