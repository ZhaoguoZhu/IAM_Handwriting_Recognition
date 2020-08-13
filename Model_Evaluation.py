import HWTRModel as ht
from numpy import std, mean


def summarize_performance(scores):
    print("Accuracy: mean=%.3f std=%.3f, n=%d" % (mean(scores)*100, std(scores)*100, len(scores)))
    #pyplot.boxplot(scores)
    #pyplot.show()

def eval_model(x_train, y_train, x_test, y_test):
    scores = list()
    
    model = ht.HWTRModel(input_dims=(32,128,1), universe_of_discourse=80)
    model.compile(learning_rate=0.001)
    model.fit(x_train, y_train)
    #model.save("final_model.h5")
    #pred,_ = model.predict(x_test, decode_using_ctc=True)
    #print(pred)
    _, acc = model.evaluate(x_test, y_test, verbose=1)
    scores.append(acc)
    summarize_performance(scores)
    
