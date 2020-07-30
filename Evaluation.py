def eval_model(Xdata, Ydata, Xtest, Ytest):
    scores, histories = list(), list()
    kfold = KFold(5, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(Xdata):
        model = define_model()
        TrainX, TrainY, TestX, TestY = Xdata[train_ix], Ydata[train_ix], Xdata[test_ix], Ydata[test_ix]
        history = model.fit(TrainX, TrainY, epochs=40, batch_size= 32, validation_data=(TestX, TestY), shuffle=True, verbose=0)
        _, acc = model.evaluate(Xtest, Ytest, verbose=0)
        scores.append(acc)
        histories.append(history)
    return scores, histories
	

def summarize_performance(scores):
    print("Accuracy: mean=%.3f std=%.3f, n=%d" % (mean(scores)*100, std(scores)*100, len(scores)))
    pyplot.boxplot(scores)
    pyplot.show()
