# Handwriting Recognition with customed Keras model
## Contribution
  
Author: Zhaoguo Zhu <br/>
Institute: College of Arts and Sciences, Boston University<br/>
Instructor and Supervisor: Dr. Dharmesh Tarapore <br/>
E-mail: zhuzg@bu.edu  

## Introduction

This python project is a pipeline inspiration of Handwriting Recognition challenge from IAM dataset website. The customed HWTRModel consists of 7 Convolutional layers(CNN), 4 Recurrent Neural Network layers(RNN) and Connectionist Temporal Classification loss calculation and decoding layer(CTC). The neural network walk through is in the following format:  

* Pass in the images in grayscale format with only one color channel activated and a shape of 32x128  
* 7 CNN layers provide a feature mapping to a sequence of size 32x256  
* 4 RNN layers provide a feature mapping to a sequence of size 32x80 with each column represents the possibility of one of the 80 characters in English word (26 lower case; 26 upper case; 26 signs and marks; one space character; one None character)  
* A CTC layer which either calculates the loss value given the matrix or decodes the 32x80 matrix with best path and word beam search decoding then providing a predicted result of size that might vary based on the actual size of the predicted word  

Caution: This project waits for further modification since some of the implementation is inefficient and incomplete. For instance, the customed model has no ['accuracy'] metrics implemented due to the fact that keras.model.compile() metrics accuracy compares the output with input by looking at the word as a whole part. If even one single character within the word is predicted wrong, the accuracy will be zero. Please wait for further update for better performances which will turn this pipeline into a machine that can solve real life handwriting recognition problem.  

## File Desciption  

### Two Dictionary python files:  

These two function return two dictionary that will support the data preprocessing:  
* Key = filename -> value = word. Sample: "a01-000u-00-00.png" -> "A" (String -> String)  
* Key = character -> value = integer representation. Sample: "A" -> 7 (String -> int)  
The integer representation will vary each time the function compile

### Data_Preprocessing.py:  

Iterate through words/ directory and transforming them into shape of 32x128 grayscale images and then turn them into numpy arrays which will be stored in TrainX and TestX numpy arrays. After finding the corresponding word of a image through the first dictionary, iterate through every single character and turn them into integer representation and store them in an array of fixed size of length of 32 by looking into the second dictionary. For example: "a01-000u-00-00.png" -> "A" -> [7,0,0,0,...,0].  

### Loading_Normalize.py:  

As we know a color activation can vary from 0 to 255. For easy training purposes, we transform each pixel from range 255 to range 1 in float type.  

### HWTRModel.py and Model_Evaluation.py:  

The first python file consists of the model described in introduction and customed method including fit(), predict(), summary(), compile() and most importantly the custom_ctc_loss_function() that return the loss in model compiling. Model_Evaluation will initiate the model, train the model and save the model.  

Model saved can be used for prediction already but user have to write a python file to load the model locally since this project is yet incompleted.  

### main.py:

run python main.py and you are ready to go.  

## Future goals:  

* Custom accuracy metrics  
* Improve accuracy by adding epochs, dropout, data augmentation, shifting optimizer  
* Adding Validation dataset and K-fold cross validation feature to check overfitting intensity  

## References:  
\[1\] [IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)  
\[2\] [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)  
\[3\] [CTC Model](https://github.com/ysoullard/CTCModel)  
\[4\] [Simple HTR](https://github.com/githubharald/SimpleHTR)
