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
* Filename -> word: 
