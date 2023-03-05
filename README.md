# Live-Face-Detection
Machine learning model that detects faces in real time. Built using the tensorflow functional API.

In this repository you will find the code to collect, annotate, load, and feed images into a two part model that leverages VGG16. This model is one part classification and one part regression. It could be described as one part finding if the image contains a face then locating that face and rendering a bounding box.

Alongside the main code I will also be including a "helper" file containing methods for visualization to help demonstrate how some of the functionalities transform each image on a lower level. These code blocks can easily be implemented into the main code. 

In my use case I only collected ninety images which I split into testing, training, and validation sets manually. If a different method is desired by whoever is using this program, you are welcome to leverage a sklearn train test split to abstract the manual partitioning of data. 

This program was built using libraries numpy,tensorflow,labelme,albumentation,uuid,json,time,matplotlib and opencv. 
