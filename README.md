# Live-Face-Detection
Machine learning model that will detect faces in real time built using the tensorflow functional API.

In this repository you will find the code to collect, annotate, load, and feed images into a two part model that leverages VGG16. This model is one part classification and one part regression. It could be described as one part finding if the image containes a face and one part locating that face and rendering a bounding box.

Alongside the main code I will also be including a "helper" file containing methods for visualization to help understand how some of the functionalities operate on each image. These code blocks can easily be implemented into the main code. 

In my use case I only collected ninety images which I split into testing, training, and validation sets manually. If a different method is desired by whoever is using this you are welcome to leverage a sklearn train test split. 

Built using libraries numpy,tensorflow,labelme,albumentation,uuid,json,time,matplotlib and opencv. 
