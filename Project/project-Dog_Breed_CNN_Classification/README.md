# Dog Breed Image Classification Algorithm for Application 

### Objective :
Main objective of this project is to develope proper algorithm procedure for dog breed image classification application. Algorithm must understand the difference between human and dogs, as most of images of dogs also have image of owner with it. For fun part Algorithm can identiy human as nearest dog breed type features. 

#### Technology :
- I used **Google colaboratory environment** as Jupyter Notebook, because it is easy to install all required dependencies as per need.
- **Python 3.7** is basic need for programming purpose.
- I used **PyTorch library (specically, Pytorch vision and basic)** for Neural Network architecture and for other machine learning procedure.
- Other **libraries** are : (1) matplotlib (2) Numpy (3) Pandas (4) PIL (5) Beautifulsoup(if needed) 
- For GPU, I used **CUDA**, and that's why you can see the use of **Tesla P4 and Tesla P 100**, in different notebooks. [ *GPU are better than TPU*  ]

#### Dependencies 
All dependencies, you can find in [Requirement.txt file]() in main project.Orca. (I use the same env. for all the project across my deeplearing module. 

#### Run time:
My average runtime was 5 to 6 hours for custome CNN.

#### Project Planning.
My first objective is to identifying best approach, means...
- First, I tryied pre trained algorithms which already trained on ImageNet dataset. By using, them I get the testing accuracy (88%) from my self. I also compare different network to see the best possible answer. The algorithm I used are : alexnet, vgg, resnet, resnext, etc...
- Secondly, I want to try my own algorithm from scratch, and I implment it successfully, but the accuracy I got is very low (63%), which are quiet less. The only advantage is, my network is quiet shallow than all pretrained networks. My network has 11 deep layer with 3 different layer group. I also try deep network but due to notebook kernel instability and long time gpu use, I can not train network for 100 epoch. 

 #### Project Files and Flow:
 >  There are total three files (mainly) and all of them have their own significance as follow.
 
 [1] [Transfer_learning_Approach_and_Comparison.ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Project/project-Dog_Breed_CNN_Classification/Transfer_learning_Approach-and-Comparison.ipynb)
  - First, I apply it to humnan face detection with the use of casecade file from github.  (face_detection.xml), and get successful result.
 - This file contains the basic primary approach of using pretrained network for image classification. Due to ImageNet's individual classes from 151 to 168 (for dog identification features / out of 1000.) its easy for me to get best answer with 99 % true positive and (0 to 7) % false positive ratio. 
 - I used same procedure for all type of pretrained popular netoworks, and get the best answer in most. I also compare all of them with bar graph give me best possible answer to decide final algorithm network out ot them. 
 
  [2] [CNN_Classfier-using_ PyTorch.ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Project/project-Dog_Breed_CNN_Classification/CNN_Classfier-using_%20PyTorch.ipynb)
 - In this notbook, I develop my own CNN algorithm from scratch using pytorch. and the steps are as follow.
1. load, and Trasnfer data
2. Model Architecture
3. Add Loass and Optimizer
4. Train and Validate Model
5. Model Testing
5. Final Conclusion for App _ Algorithm
 
 
