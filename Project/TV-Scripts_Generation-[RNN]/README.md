# Project-TV-Script-Generation-Udacity
This is the fourth project of Deep Learning Nanodegree Program (Using a DCGAN on the CelebA dataset to generate images of new and realistic human faces.)


# Project Overview: TV Script Generator
In this project, I have generated my own Seinfeld TV scripts using RNNs.I have used part of the Seinfeld dataset of scripts from 9 seasons. The Neural Network build will generate a new ,"fake" TV script, based on patterns it recognizes in the training data.

# Installation:

1. For running this project on your local computer, first make sure you have git by typing `git --version` on cmd, if version number appears that means you have git installed. Go ahead and clone the repository:

```
git clone https://github.com/Sidrah-Madiha/Project-TV-Script-Generation-Udacity.git
cd Project-TV-Script-Generation-Udacity

```
2. Now please open the file with filename: dlnd_tv_script_generation.ipynb


# Dependencies:

- Make sure to create an environment for running this project using conda (you can install [Miniconda](http://conda.pydata.org/miniconda.html) for this

- Once you have Miniconda installed, please make an environment for the project like so: 
```
conda create --name TV-Script-Generator  python=3.6
activate TV-Script-Generator

```
- Install Pytorch: 
```
conda install pytorch -c pytorch
pip install torchvision
```

- Install a few required pip packages, which are specified in the requirements text file.

`pip install -r requirements.txt`
