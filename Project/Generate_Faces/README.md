# Project-Face-Generation-Udacity
This is the fourth project of Udacity's Deep Learning Nanodegree Program (Using a DCGAN on the CelebA dataset to generate images of new and realistic human faces.)

# Project Overview: Face Generation using DCGAN
In this project, I have defined and trained a DCGAN on a dataset of faces. My goal was to  get a generator network to generate new images of faces that look as realistic as possible! Similar to this https://www.thispersondoesnotexist.com/

At the end of the project, I was able to visualize the results of my trained Generator to see how it performsed; my generated samples looked like fairly realistic faces with small amounts of noise.

# Installation:

1. For running this project on your local computer, first make sure you have git by typing `git --version` on cmd, if version number appears that means you have git installed. Go ahead and clone the repository:

```
git clone https://github.com/Sidrah-Madiha/Project-Face-Generation-Udacity.git
cd Project-Face-Generation-Udacity

```
2. Now please open the file with filename: dlnd_face_generation.ipynb


# Dependencies:

- Make sure to create an environment for running this project using conda (you can install [Miniconda](http://conda.pydata.org/miniconda.html) for this

- Once you have Miniconda installed, please make an environment for the project like so: 
```
conda create --name face_generation python=3.6
activate face_generation

```
- Install Pytorch: 
```
conda install pytorch -c pytorch
pip install torchvision
```

- Install a few required pip packages, which are specified in the requirements text file.

`pip install -r requirements.txt`
