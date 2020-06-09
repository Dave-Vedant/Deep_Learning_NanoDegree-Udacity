## Bike Sharing Prediction [Neural Network Approach]
---



### Main features.
- 2 layer Neural Network with foreward/backward pass
- hyper parameter tuning (performance improvement)
- Accuracy -> training : 98.3 % , validation : 93 % 


---

### Environmnet (downloaded packages):
environment location: C:\Users\HP\anaconda3\envs\deep-learning

  added / updated specs:
    - python=3


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    certifi-2020.4.5.1         |           py38_0         156 KB
    python-3.8.2               |      h5fd99cc_11        19.6 MB
    ------------------------------------------------------------
                                           Total:        19.7 MB

The following NEW packages will be INSTALLED:

  ca-certificates    pkgs/main/win-64::ca-certificates-2020.1.1-0
  certifi            pkgs/main/win-64::certifi-2020.4.5.1-py38_0
  openssl            pkgs/main/win-64::openssl-1.1.1f-he774522_0
  pip                pkgs/main/win-64::pip-20.0.2-py38_1
  python             pkgs/main/win-64::python-3.8.2-h5fd99cc_11
  setuptools         pkgs/main/win-64::setuptools-46.1.3-py38_0
  sqlite             pkgs/main/win-64::sqlite-3.31.1-he774522_0
  vc                 pkgs/main/win-64::vc-14.1-h0510ff6_4
  vs2015_runtime     pkgs/main/win-64::vs2015_runtime-14.16.27012-hf0eaf9b_1
  wheel              pkgs/main/win-64::wheel-0.34.2-py38_0
  wincertstore       pkgs/main/win-64::wincertstore-0.2-py38_0
  
  The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    backcall-0.1.0             |           py38_0          21 KB
    bleach-3.1.4               |             py_0         114 KB
    cycler-0.10.0              |           py38_0          14 KB
    entrypoints-0.3            |           py38_0          11 KB
    importlib_metadata-1.5.0   |           py38_0          49 KB
    ipykernel-5.1.4            |   py38h39e3cac_0         176 KB
    ipython-7.13.0             |   py38h5ca1d4c_0        1010 KB
    ipython_genutils-0.2.0     |           py38_0          40 KB
    jedi-0.16.0                |           py38_1         786 KB
    jsonschema-3.2.0           |           py38_0         113 KB
    jupyter-1.0.0              |           py38_7           8 KB
    jupyter_core-4.6.3         |           py38_0          98 KB
    kiwisolver-1.0.1           |   py38ha925a31_0          51 KB
    markupsafe-1.1.1           |   py38he774522_0          29 KB
    matplotlib-3.1.3           |           py38_0          22 KB
    matplotlib-base-3.1.3      |   py38h64f37c6_0         4.9 MB
    mistune-0.8.4              |py38he774522_1000          55 KB
    nbconvert-5.6.1            |           py38_0         471 KB
    notebook-6.0.3             |           py38_0         4.3 MB
    pandas-1.0.3               |   py38h47e9c7a_0         7.6 MB
    pandocfilters-1.4.2        |           py38_1          14 KB
    parso-0.6.2                |             py_0          70 KB
    pickleshare-0.7.5          |        py38_1000          14 KB
    pyqt-5.9.2                 |   py38ha925a31_4         3.2 MB
    pyrsistent-0.16.0          |   py38he774522_0          96 KB
    pywin32-227                |   py38he774522_1         5.6 MB
    pywinpty-0.5.7             |           py38_0          52 KB
    pyzmq-18.1.1               |   py38ha925a31_0         407 KB
    send2trash-1.5.0           |           py38_0          18 KB
    sip-4.19.13                |   py38ha925a31_0         262 KB
    terminado-0.8.3            |           py38_0          26 KB
    tornado-6.0.4              |   py38he774522_1         610 KB
    traitlets-4.3.3            |           py38_0         138 KB
    webencodings-0.5.1         |           py38_1          20 KB
    widgetsnbextension-3.5.1   |           py38_0         863 KB
    ------------------------------------------------------------
    
    
   ### Local Setup:
    (deep-learning) C:\Users\HP\deep-learning-v2-pytorch>jupyter notebook
[I 23:45:14.373 NotebookApp] Serving notebooks from local directory: C:\Users\HP\deep-learning-v2-pytorch
[I 23:45:14.373 NotebookApp] The Jupyter Notebook is running at:
[I 23:45:14.374 NotebookApp] http://localhost:8888/?token=af2d5a41eb8a74d34c3cf82d0fec4b65bb78483e804d15fe
[I 23:45:14.374 NotebookApp]  or http://127.0.0.1:8888/?token=af2d5a41eb8a74d34c3cf82d0fec4b65bb78483e804d15fe
[I 23:45:14.374 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 23:45:14.476 NotebookApp]

---

### Business Problem: 

---

### Project Data
#### [Data](https://github.com/vedantdave77/project.Orca/tree/master/Project/project-bikesharing/Bike-Sharing-Dataset)

---

### Project Model explaination :
Please visit interactive code : **[My_Network](https://github.com/vedantdave77/project.Orca/blob/master/Project/project-bikesharing/My_Neural_Network_Approach.py.py)**

Model Implementation : **[Bike_Sharing_prediction_with_Neural_Network.ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Project/project-bikesharing/Bike_Sharing_prediction_with_Neural_Network.ipynb)** 

---

#### Project Results (Ateemptes)
Multi_try with 2000 iteration but not get good result.
Please read like...

iteration, learning_rate, hidden_nodes, output_nodes ---> training_loss, validation_loss

3000,1,12,1 --> 0.074, 0.143
3000,1,15,1 ----> 0.061,0.134
3000,1,18,1 -----> 0.230, 0.413 (overfit, after 75 to 80 percent process)
3000,1,16,1 ----> 0.1,0.203 (still not work so try to lr low and more iteration)
4000,0.8,16,1 ---> 0.064, 0.164
4000,0.9,16,1---> 0.063,0.139
4000,0.8,18,1----> 0.088,0.289
4000,0.9,18,1 ----> 0.061,0.137
7000,0.6,18,1 ----> 0.051,0.146 (best training accuracy yet)
7000,0.7,18,1 ----> 0.052,0.138 (better) -- approved model
7000,0.7,22,1 ----> 0.062,0.143
7000,0.7,20,1 ---->0.057, 0.141
10,000,0.7,20,1 ------> 0.058, 0.131

Thank you. 
---
#### Keep learning, Enjoy Empowering. -@dave117
