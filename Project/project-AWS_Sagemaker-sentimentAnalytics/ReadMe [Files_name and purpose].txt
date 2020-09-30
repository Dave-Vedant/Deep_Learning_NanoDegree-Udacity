
Hello,

here, I submit my project output files from sagemaker. 

Please see the UPDATE section;

Previous...
1. index.html (file with url)
2. predict.py (predict function /from server folder)
3. train.py (trained model / from train folder'
4. Sagemaker Deployment Project (jupyter notebook with answers and run cell)
     - I run all the cells except last one, I deleted it later. 
5. review_images (contains the images of my reviews) 
6. report.pdf (pdf version of jupyternotebook as per asking)

Here, I use higher notebook instance, ml.m4.xlarge because I had problem with the cache memory and code 
gave memory issue so, I increased my instance power. (ram and cpus). 

- I tried to train model with ml.p2.xlarge but, unfortunately I can not get access for that on AWS. I requested 
AWS for increasing my instance limit, but I can not get it on time so, I used 'ml.m4.xlarge' instance for training and it
takes 3 to 4 hours to train. 

- I also tried my best to answer all the questions, please accept it. my last question's answer is in form of review_images so please
found them in the folder "review_images". 

- please consider my project. Thank you for your time and consideration. 




======================================
UPDATE:

1. I update my notebook as per your recommondation.
2. I use previously generated model and load it to get test-data prediction again

The change:
1. Change most frequent word list (consider first five - as I had [reverse=True]
2. Change the test_data_len function as per your suggession (I considered directory len as full len, which was wrong)

Please review the change, thank you for your time and consideration.