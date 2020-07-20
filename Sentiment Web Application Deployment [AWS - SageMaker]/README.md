# Sentimment Analysis Deployment

> Final Result at the end of ReadMe. :)
### Technoltoy 
- Python 3.7
- Jupyter Notebook : AWS Sagemaker Notebook Instance 
- Storage : AWS S3
- Deployment : AWS Lambda 
- API : AWS gateway API (Resetful API)

![](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/Diagram%20%5BS3-SageMaker-Lambda-API-WebAPP%5D.svg)

### Dependency 
Please visit **[requirement.txt]()**

### Project Intuition : 
Customer sentient are serious concern for any industry. Specially, when your prime goal is to provide best services to Customer. Customer Satisfaction always depends on their response. In big data world, it is not easy to identify each person's sentiment. Machine learning give use leverage to classify positive and negative reviews so, we can understand the problem betterly, and reach unsatisfy customer personally. 

### Goal:
- Identifing review position. [Positive/ Negative]
- Deploy model to Web to show customer their review position. 


### Project Flow 
1. Data Preparation
2. Extract "bag of words" algo. for NLP approach
3. XGBoost Random Forest Implementation (searching right algorithm, using AWS algo. lib)
4. Configure AWS for training
5. Model training, validation 
6. Hyperparameter Tuning
7. Model Testing
8. End Point generation
9. API gateway generation
10. [Index.html](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/Web_Application.html) creation
11. Deployment
12. Final Testing  (real review checking)

### Project Files:
- [Batch_Transform(python SDK-Testing).ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/Batch_Transform(python%20SDK-Testing).ipynb) : AWS Python SDK- Batch Transform Notebook. Useful for working on whole dataset, specially during development, preprocessing of data, model training and testing. 
- [HyperParameter_Tuning with sagemaker.ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/HyperParameter_Tuning%20with%20sagemaker.ipynb) : Apply Hyperparameter tuning for getting best model accuracy, and optimize model. 
- [Sentimental_Analytics(AWS_SageMaker_Updating_Model).ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/Sentimental_Analytics(AWS_SageMaker_Updating_Model).ipynb) : Improve accuracy by adding more data and update model with new parameter values. Very helpful to improve model performance and result accuracy.
- [AWS-Lambda-Web_deployment-Sentiment Analysis .ipynb](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/AWS-Lambda-Web_deployment-Sentiment%20Analysis%20.ipynb) : Use AWS Lambda to generate endpoints and use RESTAPI Gateway to deploy model in production. 
-- Note: You can see experimental demo at the end. (gif format)
### Problem and Solution




### The final result of my project :

![](https://github.com/vedantdave77/project.Orca/blob/master/Sentiment%20Web%20Application%20Deployment%20%5BAWS%20-%20SageMaker%5D/Sentiment-Analysis-Deployment.gif)



### Keep Learning, Enjoy Empowering. -@dave117
