# DataScienceWebApp
A data science web app to check if a message is spam or not.
Model Used:
Naive Bayes
Dependencies:
Python 3.6
dataset: .csv file
  Python libraries:
         pandas -> to access and make changes to the dataset.
      streamlit -> for designing the Web App.
         nltk   -> Natural Language Toolkit (has been used for datacleaning)
        sklearn -> For using a Machine Learning model.
The user can input a message and the NLP model will predict if the message is spam or not.
The CompleteCode.ipynb has the entire code with comments for easily understanding the whole project.
As the streamlit library requires files in .py format, the notebook was converted into  the following .py files using ipynb and nbconvert packages:
NLP_model -> contains the code for accessing and training the data.
Streamlit_code -> contains the code for launching the WebApp.
Procedure for using the project:
download all files.
make sure all dependencies are present. (pip install can be used to install all packages)
run command prompt from the downloaded project folder.
type the command -> streamlit run Streamlit_code.py
The web app will open automatically.
