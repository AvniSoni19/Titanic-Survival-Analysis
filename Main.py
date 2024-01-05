import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv('C:/Users/HITESH SONI/OneDrive/Desktop/Python/Projects/Bharat Intern/Titanic Survival/titanic.csv')

print(titanic_data.isnull().sum())

titanic_data = titanic_data.drop(columns='Cabin' , axis=1)

titanic_data['Age'].fillna(titanic_data['Age'].mean() , inplace=True)

titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0] , inplace=True) 

print(titanic_data.isnull().sum())
