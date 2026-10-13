#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Prediction Using Machine Learning
# 

# ## Project Overview:
# Customer churn is an important business problem where companies try to identify customers who are likely to stop using their products or services.
# 
# This project develops a machine learning pipeline to predict customer churn using customer-related features and Logistic Regression.
# 
# The project covers the complete machine learning workflow, including data loading, exploratory data analysis, preprocessing, feature selection, model training, prediction, and model evaluation.

# ## Objective:
# Analyze customer data
# Identify and prepare relevant features
# Preprocess categorical variables
# Train a machine learning classification model
# Predict whether a customer is likely to churn
# Evaluate model performance using classification metrics
# Visualize prediction results

# In[13]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# ## Data Loading

# In[14]:


df =pd.read_csv('customer_churn_dataset-testing-master.csv')


# ## Data Understanding

# In[15]:


df


# In[16]:


df.info()


# In[17]:


df.describe()


# ## Data Quality Check

# Missing values and duplicate records were checked using:

# In[18]:


df.isnull().sum()


# ## Exploratory Data Analysis

# Exploratory Analysis was performed to understand the available customer attributes and numerical-variable distributions.
# 
# Box plots were also used to inspect numerical features and identify potential outliers.

# In[19]:


df.duplicated().sum()


# In[20]:


df['Gender'].unique()


# In[21]:


column = df.select_dtypes(include='number')
for col in column:
    sns.boxplot(data = df,x=col)
    plt.title(col)
    plt.show()


# In[22]:


df['Gender']


# ## Data Preprocessing

# In[23]:


#Categorical variables were converted into numerical representations.
mapping = {
    "Female":0,
    "Male":1
}
df['Gender'] = df['Gender'].map(mapping)


# In[24]:


df['Usage Frequency']


# In[25]:


df.info()


# In[26]:


df['Subscription Type'].unique()


# In[27]:


mapped = {
    "Basic":0,
    "Standard":1,
    "Premium":2
}
df['Subscription Type'] = df['Subscription Type'].map(mapped)


# In[28]:


df['Contract Length'].unique()


# In[29]:


from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
df['Contract Length']=Le.fit_transform(df['Contract Length'])


# In[30]:


df.info()


# ## Feature Selection

# In[31]:


x=df.drop('CustomerID',axis=1)


# In[32]:


y=df['Churn']


# In[33]:


from sklearn.model_selection import train_test_split


# ##  Train-Test Split

# In[34]:


#The dataset was divided into training and testing sets.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state =0)


# In[35]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()


# In[36]:


x_train


# ## Feature Scaling

# In[37]:


#StandardScaler was used to scale the numerical input features.
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()


# In[38]:


x_train_scaled=SS.fit_transform(x_train)


# In[39]:


x_test_scaled =SS.fit_transform(x_test)


# ## Model Training & Prediction

# In[40]:


LR.fit(x_train_scaled,y_train)


# In[41]:


y_predict=LR.predict(x_test_scaled)


# ## Model Evaluation

# In[42]:


from sklearn.metrics import accuracy_score , classification_report,confusion_matrix


# In[43]:


print(accuracy_score(y_test,y_predict))


# In[44]:


print(classification_report(y_test,y_predict))


# In[45]:


cm=confusion_matrix(y_test,y_predict)


# ## Confusion Matrix Visualization
# 
# A normalized confusion matrix was created to visualize the model's classification performance for customers who stayed and customers who churned.

# In[46]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 1. Generate your raw matrix
cm = confusion_matrix(y_test, y_predict)

# 2. Calculate row-normalized percentages (essential for imbalanced churn data)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 3. Format the cell labels: "Percentage % (Count)"
labels = [
    f"{pct:.1%}\n({count:,})" 
    for pct, count in zip(cm_norm.flatten(), cm.flatten())
]
# Reshape the list into a 2x2 grid structure
labels = np.array(labels).reshape(2, 2)

# 4. Render the Seaborn Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_norm,          # Use normalized values for the color gradient
    annot=labels,     # Overlay our custom text labels
    fmt="",           # Required when passing raw string arrays to annot
    cmap="Blues",     # Blue color scheme
    vmin=0, vmax=1,   # Forces colorbar scale between 0% and 100%
    xticklabels=["Predicted Stay", "Predicted Churn"],
    yticklabels=["Actual Stay", "Actual Churn"]
)

# 5. Clean up chart presentation
plt.title("Churn Confusion Matrix Analysis", fontsize=14, pad=15)
plt.ylabel("Actual Customer Status", fontsize=12)
plt.xlabel("Model Prediction", fontsize=12)
plt.tight_layout()
plt.show()


# In[47]:


print(cm)


# In[48]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=x_test_scaled[:, 0],      # First feature column
    y=x_test_scaled[:, 1],      # Second feature column
    hue=y_predict,       # Colors the dots based on MODEL PREDICTIONS
    palette='coolwarm',
    alpha=0.6
)

plt.title('Predicted Churn Patterns (NumPy Matrix)', fontsize=14)
plt.xlabel('Feature Index 0', fontsize=12)
plt.ylabel('Feature Index 1', fontsize=12)
plt.legend(title='Predicted Churn')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[ ]:





# In[49]:


x_test_scaled


# ## Author
# 
# Shivansh Gupta
# 
# Interested in Data Analytics, Business Analytics and Machine Learning.
