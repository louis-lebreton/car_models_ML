
"""
Machine Learning Project
01/2024
LEBRETON Louis

Packages
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler # to normalize the data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans # K-Means
from scipy.cluster.hierarchy import dendrogram, linkage # CAH
from sklearn.decomposition import PCA


#################################################################################################
# 0 - Phase préliminaire ########################################################################
#################################################################################################

# Choix du repertoire de travail
os.chdir('C:/Users/lebre/OneDrive/Bureau/Projet')

# Data import
cars=pd.read_csv("data/cars",sep='\s+')
simu=pd.read_csv("data/simu.txt",sep=' ')
xsimutest=pd.read_csv('data/xsimutest.txt',sep=' ')
dogs=pd.read_csv("data/dogs",sep='\s+')

# Analysis: missing values
cars.isna().sum()
simu.isna().sum()
dogs.isna().sum()
# no missing values

# Analysis: duplicates
cars[cars.duplicated()]
simu[simu.duplicated()]
dogs[dogs.duplicated()]
# no duplicates

# Dataframes descriptions
cars.describe()
simu.describe()
dogs.describe()

# Univariate analysis
sns.histplot(simu['X1'],kde=True)
plt.title('Distribution of X1 (simu)')
plt.show()

sns.histplot(simu['X2'],kde=True,color='red')
plt.title('Distribution of X2 (simu)')
plt.show()

# Boxplots
plt.figure(figsize=(15,10)) 
for i, column in enumerate(cars.columns):
    plt.subplot(len(cars.columns),1,i+1) 
    sns.boxplot(x=cars[column])
    plt.title(column)

plt.tight_layout()
plt.show()

# Bivariate analysis

# Power x Speed
plt.scatter(cars.PUIS,cars.VITESSE,s=15)
plt.xlabel('Power')
plt.ylabel('Speed')
plt.title('Power x Speed')

# Speed x CO2
plt.scatter(cars.VITESSE,cars.CO2,s=15,color="purple")
plt.xlabel('Speed')
plt.ylabel('CO2')
plt.title('Speed x CO2')

# X1 x X2
plt.scatter(simu.X1,simu.X2,s=15,color="#51A07B")
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('X1 x X2')

# Multivariate analysis
sns.heatmap(cars.corr(),annot=True,cmap='coolwarm_r')
plt.title('Correlation Matrix - Car Data')
plt.show()



#################################################################################################
# 1 - Binary Regression ########################################################################
#################################################################################################

## 1 - Model Selection ##

# Dividing the dataset into train and test
x=simu[['X1','X2']]
x=StandardScaler().fit_transform(x) # data normalization
y=simu['Y']
# 80% in the train sample and 20% in the test sample
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

# Logistic Regression
model= LogisticRegression()
model.fit(x_train,y_train)

# Model evaluation: Logistic Regression
predictions=model.predict(x_test)
print(confusion_matrix(y_test, predictions)) # confusion matrix
print(classification_report(y_test,predictions)) # classification report

# Probability predictions
proba= model.predict_proba(x_test)[:,1]  # probability of belonging to the positive class
proba
# AUC
auc= roc_auc_score(y_test,proba)
auc # 0.61
          
# Quadratic Discriminant Analysis
model_qda=QuadraticDiscriminantAnalysis()
model_qda.fit(x_train, y_train)

# Model evaluation: Quadratic Discriminant Analysis
predictions_qda=model_qda.predict(x_test)
print(confusion_matrix(y_test, predictions_qda)) # confusion matrix
print(classification_report(y_test, predictions_qda))

# Probability predictions
proba= model.predict_proba(x_test)[:,1]  # probability of belonging to the positive class
proba
# AUC
auc= roc_auc_score(y_test,proba)
auc # 0.61

# Decision Tree 1

model=DecisionTreeClassifier(min_samples_leaf=30)
model.fit(x_train, y_train)
predictions=model.predict(x_test)
# Model evaluation
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Probability predictions
proba= model.predict_proba(x_test)[:,1]  # probability of belonging to the positive class
proba
# AUC
auc= roc_auc_score(y_test,proba)
auc # 0.81

# Decision Tree 2

model=DecisionTreeClassifier(max_depth =2)
model.fit(x_train, y_train)
predictions=model.predict(x_test)
# Model evaluation
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Probability predictions
proba= model.predict_proba(x_test)[:,1]  # probability of belonging to the positive class
proba
# AUC
auc= roc_auc_score(y_test,proba)
auc # 0.72

# Representation of Tree 2
plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=['X1', 'X2'], class_names=['Class 1', 'Class 2'])
plt.show()

# Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=0, min_samples_split=10)
model.fit(x_train, y_train)

# Model Evaluation
predictions = model.predict(x_test)
print(confusion_matrix(y_test, predictions)) # confusion matrix
print(classification_report(y_test, predictions))

# Predicting Probabilities
proba = model.predict_proba(x_test)[:,1]  # probability of belonging to the positive class
proba
# AUC
auc = roc_auc_score(y_test, proba)
auc # 0.86

## 2 - Prediction of xsimutest ##

xsimutest = StandardScaler().fit_transform(xsimutest) # data normalization
# Prediction
predictions = model.predict(xsimutest)
# Exporting Predictions
pd.Series(predictions).to_csv('data/predictions.txt', index=False, header=False, sep='\n')


#################################################################################################
# 2 - PCA #######################################################################################
#################################################################################################


####### 1 - Explained Inertia Percentage #######

# Data Normalization
cars_norm = StandardScaler().fit_transform(cars)

# Performing PCA
pca = PCA()
pca.fit(cars_norm)

# ratio of variance explained by each principal component
print(pca.explained_variance_ratio_)
# percentage of inertia explained by the first 3 and 2 principal components
sum(pca.explained_variance_ratio_[:3]) # 0.9475
sum(pca.explained_variance_ratio_[:2]) # 0.8874

####### 2 - Correlations of Variables with the First Two Axes #######

# PCA with 2 principal components
pca = PCA(n_components=2)
pca.fit(cars_norm)

# eigen vectors
pca.components_.T
# df of eigen vectors
eigen_vector_df = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=list(cars.columns))
eigen_vector_df

####### 3 - Representation on the First Factorial Plane #######

# quality of representation of each observation
X_projected = pca.transform(cars_norm)
cos2_row = (X_projected**2) / np.sum(X_projected**2, axis=1, keepdims=True)

# Contribution of each variable to principal components
cos2_col = (pca.components_.T** 2) / np.sum(pca.components_.T**2, axis=0, keepdims=True)

cos2_row_df = pd.DataFrame(cos2_row, index=cars.index, columns=['PC1', 'PC2'])
cos2_col_df = pd.DataFrame(cos2_col, index=cars.columns, columns=['PC1', 'PC2'])
cos2_row_df
cos2_col_df

##### Correlation Circle #####

vp = pca.components_.T

# creating the circle
plt.figure(figsize=(10,10))
circle = plt.Circle((0,0), 1, color='black', fill=False)
plt.gca().add_artist(circle)

for i in range(len(vp)):
    plt.arrow(0, 0, vp[i,0], vp[i,1], color='red')
    plt.text(vp[i,0]*1.1, vp[i,1]*1.1, cars.columns[i], color='blue', ha='center', va='center')

plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.axhline(0, color='grey', linestyle='--')
plt.axvline(0, color='grey', linestyle='--')
plt.xlabel('Principal Component 1 (69.37%)')
plt.ylabel('Principal Component 2 (19.38%)')
plt.title('Correlation Circle')

plt.show()

##### Display of Individuals #####

result = pca.fit_transform(cars_norm)

pc1 = result[:,0]
pc2 = result[:,1]

# plotting in 2D

def plot_scatter(pc1, pc2):
    fig, ax = plt.subplots(figsize=(10,10))
    
    for i, c in enumerate(cars.index):
        plt.scatter(pc1[i], pc2[i], label=c, s=20)
        try:
            ax.annotate(c, (pc1[i], pc2[i]))
        except:
            ax.annotate(c, (pc1[i], pc2[i]))
    
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.xlabel('Principal Component 1 (69.37%)')
    plt.ylabel('Principal Component 2 (19.38%)')
    plt.grid()
    
    plt.axis([-4,7.5,-3,3.5])
    
    
    
plot_scatter(pc1, pc2)

plt.show()

#################################################################################################
# 3 - Classification ############################################################################
#################################################################################################

####### 1 - K-Means #######

# Elbow method
inertia_list = []

K = range(1,10) 
for k in K:
    kmeanModel = KMeans(n_clusters=k,n_init=20)  # 20 intitialisations stage 0
    kmeanModel.fit(cars_norm)
    inertia_list.append(kmeanModel.inertia_)

plt.figure(figsize=(10,6))
plt.plot(K,inertia_list,'rx-')
plt.xlabel('Nombre de clusters')
plt.ylabel('Variance intra-cluster')
plt.title('Elbow method / Méthode du coude')
plt.show()

# K-Means 3 clusters
kmeans = KMeans(n_clusters=3,n_init=20) # 20 intitialisations stage 0
kmeans.fit(cars_norm)



# Power x Speed
plt.scatter(cars.PUIS,cars.VITESSE,c=kmeans.labels_,s=15,cmap='rainbow' )
plt.xlabel('Power')
plt.ylabel('Speed')
plt.title('Power x Speed')

# Speed x CO2
plt.scatter(cars.VITESSE,cars.CO2,c=kmeans.labels_,s=15,cmap='rainbow')
plt.xlabel('Speed')
plt.ylabel('CO2')
plt.title('Speed x C02')

# adding our clusters to our cars dataframe
cars['cluster'] = kmeans.labels_

# averages of variables by clusters
means = cars.groupby(['cluster']).mean()
means
# export this dataframe to an excel file
# means.to_excel(r'means.xlsx', index=False)

# variances of the variables by clusters
sd = cars.groupby(['cluster']).std()
sd

# number of cars by clusters
cars.groupby(['cluster']).size()
# cars by clusters
cars[['cluster']].sort_values(by=['cluster'])

## calculation of inter-cluster inertia / total inertia
# intra-cluster inertia
inertia_intra = kmeans.inertia_

# total inertia
mean = np.mean(cars_norm, axis=0)
total_inertia = np.sum((cars_norm - mean)**2)

# inter-cluster inertia / total inertia
100 - (inertia_intra / total_inertia) * 100 # 69%

####### 2 - Hierarchical clustering #######

Z = linkage(cars, method='ward') # hierarchical clustering with Ward's method

# Displaying the Dendrogram
plt.figure(figsize=(15,10))
plt.title("Dendrogram of the Hierarchical clustering of the car data")
dendrogram(Z, labels=cars.index, above_threshold_color='y', orientation='top', leaf_rotation=90)

plt.xlabel('Cars')
plt.ylabel("Ward's Distance")

plt.axhline(y=2000, c='red', lw=1, linestyle='dashed') # chosen cut line

plt.show()