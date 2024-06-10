#!/usr/bin/env python
# coding: utf-8

# # CS229: Using Machine Learning to Predict Admissions to Higher Education

# In[1]:


import statsmodels.formula.api as smf


# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import statsmodels.api as sm


# ## Data

# In[17]:


df = pd.read_csv("parcoursup.csv")


# In[18]:


df.shape


# ## Variable Exploration

# In[19]:


df.head()


# In[20]:


df['Percentile'] = df['Classement'].rank(pct=True)


# In[21]:


df['Rank'] = 1-df['Percentile']


# In[22]:


df['label'] = pd.qcut(df['Percentile'], q=5, labels=False) + 1


# In[23]:


df.head()


# In[24]:


df.columns


# In[25]:


plt.hist(df['label'], bins=10, color='skyblue', edgecolor='black')

# Add labels and a title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Data')


# In[26]:


correlations = df[['TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2', 
                   'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2',
                   'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.','Rank']].corr()


# In[27]:


cov = df[['TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2', 
                   'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2',
                   'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.','Rank']].cov()


# In[28]:


mean_values = df[['TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2', 
                   'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2',
                   'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.']].mean().round(2)

# Calculate the standard deviation of 'a', 'b', 'c', 'd' columns
std_values = df[['TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2', 
                   'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2',
                   'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.']].std().round(2)

# Calculate the correlation between 'a', 'b', 'c', 'd' columns and 'y' column
correlation_values = df[['TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2', 
                   'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2',
                   'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.']].corrwith(df['Rank']).round(2)

renames = ["TERMINALE_2_Math", "TERMINALE_2_Physics_Chemistry", "TERMINALE_2_Philosophy", "TERMINALE_2_History_Geogoly", "TERMINALE_2_PE", 
          "TERMINALE_2_Domiante",  "TERMINALE_2_Specialite", "BAC_French_Oral",  "BAC_French_Written",  "BAC_Project"]

# Create a new DataFrame to store the results
summary_df = pd.DataFrame({'Subjects': renames, 'Mean Score': mean_values, 'SD Score': std_values, 'Correlation to Rank': correlation_values})


# In[29]:


summary_df


# In[30]:


plt.figure(figsize=(8, 8))
sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.5)

# Add a title
plt.title('Correlation Heatmap')

# Show the plot
plt.savefig("plots/variable_exploration.pdf")
plt.show()


# In[ ]:


grouped = df.groupby('Dominante')['Rank'].mean()

# Create a bar plot
grouped.plot(kind='bar', ylabel='Mean Ranking', title='Mean Ranking by Dominante')
plt.show()


# In[ ]:


mean_values = df.groupby('Dominante')['Rank'].mean().reset_index()

# Create a box plot
plt.figure(figsize=(8, 6))  # Optional: set the figure size
sns.boxplot(x='Dominante', y='Rank', data=df)

# Add mean values to the plot as points
#sns.stripplot(x='Dominante', y='Rank', data=mean_values, color='red', size=8, jitter=True)

plt.title('Box Plot with Mean by Dominante')
plt.xlabel('Dominante')
plt.ylabel('Ranking')

# Show the plot
plt.xticks(rotation=45)  # Optional: rotate x-axis labels if needed
plt.show()



# In[ ]:


mean_values = df.groupby('Specialite')['Rank'].mean().reset_index()

# Create a box plot
plt.figure(figsize=(8, 6))  # Optional: set the figure size
sns.boxplot(x='Specialite', y='Rank', data=df)

# Add mean values to the plot as points
#sns.stripplot(x='Dominante', y='Rank', data=mean_values, color='red', size=8, jitter=True)

plt.title('Box Plot with Mean by Specialite')
plt.xlabel('Specialite')
plt.ylabel('Ranking')

# Show the plot
plt.xticks(rotation=45)  # Optional: rotate x-axis labels if needed
plt.show()


# # Model

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


df_clean = df[['TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
                   'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2',
                   'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.', "label"]]


# In[33]:


df_clean_more_features = df[['MINEURE_BiologiePhysiqueChimie', 'MINEURE_Droit',
       'MINEURE_EconomieGestion', 'MINEURE_SanteDesPopulations', 
        'Boursierdeslycees',  'Dominante','Specialite',
       'PrivateHighSchool', 'Departementetablissement', 
       'Option.internationale', 'RepeatGrade',
       'FICHE.AVENIR_Methode.de.travail', 'FICHE.AVENIR_Autonomie', 'FICHE.AVENIR_Capacite.a.s.investir',
       'FICHE.AVENIR_Avis.sur.la.capacite.a.reussir',
       'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
       'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.',
       'TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Philosophie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2',
       'PREMIERE_Moyenne.candidat.en.Mathematiques.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Physique.Chimie.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Francais.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Mathematiques.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Francais.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Mathematiques.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Physique.Chimie.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Langue.vivante.2.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Francais.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.3.1',
       'PREMIERE_Moyenne.candidat.en.Dominante.Trimestre.1.1',
       'PREMIERE_Moyenne.candidat.en.Dominante.Trimestre.2.1',
       'PREMIERE_Moyenne.candidat.en.Dominante.Trimestre.3.1',
       'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.1',
       'TERMINALE_Moyenne.candidat.en.Specialite.Trimestre.2', "label"]]


# In[34]:


df_clean_more_features['Dominante'].unique()


# In[35]:


Dominante_mapping = {
    'Sciences de la Vie et de la Terre': 3,
    "Sciences de l'ingenieur": 2,
    'Ecologie, Agronomie et Territoires': 1
}
df_clean_more_features['Dominante'] = df_clean_more_features['Dominante'].map(Dominante_mapping)


# In[36]:


df_clean_more_features['Dominante'].unique()


# In[37]:


df_clean_more_features['Specialite'].unique()


# In[38]:


Specialite_mapping = {
    'Mathematiques': 6,
    "Physique-chimie": 5,
    "Sciences de la vie et de la terre": 4,
    "Informatique et Sciences du numerique": 3,
    "Pas de specialite (S.I)": 1,
    "Ecologie agronomie et territoires": 2
}
df_clean_more_features['Specialite'] = df_clean_more_features['Specialite'].map(Specialite_mapping)


# In[39]:


df_clean_more_features['Specialite'].unique()


# In[40]:


missing_values = df_clean_more_features.isnull().sum()
pd.DataFrame(missing_values)


# In[41]:


df_clean_1 = df_clean.dropna()
df_clean_1.shape


# In[42]:


from sklearn.impute import SimpleImputer
from sklearn.utils import resample


# In[43]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

class RandomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_trans = X.copy()
        for column in X_trans.columns:
            missing = X_trans[column].isnull()
            if missing.any():
                non_missing = X_trans.loc[~missing, column]
                random_sample = resample(non_missing, n_samples=missing.sum(), random_state=42)
                X_trans.loc[missing, column] = random_sample.values
        return X_trans


# In[44]:


numerical_cols = df_clean_more_features.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy = "median")  # or 'median'
df_clean_more_features[numerical_cols] = num_imputer.fit_transform(df_clean_more_features[numerical_cols])


# In[45]:


df_clean_2 = df_clean_more_features #df_clean_more_features.dropna()
df_clean_2.shape


# In[46]:


# Assuming you have a DataFrame named df and you want to split it into training, evaluation, and test sets
# Define the features (X) and the target (y)
X = df_clean_2.drop(columns=['label'])  # Replace 'target_column' with the name of your target column
y = df_clean_2['label']

# Split the data into training (70%), evaluation (15%), and test (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
len(X_train.columns)


# In[47]:


X = df_clean_2.drop(columns=['label'])  # Replace 'target_column' with the name of your target column
y = df_clean_2['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[48]:


len(X_train.columns)


# In[49]:


from sklearn.model_selection import cross_val_score, KFold


# In[50]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[51]:


# Replace 'accuracy' with your desired metric


# In[52]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ### Base Model (logistic regression)

# In[53]:


def cross_validation_confusion(X, y, model): 
    conf_matrices = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        predictions = clf_lm.predict(X_test)
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrices.append(conf_matrix)
    print(conf_matrices)
    return conf_matrices
    


# In[54]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


# In[55]:


def cross_validation(k_fold,clf,X,y):
    accuracy_list = []
    precision_list = []
    recall_list = []
    
    for train, test in k_fold.split(X):
        X_train, X_test = X.iloc[train], X.iloc[test] 
        y_train, y_test = y.iloc[train], y.iloc[test] 
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, average='macro'))
        recall_list.append(recall_score(y_test, y_pred, average='macro'))

    result = [np.mean(score) for score in [accuracy_list, precision_list, recall_list]]
      
    for score in [accuracy_list,precision_list,recall_list]:
        result.append(np.mean(score))
    return result


# In[56]:


def train_models(train_X, train_y):
    k_fold = KFold(n_splits=5,shuffle = True)
    results = []
    models = []
    models.append(LogisticRegression( solver = "newton-cg", random_state=42, max_iter=1000))
    models.append(DecisionTreeClassifier(random_state=42))
    models.append(AdaBoostClassifier(random_state=42))
    models.append(RandomForestClassifier(random_state=42))
    models_name = ['LogisticRegression','Decision Tree','Ada Boost','Random Forest']
    
    for i in range(len(models)):
        print(i)
        lis = cross_validation(k_fold,models[i],train_X, train_y)
        results.append([models_name[i],lis[0],lis[1],lis[2]])
    results_df = pd.DataFrame(results)
    results_df.columns = ['MODEL', 'Accuracy','Precision','Recall']
    return results_df
    


# In[57]:


models = train_models(X_train, y_train)


# In[58]:


models


# In[59]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def cross_validation_add(k_fold, clf, X, y):
    all_y_true = []
    all_y_pred = []
    
    for train, test in k_fold.split(X):
        # Handle DataFrame or numpy array
        X_train, X_test = (X.iloc[train], X.iloc[test]) 
        y_train, y_test = (y.iloc[train], y.iloc[test]) 

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Compute metrics
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred, average='macro')
    recall = recall_score(all_y_true, all_y_pred, average='macro')

    return conf_matrix, accuracy, precision, recall


# In[60]:


clf_lm = LogisticRegression(solver = "newton-cg", random_state=42, max_iter=1000)
# Train the classifier on the training data
#clf_lm.fit(X_train, y_train)
cm, accuracy, precision, recall = cross_validation_add(kf, clf_lm, X_train, y_train)


# In[61]:


classes = [1, 2, 3, 4, 5]
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (all features)')
plt.show()


# In[62]:


clf_rf = RandomForestClassifier(random_state=42)
# Train the classifier on the training data
clf_rf.fit(X_train, y_train)

y_pred = clf_rf.predict(X_eval)
accuracy = accuracy_score(y_eval, y_pred)
confusion_mat = confusion_matrix(y_eval, y_pred)
classification_rep = classification_report(y_eval, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)


# In[63]:


from sklearn.naive_bayes import GaussianNB


# # Stepwise regression

# In[64]:


import statsmodels.api as sm


# In[65]:


X = df_clean_2.drop(columns=['label'])  # Replace 'target_column' with the name of your target column
y = df_clean_2['label']


# In[66]:


def forward_selection(data, target, significance_level=0.01):
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            
            model = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_features.append(new_pval.idxmin())
        else:
            break
    return best_features

selected_features = forward_selection(X, y)


# In[67]:


model_swl = sm.OLS(y, sm.add_constant(X[selected_features])).fit()


# In[68]:


print(model_swl.summary())


# In[69]:


mean_values = df.groupby('Boursierdeslycees')['Rank'].mean().reset_index()

# Create a box plot
plt.figure(figsize=(8, 6))  # Optional: set the figure size
sns.boxplot(x='Boursierdeslycees', y='Rank', data=df)

# Add mean values to the plot as points
#sns.stripplot(x='Dominante', y='Rank', data=mean_values, color='red', size=8, jitter=True)

plt.title('Box Plot with Mean by Dominante')
plt.xlabel('Boursierdeslycees')
plt.ylabel('Ranking')

# Show the plot
plt.xticks(rotation=45)  # Optional: rotate x-axis labels if needed
plt.show()


# In[70]:


value_counts = df['Boursierdeslycees'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is your DataFrame
# df = pd.DataFrame({'A': [category data], 'B': [value data]})

# Plotting the distribution
sns.displot(data=df, x='Rank', hue='Boursierdeslycees', kind='kde')  # You can change kind to 'hist' for histogram

plt.title('Distribution of Ranking based on high school financial aid status')
plt.xlabel('Ranking in Percentile')
plt.ylabel('Density')
plt.xlim(0, 1)
plt.show()


# In[71]:


len(selected_features)


# # Compare

# In[72]:


def train_models_after_sw(train_X, train_y):
    k_fold = KFold(n_splits=5,shuffle = True)
    results = []
    models = []
    models.append(LogisticRegression(solver='newton-cg',random_state=42, max_iter = 1000))
    models.append(DecisionTreeClassifier(random_state=42))
    models.append(AdaBoostClassifier(random_state=42,))
    models.append(RandomForestClassifier(random_state=42))
    models_name = ['LogisticRegression','Decision Tree','Ada Boost','Random Forest']
    
    for i in range(len(models)):
        print(i)
        lis = cross_validation(k_fold,models[i],train_X, train_y)
        results.append([models_name[i],lis[0],lis[1],lis[2]])
    results_df = pd.DataFrame(results)
    results_df.columns = ['MODEL', 'Accuracy','Precision','Recall']
    return results_df


# In[73]:


# Assuming you have a DataFrame named df and you want to split it into training, evaluation, and test sets
# Define the features (X) and the target (y)
X = df_clean_2[selected_features] # Replace 'target_column' with the name of your target column
y = df_clean_2['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[74]:


models_after_sw = train_models_after_sw(X_train, y_train)


# In[75]:


models_after_sw


# In[76]:


clf_lm = LogisticRegression(solver="newton-cg", random_state=42, max_iter=1000)
# Train the classifier on the training data
#clf_lm.fit(X_train, y_train)
cm, accuracy, precision, recall = cross_validation_add(kf, clf_lm, X_train, y_train)
classes = [1, 2, 3, 4, 5]
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (with feature selection)')
plt.show()


# In[77]:


clf_lm = LogisticRegression(solver='newton-cg',random_state=42, max_iter = 1000)

# Train the classifier on the training data
clf_lm.fit(X_train, y_train)
y_pred = clf_lm.predict(X_test)
cm = conf_matrix = confusion_matrix(y_test, y_pred)

#cm, accuracy, precision, recall = cross_validation_add(kf, clf_lm, X_train, y_train)


# In[131]:


coefficients = clf_lm.coef_

# Print variable names with coefficients for each class
num_classes, num_features = coefficients.shape
print("\nVariable Names with Coefficients:")
for class_idx in range(num_classes):
    print(f"\nClass {class_idx} Coefficients:")
    for feature, coef in zip(X.columns, coefficients[class_idx]):
        print(f"{feature}: {coef}")


# In[129]:


coefficients = clf_lm.coef_

# Print variable names with coefficients for each class
num_classes, num_features = coefficients.shape
print("\nVariable Names with Coefficients:")
for class_idx in range(num_classes):
    print(f"\nClass {class_idx} Coefficients:")
    for feature, coef in zip(X.columns, coefficients[class_idx]):
        print(f"{feature}: {coef}")
        
avg_importance = np.mean(np.abs(coefficients), axis=0)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': avg_importance})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))

# Dictionary mapping current variable names to custom labels
custom_labels = {
    'Dominante': 'Dominante (1-3)',
    'RepeatGrade': 'Repeated Grade (0-1)',
    'PrivateHighSchool': 'Private High School (0-1)',
    'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.': 'Baccalauréat French Oral grade (0-20)',
    'Boursierdeslycees': 'Financial Aid (0-1)',
    'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2': 'Physics-Chemistry grade (last year, 2nd quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.1': 'Physics-Chemistry grade (last year, 1st quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2': 'Maths grade (last year, 2nd quarter) (0-20)',
    'Option.internationale': 'International Baccalauréat (0-1)',
    'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.1': 'Dominante grade (last year, 1st quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2': 'Dominante grade (last year, 2nd quarter) (0-20)',
    'Specialite': 'Spécialité (1-6)',
    'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.': 'Baccalauréat French Written grade (0-20)',
    'MINEURE_SanteDesPopulations': 'Population Health minor choice (0-1)',
    'TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.1': 'Maths grade (last year, 1st quarter) (0-20)',
    'PREMIERE_Moyenne.candidat.en.Francais.Trimestre.1.1': 'French grade (penultimate year, 1st quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.2': 'Foreign Language 1 grade (last year, 2nd quarter) (0-20)',
    'PREMIERE_Moyenne.candidat.en.Mathematiques.Trimestre.1.1': 'Maths grade (penultimate year, 1st quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.1': 'Foreign Language 1 grade (last year, 1st quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.1': 'Physical Education (last year, 1st quarter) (0-20)',
    'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.': 'Baccalauréat Group Project grade (0-20)',
    'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.1': 'History-Geography (last year, 1st quarter) (0-20)',
    'PREMIERE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2.1': 'Physics-Chemistry (penultimate year, 2nd quarter) (0-20)',
    'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2': 'History-Geography (last year, 2nd quarter) (0-20)'
}

# Replace variable names with custom labels
feature_importance['Feature'] = feature_importance['Feature'].map(custom_labels).fillna(feature_importance['Feature'])

# Plotting with customized labels
plt.figure(figsize=(8, 8))
bar_plot = feature_importance.plot(x='Feature', y='Importance', kind='barh', color='black', legend=False)
ax.set_ylabel('Importance')

# Show the plot
plt.show()


# In[668]:


classes = [1, 2, 3, 4, 5]
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[135]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Assuming you have the coefficients and feature names
coefficients = clf_lm.coef_
feature_names = X.columns

# Create a custom order for features
custom_feature_order = [
    'Dominante', 
    'RepeatGrade', 
    'PrivateHighSchool',
    'BAC_Note.a.l.epreuve.de.Oral.de.Francais..epreuve.anticipee.', 
    'Boursierdeslycees',
    'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2', 
    'TERMINALE_Moyenne.candidat.en.Physique.Chimie.Trimestre.1', 
    'TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.2',
    'Option.internationale', 
    'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.1', 
    'TERMINALE_Moyenne.candidat.en.Dominante.Trimestre.2', 
    'Specialite',
    'BAC_Note.a.l.epreuve.de.Ecrit.de.Francais..epreuve.anticipee.',
    'MINEURE_SanteDesPopulations', 
    'TERMINALE_Moyenne.candidat.en.Mathematiques.Trimestre.1', 
    'PREMIERE_Moyenne.candidat.en.Francais.Trimestre.1.1', 
    'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.2', 
    'PREMIERE_Moyenne.candidat.en.Mathematiques.Trimestre.1.1', 
    'TERMINALE_Moyenne.candidat.en.Langue.vivante.1.Trimestre.1',
    'TERMINALE_Moyenne.candidat.en.Education.Physique.et.Sportive.Trimestre.1', 
    'BAC_Note.a.l.epreuve.de.Travaux.Personnels.Encadres..epreuve.anticipee.', 
    'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.1',
    'PREMIERE_Moyenne.candidat.en.Physique.Chimie.Trimestre.2.1', 
    'TERMINALE_Moyenne.candidat.en.Histoire.Geographie.Trimestre.2'
]

# Reorder the coefficients based on custom order
ordered_coefficients = coefficients[:, [feature_names.get_loc(feature) for feature in custom_feature_order]]

# Create a custom colormap
cmap = ListedColormap(['red', 'green'])

# Create a heatmap without displaying coefficients
plt.figure(figsize=(2, 6))
sns.heatmap(ordered_coefficients.transpose(),
            cmap=cmap, cbar=False, annot=False, vmin=-np.max(np.abs(coefficients)), vmax=np.max(np.abs(coefficients)))

# Customize x-axis labels
plt.xticks(np.arange(coefficients.shape[0]) + 0.5, np.arange(1, coefficients.shape[0] + 1))
plt.xlabel('Classes')

# Customize y-axis labels
plt.yticks(np.arange(len(custom_feature_order)), custom_feature_order, rotation=0)
plt.ylabel('Features')

# Set plot title
plt.title('Coefficients for Each Feature and Class (Red: Negative, Green: Positive)')

# Show the plot
plt.show()

