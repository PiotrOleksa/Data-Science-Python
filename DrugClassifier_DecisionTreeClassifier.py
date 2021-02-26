import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics




#---------------------- Import the data
df = pd.read_csv(r'C:\Users\Piotr\Desktop\Projekty Python\5. Decision Tree\drug200.csv', sep=',')


#---------------------- Data Exploring
#Describe
pd.set_option('float_format', '{:.2f}'.format)
sub_plot = plt.subplot(111, frame_on=False)

sub_plot.xaxis.set_visible(False) 
sub_plot.yaxis.set_visible(False) 

table(sub_plot, df.describe(), loc='upper right').scale(1, 2)
plt.title("Patients Describe")
plt.savefig('pat_desc.png', dpi=199)

#Pie charts
#Sex
fig1, ax1 = plt.subplots()
ax1.pie(df['Sex'].value_counts(), labels=['Males', 'Females'],autopct='%1.1f%%',startangle=90)
ax1.axis('equal')
plt.title('Sex Proportions')
plt.savefig('sx_pie.png', dpi=199)

#BP
fig1, ax1 = plt.subplots()
ax1.pie(df['BP'].value_counts(), labels=df['BP'].unique(),autopct='%1.1f%%',startangle=90)
ax1.axis('equal')
plt.title('BP')
plt.savefig('bp_pie.png', dpi=199)

#Cholesterol
fig1, ax1 = plt.subplots()
ax1.pie(df['Cholesterol'].value_counts(), labels=df['Cholesterol'].unique(),autopct='%1.1f%%',startangle=90)
ax1.axis('equal')
plt.title('Chorestelor')
plt.savefig('ch_pie.png', dpi=199)

#Drug
drug_count = pd.value_counts(df['Drug'].values, sort=True)
drug_count.plot.bar(rot=0)
plt.title('Drugs')
plt.savefig('drug bar.png', dpi=199)



#---------------------- Import the data
#Features
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

pre_sex = preprocessing.LabelEncoder()
pre_sex.fit(['F', 'M'])
X[:,1] = pre_sex.transform(X[:,1])

pre_BP = preprocessing.LabelEncoder()
pre_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = pre_BP.transform(X[:,2])

pre_chol = preprocessing.LabelEncoder()
pre_chol.fit(['NORMAL', 'HIGH'])
X[:,3] = pre_chol.transform(X[:,3])

#Lables
y = df['Drug']

#Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)


#---------------------- Creating a model
model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
model.fit(X_train, y_train)
pred = model.predict(X_test)

acc = metrics.accuracy_score(y_test, pred)


