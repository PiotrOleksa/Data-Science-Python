import pandas as pd
from pandas.plotting import table
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA




#---------------------- Import the data
df = pd.read_csv('ks-projects-2018.csv', sep=',')



#---------------------- Data Exploring
#Drop unnecessary columns
df = df.drop(['ID', 'name', 'currency', 'goal', 'pledged'], axis=1)
df = df[df.state != 'undefined']

#Succcessul projects
suc = df[df.org_state == 'successful']

#Replace nan values
df['usd pledged'] = df['usd pledged'].fillna((df['usd pledged'].mean()))

#Describe
pd.set_option('float_format', '{:.2f}'.format)
sub_plot = plt.subplot(111, frame_on=False)

sub_plot.xaxis.set_visible(False) 
sub_plot.yaxis.set_visible(False) 

table(sub_plot, df.describe(), loc='upper right').scale(1, 2)
plt.title("Kickstarter Projects Describe")
plt.savefig('kick_inf.png', dpi=199)

#Number of succesfull projects
df['state'].value_counts(normalize = True).plot.bar(rot=0)
plt.title("State")
plt.savefig('State', dpi=199)

#Categories successful
cleanup_nums = {"state":{"failed": 0, "canceled": 0, "suspended" :0, "successful" : 1, "live" : 1}}
df["org_state"] = df["state"]
df.replace(cleanup_nums, inplace=True)



suc = df[df.org_state == 'successful']

a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x=suc.main_category.value_counts().index, y=suc.main_category.value_counts())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title("Categories")
plt.savefig('SCategories', dpi=199)

#Backers
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x="main_category", y="backers", hue="state", data=df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.title("Backers")
plt.savefig('backers', dpi=199)


#Remove undefined
df = df[df.state != 'undefined']

#Converting non-numeric data to numeric
df["deadline"] = pd.to_datetime(df["deadline"]).dt.strftime("%Y%m%d")
df["launched"] = pd.to_datetime(df["launched"]).dt.strftime("%Y%m%d")
        

#Create ne column 'days'
df['days'] = np.busday_count(pd.to_datetime(df['launched']).values.astype('datetime64[D]'),\
                             pd.to_datetime(df['deadline']).values.astype('datetime64[D]'))

#Correlation
plt.figure(figsize=(15, 3))
ax = plt.subplot(111, frame_on=False) 
ax.xaxis.set_visible(False)  
ax.yaxis.set_visible(False)  
table(ax, df.corr(), loc='upper right').scale(1, 2)
plt.title("Kickstarter Projects Correlations")
plt.savefig('corr.png', dpi=199)
 



columns = ['category', 'main_category', 'country']

for column in columns:
    num_cat = np.arange(len(df[column].unique()))
    for x, i in zip(df[column].unique(), num_cat):
        df[column] = df[column].replace({f'{x}': i})
        
#Principal Component Analys
df_clust = df[['state','backers','days','main_category','usd_pledged_real']]
z = df_clust.iloc[:,0:5]

#Building a model
model = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
    )

model.fit(z)

df_clust['clusters'] = model.labels_
df.clust.info()









   
   


        
