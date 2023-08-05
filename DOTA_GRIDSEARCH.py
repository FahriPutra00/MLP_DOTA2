# %% [markdown]
# **FAHRI** **PUTRA** **HERLAMBANG** - **5200411389**

# %% [markdown]
# Pada tugas kali ini diberikan perintah untuk mereplikasi dan memperbaiki akurasi dari Model MLP yang sudah ada, dengan menggunakan dataset Bank Markering dataset.

# %% [markdown]
# # **1. IMPORT LIBRARY**

# %% [markdown]
# Pertama kita melakukan import yang diperlukan dalam melakukan membaca data, mendeskripsikan data, membuat model, dan melakukan test model

# %%
# library umum
import numpy as np 
import pandas as pd
from typing import Literal
# library sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from scipy.stats.mstats import winsorize

# library menampilkan data
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # **2. DATA UNDERSTANDING**

# %% [markdown]
# ## **2.1. IMPORT DATA**

# %% [markdown]
# Melakukan import dang diperlukan untuk membuat model, data ini di dapat dari website https://archive.ics.uci.edu/ml/datasets/bank+marketing# 

# %% [markdown]
# melakukan koneksi antara google drive dengan file collab

# %%
# # mounting google drive to Colab Runtime environment. Koneksi Ke G.Drive
# from google.colab import drive
# drive.mount("/content/gdrive")

# %% [markdown]
# membuka dataset yang telah dihubungkan dengan googgle drive atau path penyimpanan dataset tersebut

# %%
# loading dataset
# in this dataset, the dataset_dota2 are separated using ';' symbol. Therefore, when reading the CSV, 
# we should instruct the Pandas DataFrame about the separater. This is because the default separater is the ',' 
# and that if we do not specify, the DataFrame will have all the row dataset_dota2 into one cell.

# COLLAB
# dataset_dota2 = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/Dataset/dota2-full.csv',sep=';')

# VSC
dataset_dota2_train = pd.read_csv('D:\\CODING\\SMT5\\NNDL\\DATASET\\dota2Dataset\\dota2Train.csv',encoding= 'unicode_escape',header=None)
dataset_dota2_test = pd.read_csv('D:\\CODING\\SMT5\\NNDL\\DATASET\\dota2Dataset\\dota2Test.csv',encoding= 'unicode_escape',header=None)
# getting a glimpse of the dataset_dota2

# %%
# Load the heroes_names into a DataFrame
heroes_names = pd.read_json('https://raw.githubusercontent.com/kronusme/dota2-api/bbc512369ccb1490804d0e4ed13f5b58f20ca004/data/heroes.json')

# Rename the column
heroes_name = [heroes_names['heroes'][i]['localized_name'] for i in range(0,len(heroes_names['heroes']))]
heroes_id = [heroes_names['heroes'][i]['id'] for i in range(0,len(heroes_names['heroes']))]
heroes_names = pd.DataFrame(heroes_name, columns=['heroes_name'], index=heroes_id)
# heroes_names.sort_values('id', inplace=True)
        


# %%
lobbies_names = pd.read_json('https://raw.githubusercontent.com/kronusme/dota2-api/master/data/lobbies.json')

lobbies_name = [lobbies_names['lobbies'][i]['name'] for i in range(0,len(lobbies_names['lobbies']))]
lobbies_id = [lobbies_names['lobbies'][i]['id'] for i in range(0,len(lobbies_names['lobbies']))]
lobbies_names = pd.DataFrame(lobbies_name, columns=['lobbies_name'], index=lobbies_id)
# lobbies_names.sort_values('id', inplace=True)
        


# %%
mods_names = pd.read_json('https://raw.githubusercontent.com/kronusme/dota2-api/master/data/mods.json')

mods_name = [mods_names['mods'][i]['name'] for i in range(0,len(mods_names['mods']))]
mods_id = [mods_names['mods'][i]['id'] for i in range(0,len(mods_names['mods']))]
mods_names = pd.DataFrame(mods_name, columns=['mods_name'], index=mods_id)
# mods_names.sort_values('id', inplace=True)
        


# %%
regions_names = pd.read_json('https://raw.githubusercontent.com/kronusme/dota2-api/master/data/regions.json')

regions_name = [regions_names['regions'][i]['name'] for i in range(0,len(regions_names['regions']))]
regions_id = [regions_names['regions'][i]['id'] for i in range(0,len(regions_names['regions']))]
regions_names = pd.DataFrame(regions_name, columns=['regions_name'], index=regions_id)
# regions_names.sort_values('id', inplace=True)
        

# %%
# list heroes https://github.com/Ayub-Khan/Dota-2-Hero-Suggester/blob/master/heroes.txt
data_hero = open('D:/CODING/SMT5/NNDL/DATASET/68cc46b435a44898b35bee383bd69f9b-b157c3a655aeb8f3fbf8a33358c69290cff274b8/ListHero.txt', 'r')
data_hero = [line.strip() for line in data_hero] #WASP = IO, LYCAN = LYCANTROPY, UNDERLORD = ABBYSAL UNDERLORD, SKELETON KING = WRAITH KING,

# MENGECEK HERO YANG TIDAK ADA DI DATASET
for i in data_hero:
    if i not in heroes_names['heroes_name'].values:
        print(i)
print('JUMLAH DATA HERO YANG ADA',len(data_hero))
# MONKEY KING TIDAK ADA KARNA BELUM DI RELEASE ID MONKEY KING 114
#DOTA 2 API TERBARU https://github.com/leamare/D2-LRG-Metadata/blob/master/heroes.json


# %%
for i in range(1,113):
    if i not in heroes_names.index:
        print(i)

#DOTA 2 API TERBARU https://github.com/leamare/D2-LRG-Metadata/blob/master/heroes.json
#ID HERO 24 MEMANG TIDAK ADA

# %%
# for i in range(1,113):
#     dataset_dota2_train.rename(columns={i:heroes_names['name'][i-1]}, inplace=True)
#     dataset_dota2_test.rename(columns={i:heroes_names['name'][i-1]}, inplace=True)


# %%
dataset_dota2_train.rename(columns = {0:'TeamWIN',1:'ClusterID',2:'GameMode',3:'GameType'}, inplace = True)
dataset_dota2_train

# %%
Stat_game = dataset_dota2_train[["TeamWIN","ClusterID","GameMode","GameType"]]
Heroes_pick = dataset_dota2_train.drop(["TeamWIN","ClusterID","GameMode","GameType"], axis=1)
Heroes_pick.columns = range(1,Heroes_pick.columns.size+1)
dataset_dota2_train = pd.concat([Stat_game, Heroes_pick], axis=1)
dataset_dota2_train.drop(24, axis=1, inplace=True)
for i in heroes_names.index:
    dataset_dota2_train.rename(columns={i:heroes_names['heroes_name'][i]}, inplace=True)
for i in lobbies_names.index:
    dataset_dota2_train["GameMode"] = dataset_dota2_train["GameMode"].replace({i:lobbies_names['lobbies_name'][i]})
for i in mods_names.index:
    dataset_dota2_train["GameType"] = dataset_dota2_train["GameType"].replace({i:mods_names['mods_name'][i]})
for i in regions_names.index:
    dataset_dota2_train["ClusterID"] = dataset_dota2_train["ClusterID"].replace({i:regions_names['regions_name'][i]})
dataset_dota2_train



# %%
dataset_dota2_test.rename(columns = {0:'TeamWIN',1:'ClusterID',2:'GameMode',3:'GameType'}, inplace = True)
dataset_dota2_test

# %%
Stat_game = dataset_dota2_test[["TeamWIN","ClusterID","GameMode","GameType"]]
Heroes_pick = dataset_dota2_test.drop(["TeamWIN","ClusterID","GameMode","GameType"], axis=1)
Heroes_pick.columns = range(1,Heroes_pick.columns.size+1)
dataset_dota2_test = pd.concat([Stat_game, Heroes_pick], axis=1)
dataset_dota2_test.drop(24, axis=1, inplace=True)
for i in heroes_names.index:
    dataset_dota2_test.rename(columns={i:heroes_names['heroes_name'][i]}, inplace=True)
for i in lobbies_names.index:
    dataset_dota2_test["GameMode"] = dataset_dota2_test["GameMode"].replace({i:lobbies_names['lobbies_name'][i]})
for i in mods_names.index:
    dataset_dota2_test["GameType"] = dataset_dota2_test["GameType"].replace({i:mods_names['mods_name'][i]})
for i in regions_names.index:
    dataset_dota2_test["ClusterID"] = dataset_dota2_test["ClusterID"].replace({i:regions_names['regions_name'][i]})
dataset_dota2_test



# %% [markdown]
# ## **2.2. DESKRIPSI DATA**

# %% [markdown]
# melihat isi setiap kolom yang ada pada dataset tersebut

# %%
dataset_dota2_train.dropna( axis=0, how='any', thresh=None, subset=None, inplace=False)

# %%
dataset_dota2_test.dropna( axis=0, how='any', thresh=None, subset=None, inplace=False)

# %% [markdown]
# melihat data kolom target atau 'y' lalu menampilkan dalam bentuk histogram

# %%
dataset_dota2_train['TeamWIN'].value_counts().plot(kind='bar')
dataset_dota2_train['TeamWIN'].value_counts()

# %% [markdown]
# Melihat deskripsi data 

# %%
#deskripsi data
dataset_dota2_train.describe()

# %%
#deskripsi data
dataset_dota2_test.describe()

# %%
dataset_dota2_train.loc[:, 'ClusterID':'GameMode'].describe(include=['O'])

# %%
dataset_dota2_train.pivot_table('TeamWIN', 
                     index='ClusterID', 
                     columns=['GameMode', 'GameType'], 
                     aggfunc='count',
                     fill_value=0)

# %%
htrain_df = dataset_dota2_train.iloc[:, 4:]
htrain_df

# %% [markdown]
# Melihat Info data

# %%
dataset_dota2_test.dtypes

# %%
dataset_dota2_train.dtypes

# %%
#melihat info data
dataset_dota2_train.info()

# %%
#melihat info data
dataset_dota2_test.info()

# %%
print(dataset_dota2_train.isnull().sum())

# %% [markdown]
# ## **2.3. PLOTTING DATA**

# %% [markdown]
# Melakukan Plotting Untuk Melihat Bentuk data, jumlah data, dan sebaran data

# %% [markdown]
# ### **2.3.2. PLOTTING JUMLAH DATA**

# %%
# #plotting jumlah data
# dataset_dota2_train.hist(alpha=0.5, figsize=(15, 15), color='red')
# plt.show()

# %%
# #plotting jumlah data
# dataset_dota2_test.hist(alpha=0.5, figsize=(15, 15), color='red')
# plt.show()

# %% [markdown]
# ### **2.3.3. MELIHAT SEBARAN DATASET**

# %%
# #MELIHAT SEBARAN DATASET
# sns.pairplot(data=dataset_dota2_train,diag_kind='hist')
# plt.show()

# %%
# #MELIHAT SEBARAN DATASET
# sns.pairplot(data=dataset_dota2_test,diag_kind='hist')
# plt.show()

# %% [markdown]
# ## **2.4. PREPROCESSING**

# %% [markdown]
# ### **2.4.1. CHECK DUPLICATED**

# %% [markdown]
# Untuk mengecheck duplikasi data dan menghapusnya

# %%
# Check for duplicate rows.
print(f"There are {dataset_dota2_train.duplicated().sum()} duplicate rows in the dataset_dota2_train set.")

# Remove duplicate rows.
dataset_dota2_train=dataset_dota2_train.drop_duplicates()
print("The duplicate rows were removed.")

# %%
# Check for duplicate rows.
print(f"There are {dataset_dota2_test.duplicated().sum()} duplicate rows in the dataset_dota2_test set.")

# Remove duplicate rows.
dataset_dota2_test=dataset_dota2_test.drop_duplicates()
print("The duplicate rows were removed.")

# %% [markdown]
# Mengurutkan kembali kolom, sehingga y atau target berada di paling belakang

# %%
# # rearrange the columns in the dataset to contain the TeamWIN (target/label) at the end
# cols = list(dataset_dota2_train.columns.values)
# cols.pop(cols.index('TeamWIN')) # pop TeamWIN out of the list
# dataset_dota2_train = dataset_dota2_train[cols+['TeamWIN']] #Create new dataframe with columns in new 
# 

# %%
# # rearrange the columns in the dataset to contain the TeamWIN (target/label) at the end
# cols = list(dataset_dota2_test.columns.values)
# cols.pop(cols.index('TeamWIN')) # pop TeamWIN out of the list
# dataset_dota2_test = dataset_dota2_test[cols+['TeamWIN']] #Create new dataframe with columns in new 
# 

# %% [markdown]
# # **3. IMPLEMENTASI MLP**

# %% [markdown]
# ## **3.1. PEMBAGIAN DATA**

# %%
def prepare_train_test_data(train_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            target: str,
                            features: list,
                            encoding_method: Literal['one_hot_encoding', 'label_encoding'] = 'label_encoding',
                            encode_target=False):
        """
        Prepearing training and testing data by splitting dataframes to the train and test one.
        Also performing features and target encoding by either One Hot Encoding or Lable Encoding. 
        """    

        X_train = train_df.drop(target, axis=1)
        X_test = test_df.drop(target, axis=1)

        y_train = train_df[[target]]
        y_test = test_df[[target]]
    
        if encode_target:
            y_train, y_test, y_encoders = labels_encoding(y_train, y_test, [target])
    
        y_train = np.array(y_train).ravel()
        y_test = np.array(y_test).ravel()
    
        if encoding_method == 'label_encoding':
            X_train, X_test, X_encoders = labels_encoding(X_train, X_test, features)
            return X_train, X_test, y_train, y_test

        elif encoding_method == 'one_hot_encoding':
            X_train, X_test = one_hot_encoding(X_train, X_test, features)
            return X_train, X_test, y_train, y_test

        else:
            return X_train, X_test, y_train, y_test
            
def labels_encoding(train_data, test_data, features):
    """
    Performs labels encoding for the given training and testing data
    """
    
    label_encoders = {}

    for feature in features:
        label_encoders['train_' + feature] = preprocessing.LabelEncoder()
        label_encoders['test_' + feature] = preprocessing.LabelEncoder()

        label_encoders['train_' + feature].fit(train_data[[feature]])
        label_encoders['test_' + feature].fit(test_data[[feature]])

        train_data[feature] = label_encoders['train_' + feature].transform(train_data[[feature]])
        test_data[feature] = label_encoders['test_' + feature].transform(test_data[[feature]])

    return train_data, test_data, label_encoders


def one_hot_encoding(train_data, test_data, features):
    """
    Performs one hot encoding for the given training and testing data 
    """
    
    train_data_ohe_features = pd.get_dummies(train_data[[*features]])
    test_data_features = pd.get_dummies(test_data[[*features]])
    
    train_data.drop([*features], axis=1, inplace=True)
    test_data.drop([*features], axis=1, inplace=True)

    train_data = pd.concat((train_data, train_data_ohe_features), 1)
    test_data = pd.concat((test_data, test_data_features), 1)

    return train_data, test_data

def show_model_accuracy(X_train, X_test, y_train, y_test):
    print('====================== Accuracy =======================')
    print(f'Training data:\t{mlp.score(X_train, y_train) * 100:.2f} %')
    print(f'Testing data:\t{mlp.score(X_test, y_test) * 100 :.2f} %')
    print('=======================================================')    

# %% [markdown]
# membagi dat untuk dibuat ke dalam X_train, X_test, y_train, y_test

# %% [markdown]
# ## **3.2. MEMBUAT MODEL dan TRAINING MODEL**

# %% [markdown]
# proses pembuatan model mlp yang telah di uji untuk mendapatkan akurasi terbaik

# %%
## The Winning Team Prediction (Multi Layer Perceptron)

#### Prepearing data with one hot encoding

X_train, X_test, y_train, y_test = prepare_train_test_data(
    dataset_dota2_train,
    dataset_dota2_test,
    target='TeamWIN',
    features=['GameType', 'ClusterID', 'GameMode'],
    encoding_method='one_hot_encoding',
    encode_target=True
)

print('Data shape')
print(f"X_train: {X_train.shape} X_test: {y_train.shape}")
print(f"t_train: {X_test.shape} y_test: {y_test.shape}")


parameter_space = {
    'hidden_layer_sizes': [(128, 64, 32),(100,200,100)],
    'activation': ['identity','tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'max_iter': [100],
    'learning_rate_init': [0.001, 0.01],
}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(MLPClassifier(), parameter_space, refit = True, verbose = 3,n_jobs=-1)
  
# fitting the model for clf search
clf.fit(X_train, y_train)
# print best parameter after tuning
print(clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(clf.best_estimator_)

print('Best parameters found:\n', clf.best_params_)
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
y_true, y_pred = y_test , clf.predict(X_test)
from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))





