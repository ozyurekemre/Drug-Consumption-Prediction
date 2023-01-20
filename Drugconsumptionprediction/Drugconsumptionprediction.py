import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

######################################
### ANALYSIS ###
######################################

######################################
# EDA
######################################
df = pd.read_csv("datasets/Drug_Consumption_Quantified.csv",encoding="latin-1")
df.shape
df_copy = df.copy()
######################################
# 1.Genel Resim
######################################
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

######################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)
######################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)
######################################
# 4. Korelasyon Analizi (Analysis of Correlation)
######################################

df[df["SEMER"] != "CL0"] # overclaimers
df = df.drop(df[df["SEMER"] != 'CL0'].index)
df.drop("SEMER",axis=1,inplace = True)

df.drop('ID',axis=1,inplace=True)
df = df.reset_index(drop=True)

drugs = ['Alcohol',
         'Amyl',
         'Amphet',
         'Benzos',
         'Caff',"Choc",
         'Cannabis',
         'Coke',
         'Crack',
         'Ecstasy',
         'Heroin',
         'Ketamine',
         'Legalh',
         'LSD',
         'Meth',
         'Mushrooms',
         'Nicotine',
         'VSA']

def drug_encoder(x):
    if x == 'CL0':
        return 0
    elif x == 'CL1':
        return 1
    elif x == 'CL2':
        return 2
    elif x == 'CL3':
        return 3
    elif x == 'CL4':
        return 4
    elif x == 'CL4':
        return 5
    elif x == 'CL5':
        return 6
    else:
        return 7

for column in drugs:
    df[column] = df[column].apply(drug_encoder)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 8}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

correlation_matrix(df,drugs)
correlation_matrix(df,df.columns)
correlation_matrix(df,num_cols)

######################################
#Feature Engineering#
######################################

######################################
# Aykırı Değer Analizi
######################################

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

######################################
# Eksik Değer Analizi
######################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
###########################
### Grafikler ###
###########################
df = pd.read_csv("Drug_Consumption.csv")

#'Drugs vs Age'
plt.figure(figsize=(20,15))
sns.lineplot(df.Age, df.Alcohol, label='Alcohol')
sns.lineplot(df.Age, df.Amphet, label='Amphet')
sns.lineplot(df.Age, df.Amyl, label='Amyl')
sns.lineplot(df.Age, df.Benzos, label='Benzos')
sns.lineplot(df.Age, df.Caff, label='Caff')
sns.lineplot(df.Age, df.Cannabis, label='Cannabis')
sns.lineplot(df.Age, df.Choc, label='Choc')
sns.lineplot(df.Age, df.Coke, label='Coke')
sns.lineplot(df.Age, df.Crack, label='Crack')
sns.lineplot(df.Age, df.Ecstasy, label='Ecstasy')
sns.lineplot(df.Age, df.Heroin, label='Heroin')
sns.lineplot(df.Age, df.Ketamine, label='Ketamine')
sns.lineplot(df.Age, df.Legalh, label='Legalh')
sns.lineplot(df.Age, df.LSD, label='LSD')
sns.lineplot(df.Age, df.Meth, label='Meth')
sns.lineplot(df.Age, df.Mushrooms, label='Mushrooms')
sns.lineplot(df.Age, df.Nicotine, label='Nicotine')
sns.lineplot(df.Age, df.Semer, label='Semer')
sns.lineplot(df.Age, df.VSA, label='VSA')
plt.legend(loc='upper left')
plt.title('Drugs vs Age')
plt.xlabel('Age')
plt.ylabel('Drugs')
plt.show(block = True)


#Drug vs classes
df.head()
columns = ['Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
           'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'VSA']


for column in columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

fig, axes = plt.subplots(5,3,figsize = (14,14))
fig.suptitle("Count of Different Classes Vs Drug",fontsize=14)
k=0
for i in range(5):
    for j in range(3):
        sns.countplot(x=columns[k], data=df,ax=axes[i][j])
        k+=1

plt.tight_layout()
plt.show(block = True)

######### base model test
df["User_Coke"]= df["Coke"].apply(lambda x:1 if x > 0 else 0)
y = df['User_Coke']
X = df.drop(['User_Coke','Coke'], axis=1)
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression(solver='liblinear')),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")
base_models(X, y, scoring="precision")
base_models(X, y, scoring="recall")
base_models(X, y, scoring="f1")
base_models(X, y, scoring="roc_auc")



######################################
### PREDICTION ###
######################################
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("Drug_Consumption.csv")

df.head()
df.groupby("Country")["Country"].count()

df.loc[(df["Education"] == "Left school at 18 years") | (df["Education"] == "Left school at 16 years") | (df["Education"] == 'Left school before 16 years') | (df["Education"] == 'Left school at 17 years'),"NEW_NON_DEGREE"] = 1
df.loc[(df["Education"]=='Some college or university, no certificate or degree'),"NEW_STUDYING"] = 1
df.loc[(df["Education"]=='University degree'),"NEW_UNIVERSITY_DEGREE"] = 1
df.loc[(df["Education"]=='Doctorate degree') | (df["Education"]=='Professional certificate/ diploma') | (df["Education"]=='Masters degree') ,"NEW_HIGHLY_EDUCATED"] = 1
df = df.fillna(0)
df.drop('Education',axis=1,inplace=True)

df.loc[(df['Age'] == "18-24") | (df['Age'] == "25-34"), "NEW_AGE_CAT"] = 0
df.loc[(df['Age'] == "35-44") | (df['Age'] == "45-54"), "NEW_AGE_CAT"] = 1
df.loc[(df['Age'] == "55-64") | (df['Age'] == "65+"), "NEW_AGE_CAT"] = 2

df.drop('Age',axis=1,inplace=True)

df.drop('ID',axis=1,inplace=True)
df = df.reset_index(drop=True)

df[df["Semer"] != "CL0"] # overclaimers
df = df.drop(df[df['Semer'] != 'CL0'].index)
df.drop("Semer",axis=1,inplace = True)

df.head()

cat_cols, cat_but_car, num_cols = grab_col_names(df)

#one_hot_encoder

ohe = ['Country','Ethnicity','Gender']
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe, drop_first=True)

#label_encoder

drugs = ['Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
           'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'VSA']


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in drugs:
    label_encoder(df, col)
df.head()

df.columns = [col.upper() for col in df.columns]

# METH
meth_df = df.copy()
meth_df['METH_USER'] = meth_df['METH'].apply(lambda x: 1 if x not in [0,1] else 0)
meth_df = meth_df.drop(['METH'], axis=1)
#COCAINE
coke_df = df.copy()
coke_df['COKE_USER'] = coke_df['COKE'].apply(lambda x: 1 if x not in [0,1] else 0)
coke_df = coke_df.drop(['COKE'], axis=1)
#ALCOHOL
alcohol_df = df.copy()
alcohol_df['ALCOHOL_USER'] = alcohol_df['ALCOHOL'].apply(lambda x: 1 if x not in [0,1] else 0)
alcohol_df = alcohol_df.drop(['ALCOHOL'], axis=1)
#ECSTASY
ecstasy_df = df.copy()
ecstasy_df['ECSTASY_USER'] = ecstasy_df ['ECSTASY'].apply(lambda x: 1 if x not in [0,1] else 0)
ecstasy_df  = ecstasy_df .drop(['ECSTASY'], axis=1)

#MODELLEME
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score


y = coke_df['COKE_USER']
X = coke_df.drop(['COKE_USER'], axis=1)
###############################################
y = meth_df['METH_USER']
X = meth_df.drop(['METH_USER'], axis=1)
##############################################
#Base Models
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression(solver='liblinear')),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")
base_models(X, y, scoring="precision")
base_models(X, y, scoring="recall")
base_models(X, y, scoring="f1")
base_models(X, y, scoring="roc_auc")

# feature importance
from lightgbm import LGBMRegressor

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block= True)
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)

#Automated Hyperparameter Optimization
knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier( eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]



def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)
best_models = hyperparameter_optimization(X, y,cv=5,scoring="accuracy")

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

#Prediction for a New Observation

X.columns
random_user = X.sample(1)
voting_clf.predict(random_user)













































































































































































































