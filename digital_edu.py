#создай здесь свой индивидуальный проект!
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
df = pd.read_csv('train.csv')

df.drop(['has_photo','bdate','id','has_mobile','followers_count','graduation','relation','life_main','people_main','city','last_seen','occupation_name','career_start','career_end'],axis=1,inplace=True)
df.info()
def apply_sex(sex):
    if sex == 2:
        return 0
    return 1
print(df['education_form'].value_counts())
df['education_form'].fillna('Full-time',inplace=True)
df['sex'] = df['sex'].apply(apply_sex)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop('education_form',axis=1,inplace=True)
df.info()

def apply_education_status(status):
    if status.find('Student') != -1:
        return 0
    elif status.find('Alumnus') != -1:
         return 1
    elif status == 'Undergraduate applicant':
        return 2
    elif status == 'Candidate of Sciences':
        return 3
    else:
        return 4
    
def apply_langs(langs):
    if 'Русский' in langs.split(';'):
        return 1
    else:
        return 0
df['langs'] =  df['langs'].apply(apply_langs)
df['education_status'] =  df['education_status'].apply(apply_education_status)
df[list(pd.get_dummies(df['education_status']).columns)] = pd.get_dummies(df['education_status'])
df.drop('education_status',axis=1,inplace=True)
def apply_occupation_type(type):
    if 'university':
        return 0
    return 1
df['occupation_type'] =  df['occupation_type'].apply(apply_occupation_type)
x = df.drop('result',axis=1)
y = df['result']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
percent = accuracy_score(y_test,y_pred) * 100
print(str(round(percent,1))+'%')
