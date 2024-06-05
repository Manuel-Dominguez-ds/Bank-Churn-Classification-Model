import pandas as pd
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import CountFrequencyEncoder,OrdinalEncoder
import pickle

#def new_variables(df):
def new_variables(df):
  df['CreditScore_x_Age']=df['CreditScore']/df['Age']
  df['CreditScore_x_Balance']=df['Balance']/df['CreditScore']
  df['NumOfProducts_x_Age']=df['NumOfProducts']/df['Age']
  df['Tenure_x_Age']=df.apply(lambda x: x['Tenure']/x['Age'] if x['Age']!=None else 0,axis=1)
  df['%SalaryInBank']=(df['Tenure']*df['Balance'])/df['EstimatedSalary']
  df['Balance_x_EstimatedSalary']=df['Balance']/df['EstimatedSalary']
  df['AgeofEntry']=df['Age']-df['Tenure']
  df['CustomerEngagement']=df.apply(lambda x:x['Age']*x['CreditScore']*x['NumOfProducts'],axis=1)
  df['EducationProduct']=df.apply(lambda x:x['Age']*x['EducationYears']*x['NumOfProducts'],axis=1)

  return df

def encoding(df):
  df=pd.get_dummies(data=df,columns=['Geography'])
  return df

def imputer(df,cat_imputer,num_imputer,train=True):
  cat=[]
  num=[]
  if 'Id' in df.columns:
    df=df.drop('Id',axis=1)
  else:
    pass
  if 'Exited' in df.columns:
    for col in df.drop('Exited',axis=1).columns:
      if df[col].dtype=='object':
        cat.append(col)
      else:
        num.append(col)
    if train==True:
      df[cat]=cat_imputer.fit_transform(df[cat])
      df[num]=num_imputer.fit_transform(df[num])
      with open('Pickle/num_imputer.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'num_imputer.pkl'⬇️💾")
            pickle.dump(num_imputer, file)
      with open('Pickle/cat_imputer.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'cat_imputer.pkl'⬇️💾")
            pickle.dump(cat_imputer, file)
    else:
      df[cat]=cat_imputer.transform(df[cat])
      df[num]=num_imputer.transform(df[num])
    return df
  else:
    for col in df.columns:
      if df[col].dtype=='object':
        cat.append(col)
      else:
        num.append(col)
    if train==True:
      df[cat]=cat_imputer.fit_transform(df[cat])
      df[num]=num_imputer.fit_transform(df[num])
      with open('Pickle/num_imputer.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'num_imputer.pkl'⬇️💾")
            pickle.dump(num_imputer, file)
      with open('Pickle/cat_imputer.pkl', 'wb') as file:
            print(f"\n > Exporting best model to pkl file 'cat_imputer.pkl'⬇️💾")
            pickle.dump(cat_imputer, file)
    else:
      df[cat]=cat_imputer.transform(df[cat])
      df[num]=num_imputer.transform(df[num])
    return df

def scaler(X_train,X_val):
    print("\n > Scaling data.")
    ss=StandardScaler()
    train_columns=X_train.columns
    val_columns=X_val.columns
    X_train_encoded = ss.fit_transform(X_train) 
    X_val_encoded = ss.transform(X_val)  
    
    X_train=pd.DataFrame(X_train_encoded,columns=train_columns)
    X_val=pd.DataFrame(X_val_encoded,columns=val_columns)
    
    with open('Pickle/Scaler.pkl', 'wb') as file:
        print(f"\n > Exporting Scaler to pkl file 'Scaler.pkl'⬇️💾")
        pickle.dump(ss, file)
    return X_train_encoded,X_val_encoded

def encoder(X_train,X_val,y_train):
    print("\n > Encoding data.")
    encoder = CountFrequencyEncoder(
        encoding_method='frequency',
        variables=['Geography'])
    
    encoder2 = OrdinalEncoder(
        encoding_method='ordered',
        variables=['Geography'],
        ignore_format=True)
    
    X_train=encoder.fit_transform(X_train)
    X_train=encoder2.fit_transform(X_train,y_train)
    X_train=encoding(X_train)
    
    X_val=encoder.transform(X_val)
    X_val=encoder2.transform(X_val)
    X_val=encoding(X_val)
    
    with open('Pickle/CountFrequency.pkl', 'wb') as file:
        print(f"\n > Exporting CountFrequency to pkl file 'CountFrequency.pkl'⬇️💾")
        pickle.dump(encoder, file)
        
    with open('Pickle/OrdinalEncoder.pkl', 'wb') as file:
        print(f"\n > Exporting OrdinalEncoder to pkl file 'OrdinalEncoder.pkl'⬇️💾")
        pickle.dump(encoder2, file)
        
    return X_train,X_val