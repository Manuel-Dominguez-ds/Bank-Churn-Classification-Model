import pickle
import numpy as np
import pandas as pd
from Functions import *
from datetime import date

class Scorer():
    def __init__(self, df_path) -> None:
        print("\n > Initialized Scoring Pipeline.")
        self.df_path = df_path

    def orchestrator(self):
        print("\n > Initialized Scoring process.")
        test_data=pd.read_csv(self.df_path)
        print("\n > Test data read.")
        sample_data=test_data.drop(['Id'],axis=1)
        print("\n > Loading model.")
        model = self.load_model()
        print("\n > Model loaded.")
        
        print("\n > Loading scaler.")
        scaler = self.load_scaler()
        print("\n > Scaler loaded.")
        
        print("\n > Loading Encoders.")
        encoder1,encoder2 = self.load_encoders()
        print("\n > Encoders loaded.")
        
        print("\n > Loading Imputers.")
        cat_imputer,num_imputer = self.load_imputers()
        print("\n > Imputers loaded.")
        
        print("\n > Preprocessing.")
        sample_data = self.preprocessing(sample_data,scaler,encoder1,encoder2,cat_imputer,num_imputer)
        print("\n > Loading vectorizer.")
      
        print("\n > Making predictions.")
        predictions = self.predict(model=model, X=sample_data)
        test_data["Predictions"] = predictions
        test_data=test_data[['Id','Predictions']]
        print('\n------------------\nPredictions for Test Data\n------------------\n')
        #print(sample_data)
        filename= f'Predictions - {date.today()}.csv'
        with open(filename, 'a') as file:
            file.write("ID,Predictions\n")
            for i in range(test_data.shape[0]):
                text = str(test_data.Id[i])
                predictions = str(test_data.Predictions[i])
                encoded_text = text.encode('utf-8', errors='ignore')
                encoded_predictions = predictions.encode('utf-8', errors='ignore')
                decoded_text = encoded_text.decode('utf-8')
                decoded_predictions = encoded_predictions.decode('utf-8')

                line = f"{decoded_text},{decoded_predictions}"
                file.write(line + "\n")
        print(f"Training results have been saved to {filename}")

    def load_model(self):
        print("\n > Loading BestModel from memory.")
        with open('Pickle/best_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
            return loaded_model
        
    def load_scaler(self):
        print("\n > Loading Scaler from memory.")
        with open('Pickle/Scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            return scaler
        
    def load_encoders(self):
        print("\n > Loading Scaler from memory.")
        with open('Pickle/CountFrequency.pkl', 'rb') as file:
            encoder1 = pickle.load(file)
            
        with open('Pickle/OrdinalEncoder.pkl', 'rb') as file:
            encoder2 = pickle.load(file)
        return encoder1,encoder2
        
    def load_imputers(self):
        print("\n > Loading Imputers from memory.")
        with open('Pickle/cat_imputer.pkl', 'rb') as file:
            cat_imputer = pickle.load(file)
            
        with open('Pickle/num_imputer.pkl', 'rb') as file:
            num_imputer = pickle.load(file)
        return cat_imputer,num_imputer
    
    def preprocessing(self, X_test,scaler,encoder1,encoder2,cat_imputer,num_imputer):
        print("\n > Preprocessing text data.")
        print("\n > Null treatment.")
        #X_test=imputer(X_test,cat_imputer,num_imputer,train=False)
        
        print("\n > Performing Feature Engineering.")
        X_test=X_test.drop(['Surname','Passport','Gender'],axis=1)
        X_test=new_variables(X_test)
        
        print("\n > Encoding data.")
        X_test=encoder1.transform(X_test)
        X_test=encoder2.transform(X_test)
        X_test=encoding(X_test)
        
        print("\n > Scaling data.")
        X_test=scaler.transform(X_test)
        
        print("\n > Data preprocessed.")
        return X_test
    
    def predict(self, model, X):
        print("\n > Predicting...")
        predictions = model.predict_proba(X)[:, 1]
        predictions = (predictions > 0.555).astype(int)
        return predictions
