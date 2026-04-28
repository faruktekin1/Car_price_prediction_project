import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import numpy as np
import joblib

# Dosya yolunun başındaki 'r' harfi Windows'taki ters bölü (\) işaretlerinin hata vermesini engeller.
df = pd.read_csv(r"C:\Users\faruk\OneDrive\Desktop\python işlerim\car_sales_data.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())

verbal_columns=["Make","Model","Fuel Type","Transmission"]
numerical_columns=["Year","Engine Size","Mileage"]

X=df.drop("Price",axis=1)
y=df["Price"]

x_train,x_test,y_train,y_test=train_test_split(
     X,
     y,
     train_size=0.8,
     random_state=40
)

my_models={
    "linear regression":LinearRegression(),
    "Random forest":RandomForestRegressor(n_estimators=100, random_state=40),
    "Decision tree":DecisionTreeRegressor(random_state=40),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=40) 
}
preprocess=ColumnTransformer(
    transformers=[("num",StandardScaler(),numerical_columns),
                    ("verbal",OneHotEncoder(handle_unknown='ignore'),verbal_columns)])


model=[]

def train(models):
    for key,value in models.items():
        pipe=Pipeline(
            steps=[("preprocess",preprocess),
                   ("model",value)])
        pipe.fit(x_train,y_train)
        y_pred = pipe.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) 

        print(f"\n{key}")
        print("-" * 40)
        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R2   : {r2:.4f}")
        print("-"*40)
        model.append({
            "model":key,
            "r2_score":r2,
            "mse":mse,
            "mae":mae,
            "my_pipe": pipe,
            "rmse":rmse})
        

train(my_models)

def car_prediction_application():
    Make = input("Make: ")
    Car_Model = input("Model (ex" \
    ": Corolla, A3): ")
    Year = int(input("Year: "))
    Engine_Size = float(input("Engine Size: "))
    Mileage = int(input("Mileage: "))
    Fuel_type = input("Fuel Type: ")
    Transmission = input("Transmission: ")

    new_car = {
        "Make": [Make],
        "Model": [Car_Model],
        "Year": [Year],
        "Engine Size": [Engine_Size],
        "Mileage": [Mileage],
        "Fuel Type": [Fuel_type],
        "Transmission": [Transmission]
    }

    input_df = pd.DataFrame(new_car)

    print("\n" + "="*40)
    print("PRICE PREDICTIONS OF ALL MODELS")
    print("="*40)
    
    
    for item in model:

        predict = item["my_pipe"].predict(input_df)[0]
        
        print(f"{item['model']:20}: {predict:,.2f} TL")
    
    print("="*40)

car_prediction_application()

joblib.dump(model, 'trained_car_models.pkl')

            


        

