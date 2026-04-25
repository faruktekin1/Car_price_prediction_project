# -*- coding: utf-8 -*-


import numpy as np
#Sayısal ve boyutsal işlemler rahat yapan python kütüphanesi
import pandas as pd
import matplotlib.pyplot as plt
#yukardaki 2 kütüphane benim veri üzerine çalışmamı ve erişmemi sağlar.

from sklearn.datasets import fetch_california_housing
#sklearn içerisinden datasets kısmına erişim hazırda bulunan ev veri setini sayfaya dahil eder.
from sklearn.model_selection import train_test_split
#model_selection yani model için seçilecek eğitim ve test verilerini parçalayan kısım

from sklearn.linear_model import LinearRegression
#doğrusal tahminleme olarak geçer.
from sklearn.tree import DecisionTreeRegressor
#sade karar ağacı olarak karşımıza çıkar basit bir ağaç modelidir.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#sade karar ağacı hariç her ağaç modeli ensemble uzantısı içerisinde bulunur.
#random forest ise sade karar ağacına göre birden fazla ağacı rastgele kullanmamızı sağlar
#veriler için de aynı yöntem sağlanır.

#Gradient ise desen yakalama yani verinin değerleri doğru yakalamak içib seçilen modeldir.
#kendisi bir önceki öğrenmesinde yapılan hatasını düzelterek geliştirir.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#metrikler yani modelden alınan sonuçları elimde bulunan sonuçlar ile karşılaştırdığım ve aralarındaki oranı
#kontrol ettiğim değerlendirme metrikleridir.

data = fetch_california_housing()
#veri setini çek ve data değişkenine at

df = pd.DataFrame(data.data, columns=data.feature_names)
#burda ise seçilen değeri sütunları ile ver dataframe çevir df içerisinde tut
df["Price"] = data.target
#price sütunune target olarak tanımlanan veri setindeki değerleri çek

print(df.head())
print(df.shape)

# 5. X ve y Ayrımı

X = df.drop("Price", axis=1)
#X -> özelliktir burada hedef sütununu almayız bu sayede sadece özellikler ile eğitilir.
y = df["Price"]
#y -> bu kısım ise hedeftir bunuda doğrudan modele veremeyiz o yüzden ayırıyorum 
#bunu yapma sebebim modelin neyin hedef neyinde özellik olduğunu anlaması

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    train_size=0.8,
    random_state= 42
)
#train veya test sizeları vermediğimiz durumda ise kendisi default olarak %75 e %25 olarak parçalar
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

def evaluate_model(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    #burdaki metrik modelin tahmini ve gerçek değer arasındaki mutlak hata farkını gösterir yani negatif çıktı olamaz
    # değer 100 bin  model 105 tahmin ederse 5000 hata 95 tahmin ederse yine aynı 5000 hata gösterir.
    mse = mean_squared_error(y_true, y_pred)
    #burda ise yine ortalama hatanın karesi alınır
    # istenen cevap 5 ama verile değer 7 ise bu bize 25 - 49 = 24 farkı verir.
    rmse = np.sqrt(mse)
    #burda ise alınan kare hatısı köke çeker bu sayede farkı gözlemleriniz 25 -> 5
    r2 = r2_score(y_true, y_pred)
    #20 sorunun cevabı y_test tutuluyor
    #20 pred ise -> x testte bulunan 20 özelliğin model tarafında tahmin edilen sonucu
    #y_test = 150 y_pred = 120 -> 30 
    #150 / 30 = 5 -> 5 te biri hataysa -> %80 doğru tahmin r2 = 0.8 
    #burada ise tahmin ile gerçek değer arasında olan farkı 0 ila 1 arasında gözlemleriz
    #eğer çıktının değeri 1 e yakınsa yanı 0.89 gibi değerler ise model veriyi iyi anlamış doğru tahmin yapıyor deriz
    #ama değer 0.3 gibi veya daha aşağı ise yani 0 a yakında model veriyi hiç iyi anlamamış ve çıktılar doğru değil

    print(f"\n{model_name}")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
evaluate_model("Linear Regression", y_test, lin_pred)

tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

evaluate_model("Decision Tree Regressor", y_test, tree_pred)

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

evaluate_model("Random Forest Regressor", y_test, rf_pred)

