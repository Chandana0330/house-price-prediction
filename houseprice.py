import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("train.csv")
X = data[['GrLivArea','BedroomAbvGr','FullBath']]
y = data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
new_house = pd.DataFrame([[2000,3,2]],
columns=['GrLivArea','BedroomAbvGr','FullBath'])
prediction = model.predict(new_house)
print("Predicted Price:", prediction)
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
