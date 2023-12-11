import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('Boston.csv')
data = data.sort_values(by=['tax'])



X1 = pd.DataFrame(data['tax'])
Y1 = pd.DataFrame(data['ptratio'])
lin_model1 = LinearRegression()
lin_model1.fit(X1, Y1)
equation1 = f"ptratio = {lin_model1.coef_[0][0]} * tax + {lin_model1.intercept_[0]}"

coefficients1 = {
    'a': round(lin_model1.coef_[0][0], 2),
    'b': round(lin_model1.intercept_[0], 2)
}

r2_value1 = r2_score(Y1, lin_model1.predict(X1))
rmse1 = np.sqrt(mean_squared_error(Y1, lin_model1.predict(X1)))

X2 = pd.DataFrame(data['rm'])
Y2 = pd.DataFrame(data['medv'])
lin_model2 = LinearRegression()
lin_model2.fit(X2, Y2)
equation2 = f"medv = {lin_model2.coef_[0][0]} * rm + {lin_model2.intercept_[0]}"

coefficients2 = {
    'a': round(lin_model2.coef_[0][0], 2),
    'b': round(lin_model2.intercept_[0], 2)
}

r2_value2 = r2_score(Y2, lin_model2.predict(X2))
rmse2 = np.sqrt(mean_squared_error(Y2, lin_model2.predict(X2)))

plt.subplot(1, 2, 1)
plt.scatter(data['tax'], data['ptratio'])
plt.plot(X1, lin_model1.predict(X1), label='Linear Regression', color='red')
plt.xlabel('tax')
plt.ylabel('ptratio')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(data['rm'], data['medv'])
plt.plot(X2, lin_model2.predict(X2), label='Linear Regression', color='blue')
plt.xlabel('rm')
plt.ylabel('medv')
plt.legend()

plt.tight_layout()
plt.show()

print("--- TAX(PTRATIO) ---")
print("Рівняння: ", equation1)
print("Коефіцієнти: ", coefficients1)
print("Значення R2: ", r2_value1)
print("RMSE: ", rmse1)

print("\n--- RM(MEDV) ---")
print("Рівняння: ", equation2)
print("Коефіцієнти: ", coefficients2)
print("Значення R2: ", r2_value2)
print("RMSE: ", rmse2)
