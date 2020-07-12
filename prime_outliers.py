# evaluate model on the raw dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest # First method for outliers
from sklearn.covariance import EllipticEnvelope # Second method for outliers
from sklearn.neighbors import LocalOutlierFactor # Third method for outliers
from sklearn.svm import OneClassSVM # Fourth method for outliers
from sklearn.metrics import mean_absolute_error


# load the dataset
df = pd.read_csv("./data/housing.csv", header=None)

# retrieve the array
data = df.values

# split into input and output elements
X, y = data[:, :-1], data[:, -1]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)

# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)

# identify outliers in training dataset with Isolation Forest Algorithm
#iso = IsolationForest(contamination=0.1)
#yhat = iso.fit_predict(X_train) # Find outliers

# identify outliers with Minimum Covarience Determinant 
#ee = EllipticEnvelope(contamination=0.01)
#yhat = ee.fit_predict(X_train)

# identify outliers with Local Outlier Factor
#lof = LocalOutlierFactor()
#yhat = lof.fit_predict(X_train)

# identify outliers with One Class SVM (Support Vector Machine)
ocs = OneClassSVM(nu=0.03)
yhat = ocs.fit_predict(X_train)

# select all raws that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask,:], y_train[mask]

# summarize the shape of the updated training dataset
print(X_train.shape, y_train.shape)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# evaluate the model
yhat = model.predict(X_test)

# evaluate predictions
mae = mean_absolute_error(y_test, yhat)
print(f"MAE: {mae:.3f}")
