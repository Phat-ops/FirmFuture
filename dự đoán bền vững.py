import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
#from ydata_profiling.profile_report import ProfileReport
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
#read file
df = pd.read_csv("company_esg_financial_dataset.csv")

#statictis
# y_data = ProfileReport(df)
# y_data.to_file("predict.html")
#print(df.info())

#split data, axis =1
df= df.drop(df[["Year"]],axis=1)
x= df.iloc[:,2:-1]
y = df.iloc[:,-1:]

#split data, axis =0
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42)
#create pipeline

#missing stand
growth_col = Pipeline(steps=[
    ("missing", SimpleImputer(strategy="median")),
    ("stand", StandardScaler()),
])

processing = ColumnTransformer([
    ("fix", growth_col, ["GrowthRate"] ),
    ("onehot", OneHotEncoder(),["Industry","Region"]),
    ("stand", StandardScaler(),['Revenue','ProfitMargin','MarketCap','ESG_Overall','ESG_Environmental','ESG_Social','ESG_Governance','CarbonEmissions','WaterUsage'])
])

model = Pipeline(steps=[
    ("process",processing),
    ("model",RandomForestRegressor(random_state=42) )
])

model.fit(x_train,y_train)
result = model.predict(x_test)
print(r2_score(y_test,result))
#0.9062508455884108