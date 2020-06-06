import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

clf = linear_model.LinearRegression()

# Lấy các cột có kiểu dữ liệu "object" đưa vào 1 Data Frame mới
def get_obj_types(df):
    obj_df = df.select_dtypes(include=['object']).copy()

    return obj_df


# Hàm chuyển cột "manufacturer"
def transform_manufacture_transmisson(obj_df):
    lb_make = LabelEncoder()
    obj_df["manufacturer"] = lb_make.fit_transform(obj_df["manufacturer"])
    obj_df["transmission"] = lb_make.fit_transform(obj_df["transmission"])
    obj_df["engineFuel"] = lb_make.fit_transform(obj_df["engineFuel"])
    obj_df["engineType"] = lb_make.fit_transform(obj_df["engineType"])
    obj_df["color"] = lb_make.fit_transform(obj_df["color"])
    obj_df["model"] = lb_make.fit_transform(obj_df["model"])
    obj_df["drivetrain"] = lb_make.fit_transform(obj_df["drivetrain"])
    obj_df["bodyType"] = lb_make.fit_transform(obj_df["bodyType"])
    obj_df["feature_0"] = lb_make.fit_transform(obj_df["feature_0"])
    obj_df["feature_1"] = lb_make.fit_transform(obj_df["feature_1"])
    obj_df["feature_2"] = lb_make.fit_transform(obj_df["feature_2"])
    obj_df["feature_3"] = lb_make.fit_transform(obj_df["feature_3"])
    obj_df["feature_4"] = lb_make.fit_transform(obj_df["feature_4"])
    obj_df["feature_5"] = lb_make.fit_transform(obj_df["feature_5"])
    obj_df["feature_6"] = lb_make.fit_transform(obj_df["feature_6"])
    obj_df["feature_7"] = lb_make.fit_transform(obj_df["feature_7"])
    obj_df["feature_8"] = lb_make.fit_transform(obj_df["feature_8"])
    obj_df["feature_9"] = lb_make.fit_transform(obj_df["feature_9"])

    return obj_df


# Hàm chuyển cột "body types"
def transform_bodyType_drivetrain(obj_df):
    obj_df = pd.get_dummies(obj_df, columns=["bodyType","drivetrain","manufacturer","model"], prefix=["body","drive","manu","model"])
    return obj_df


# Hàm đọc dữ liệu từ file csv đưa vào Data Frame
def read_file_csv(path):
    raw_data = pd.read_csv(path, sep=",")
    return raw_data



def RMSE(x_test,y_test):
    sum = 0
    result = clf.predict(x_test)
    y_test = y_test.to_numpy()
    for i in range(0,len(x_test)):
        sum+=(result[i]-y_test[i])**2
    return np.math.sqrt(sum/len(x_test))




X_train = read_file_csv("res\\X_train.csv")
Y_train = read_file_csv("res\\Y_train.csv")

X_transform = transform_manufacture_transmisson(X_train)
X_transform = transform_bodyType_drivetrain(X_transform)

X_transform["engineCapacity"] = X_transform['engineCapacity'].fillna(0)
X_transform = X_transform.drop(columns = ["id","photos"])
Y_train = Y_train.drop(columns = "id")

#Chia tập dữ liệu thành 2 phần
X_train, X_test, y_train, y_test = train_test_split(X_transform, Y_train, test_size=0.2, random_state=0)


clf.fit(X_train, y_train)

print(clf.coef_)
print(clf.predict(X_test))
print(RMSE(X_test,y_test))

