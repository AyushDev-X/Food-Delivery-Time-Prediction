# import pandas as pd
# import requests
# from pathlib import Path

# # path for data
# root_path = Path(__file__).parent.parent
# data_path = root_path /"food-delivery-time-prediction" / "data" / "raw" / "swiggy.csv"

# # prediction endpoint
# # predict_url = "http://13.201.83.141//predict"
# predict_url = "http://13.201.83.141:8000/predict"

# # sample row for testing the endpoint
# sample_row = pd.read_csv(data_path).dropna().sample(1)
# print("The target value is", sample_row.iloc[:,-1].values.item().replace("(min) ",""))
    
# # remove the target column
# data = sample_row.drop(columns=[sample_row.columns.tolist()[-1]]).squeeze().to_dict()
# print(data)

# # get the response from API
# response = requests.post(url=predict_url,json=data,timeout=10)

# print("The status code for response is", response.status_code)

# if response.status_code == 200:
#     print(f"The prediction value by the API is {float(response.text):.2f} min")
# else:
#     print("Error:", response.status_code)


import pandas as pd
import requests
from pathlib import Path

root_path = Path(__file__).parent.parent
data_path = root_path / "food-delivery-time-prediction" / "data" / "raw" / "swiggy.csv"

predict_url = "http://127.0.0.1:8000/predict"

sample_row = pd.read_csv(data_path).dropna().sample(1)

print("Target:", sample_row.iloc[:,-1].values.item())

# drop target
X = sample_row.drop(columns=[sample_row.columns[-1]])

# FIX: strip spaces + convert to proper JSON-safe format
X = X.apply(lambda col: col.astype(str).str.strip() if col.dtype == "object" else col)

data = X.squeeze().to_dict()

print("Clean payload:", data)

response = requests.post(predict_url, json=data, timeout=10)

print("Status:", response.status_code)
print("Response:", response.text)