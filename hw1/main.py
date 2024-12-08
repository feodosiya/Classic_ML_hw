from fastapi import FastAPI, UploadFile, File
import pickle
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import BytesIO
from joblib import load
import re
from starlette.responses import StreamingResponse
import numpy as np

app = FastAPI()
# Загрузка модели и OneHotEncoder
with open('model.pkl', 'rb') as f:
    model = load(f)
with open('ohe2.pkl', 'rb') as f:
    ohe2 = load(f)
with open('scalerv.pkl', 'rb') as f:
    scaler = load(f)

class DataPreprocessing:
    def __init__(self, data):
        if isinstance(data, dict):
            self.df = pd.DataFrame(data, index=[0])
        else:
            self.df = data

    # Убираем единицы измерения
    def extract_number(self, value):
        match = re.search(r'\d+(?:\.\d+)?', str(value))
        return float(match.group()) if match else pd.isna

    def data_cleaning(self):
        self.df.name = self.df.name.str.split(' ').str[0]
        change_columns = ['mileage', 'engine', 'max_power']
        for column in change_columns:
            if column in self.df.columns:
                self.df[column] = self.df[column].apply(self.extract_number)

        # Приведение типов
        if 'engine' in self.df.columns:
            self.df['engine'] = self.df['engine'].astype(int)
        if 'seats' in self.df.columns:
            self.df['seats'] = self.df['seats'].astype(int)
        return self
        
    def drop_col_torque(self):
        if 'torque' in self.df.columns:
            self.df = self.df.drop(columns=['torque'])
        return self

    # Кодирование категориальных признаков
    def preproc(self):
        num_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
        scaled_df = pd.DataFrame(scaler.transform(self.df[num_features]), columns=self.df[num_features].columns)

        self.df.seats = self.df.seats.astype(str)
        categorical_features = ['name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']

        if all(feature in self.df.columns for feature in categorical_features):
            encoded_data = ohe2.transform(self.df[categorical_features])
            encoded_df = pd.DataFrame(encoded_data, columns=ohe2.get_feature_names_out(categorical_features))
            self.df = scaled_df.join(encoded_df)
            
        if 'selling_price' in self.df.columns:
            self.df.drop(columns=['selling_price'], inplace=True)
            
        self.df = self.df[self.df.columns[5:].tolist() + self.df.columns[:5].tolist()]
        return self.df

# models
class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

# routes
@app.post("/predict_item")
def predict_item(item: Item) -> str:
    data = item.model_dump()
    preprocessor = DataPreprocessing(data)
    preprocessor = preprocessor.data_cleaning().drop_col_torque().preproc()

    prediction = model.predict(preprocessor)

    return f'prediction price: {float(prediction[0]):.2f}'

@app.post("/predict_items", response_class=StreamingResponse)
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    df_test = pd.read_csv(BytesIO(content))

    required_columns = ['mileage', 'engine', 'max_power', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    missing_columns = [col for col in required_columns if col not in df_test.columns]
    if missing_columns:
        return {"error": f"The following required columns are missing: {', '.join(missing_columns)}"}

    if 'selling_price' in df_test.columns:
        df_test.drop(columns=['selling_price'], inplace=True)

    formatting = DataPreprocessing(df_test)
    formatting_data = formatting.data_cleaning().drop_col_torque().preproc()

    predict = model.predict(formatting_data)
    df_test['predicted_price'] = predict

    output_stream = BytesIO()
    df_test.to_csv(output_stream, index=False)
    output_stream.seek(0)

    response = StreamingResponse(output_stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

