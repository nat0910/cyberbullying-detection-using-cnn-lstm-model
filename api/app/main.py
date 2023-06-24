
# ---------------- API Code ---------------- #


from fastapi import FastAPI,UploadFile,HTTPException,File,Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse,FileResponse
from uvicorn import run
from typing import Union
import os

from typing_extensions import Annotated

from io import BytesIO,StringIO

from starlette.background import BackgroundTask

import pandas as pd

from app.modules.preprocessing.text_preprocessing import preprocessing_input_text_cnn,preprocessing_input_text_ltsm
from app.modules.preprocessing.datafram_preprocessing import df_prepocessing_cnn,df_prepocessing_lstm

from app.modules.cnn.cnn_cyberbullying_detection import detect_cyberbullying_cnn_text
from app.modules.lstm.lstm_cyberbullying_detection import detect_cyberbullying_ltsm_text

from app.modules.cnn.cnn_cyberbullying_detection_dataframe import detect_cyberbullying_cnn_dataframe
from app.modules.lstm.lstm_cyberbullying_detection_dataframe import detect_cyberbullying_ltsm_dataframe

import time

timestr = time.strftime("%Y%m%d-%H%M%S")

tags_metadata =[
{
    "name": "default",
    "description": "Default endpoint of the API"
},
{
    "name":"cnn-endpoints",
    "description": "These endpoints is used to detect cyberbullying in single text as well as for a file using the CNN model."
},
{
    "name":"lstm-endpoints",
    "description": "These endpoints is used to detect cyberbullying in single text as well as for a file using the LSTM model."
}
]

app = FastAPI(
    title="Cyberbullying detection",
    description="This API is for detection of cyberbullying in single text as well as for a file ('csv','excel') containing multiple texts.This API has 2 models CNN having accuarcy of 90.70% And LTSM having accuracy of 94.00%.",
    openapi_tags=tags_metadata,
)

headers = ["*"]
methods = ["*"]
origins = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR,"uploads")

@app.get('/',tags=['default'])
async def starting_endpoint():
    return 'Please visit http://127.0.0.1:8000/docs to learn more about the API.'
@app.post('/cnn-predict-text',tags=['cnn-endpoints'])
async def predict_cyberbullying_using_cnn_model(q:str):
    __dtc = detect_cyberbullying_cnn_text()
    __pit_cnn = preprocessing_input_text_cnn()
    __pit_cnn.fit(q)
    __clean_data = __pit_cnn.clean_data()
    __dtc.fit(__clean_data)
    return __dtc.predict()

@app.post("/cnn-predict-dataframe",tags=['cnn-endpoints'])
async def upload_file(file:UploadFile):
    if not (file.content_type == 'text/csv'):
        if not (file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            raise HTTPException(400, detail="Invalid document type")
        
    contents = file.file.read()
    data = BytesIO(contents)

    if file.content_type == 'text/csv':    
        __df = pd.read_csv(data)
        file.file.close()

    if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':    
        __df = pd.read_excel(data)
        file.file.close()


    __df_pre_cnn = df_prepocessing_cnn(__df. iloc[:,0])
    __clean_data = __df_pre_cnn.clean_data()
    __dcdf = detect_cyberbullying_cnn_dataframe(__clean_data)
    __predicted_output_df = __dcdf.predict()

    __df['cyberbullying_type_predicted'] = __predicted_output_df
    
    __new_filename = "{}_{}_cnn.xlsx".format(os.path.splitext(file.filename)[0],timestr)
    __SAVE_FILE_PATH = os.path.join(UPLOAD_DIR,__new_filename)
    __df.to_csv(__SAVE_FILE_PATH,index=False)

    def cleanupFunction():
        os.remove(__SAVE_FILE_PATH)

    return FileResponse(filename=__new_filename.split('.')[0],path=__SAVE_FILE_PATH,media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',background=BackgroundTask(cleanupFunction))


@app.post('/ltsm-predict-text',tags=['lstm-endpoints'])
async def predict_cyberbullying_using_lstm_model(q:str):
    __dltsm = detect_cyberbullying_ltsm_text()
    __pit_ltsm = preprocessing_input_text_ltsm()
    __pit_ltsm.fit(q)
    __clean_data = __pit_ltsm.clean_data()
    __dltsm.fit(__clean_data)
    return __dltsm.predict()



@app.post("/lstm-predict-dataframe",tags=['lstm-endpoints'])
async def upload_file(file:UploadFile,start:int,end:int):

    if start>end:
        raise HTTPException(400,detail="Starting position cannot be greater than ending position")

    if (start-end)>500:
        raise HTTPException(400,detail="Please dercease the end number")

    if not (file.content_type == 'text/csv'):
        if not (file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
            raise HTTPException(400, detail="Invalid document type")
        
    __contents = file.file.read()
    __data = BytesIO(__contents)

    if file.content_type == 'text/csv':    
        df = pd.read_csv(__data)
        file.file.close()

    if file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':    
        df = pd.read_excel(__data)
        file.file.close()

    __df = df.iloc[start:end,0]
    __df_pre = df_prepocessing_lstm(__df)
    __clean_data = __df_pre.clean_data()
    __dlstmdf = detect_cyberbullying_ltsm_dataframe()
    __dlstmdf.fit(__clean_data)
    __predicted_output_df = __dlstmdf.predict()
    __output = pd.DataFrame({"text":__df,"predicted":__predicted_output_df})


    __new_filename = "{}_{}_lstm.xlsx".format(os.path.splitext(file.filename)[0],timestr)
    __SAVE_FILE_PATH = os.path.join(UPLOAD_DIR,__new_filename)
    __output.to_csv(__SAVE_FILE_PATH,index=False)


    def cleanupFunction():
        os.remove(__SAVE_FILE_PATH)

    return FileResponse(filename=__new_filename.split('.')[0],path=__SAVE_FILE_PATH,media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',background=BackgroundTask(cleanupFunction))



# @app.get("/predicted_file",tags=['get-file'])
# async def get_predicted_file(filename:str):
#     SAVE_FILE_PATH = os.path.join(UPLOAD_DIR,filename)
#     if os.path.exists(SAVE_FILE_PATH):
#         return FileResponse(path=SAVE_FILE_PATH,media_type='text/csv',filename=filename)
#     raise HTTPException(404,detail="File Not Found!!")