import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
from inference_api import get_FINED_edge, read_imagefile
import cv2
import numpy as np
import io



app = FastAPI()

#route
@app.get('/')
def index():
    return {"Data" : "Homepage Test"}

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    #img = read_imagefile(file.read())
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    prediction = get_FINED_edge(img)
    res, im_png = cv2.imencode(".png", prediction)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
    #return {"Data" : "OK"}

if __name__ == '__main__':
    uvicorn.run(app,host="0.0.0.0",port=8000)

