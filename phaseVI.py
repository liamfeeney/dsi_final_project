import base64
import io
import pandas as pd
import numpy as np
import cv2
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pickle
from skimage.feature import graycomatrix, graycoprops, canny
from skimage.filters import rank
from skimage.morphology import disk
from skimage.measure import find_contours 
from scipy.stats import entropy, skew, kurtosis


app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Pneumonia Prediction App"

import base64
import phase6 as ha
import cv2
import numpy as np
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
   


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

models = [
        {"label": "CNN", "value": "cnn"},
        {"label": "Random Forest", "value": "rf"},
        {"label": "Naive Bayes", "value": "nb"},
        {"label": "Decision Tree", "value": "clf"},
        {"label": "SVM", "value": "svm"},
        {"label": "XGBoost", "value": "xgb"},
        {"label": "Logistic Regression", "value": "logreg"},
    ]
model_labels = {m["value"]: m["label"] for m in models}

model_dropdown = dbc.Select(
    id="model-select",
    options= models,
    value="cnn",
)

app.layout = dbc.Container(
    [
        # Title
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Pneumonia Prediction",
                    className="text-center my-4",
                ),
                width=12,
            )
        ),

        # ------------------------------------------
        # 1. Upload image
        # ------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("1. Upload X-Ray Image"),

                    dcc.Upload(
                        id="upload-image",
                        children=dbc.Button(
                            "Upload Image",
                            color="primary",
                            size="lg",
                        ),
                        multiple=False,
                    ),

                    html.Div(
                        id="upload-status",
                        className="mt-3",
                    ),
                ]
            ),
            className="mb-4",
        ),

        # Store the uploaded image
        dcc.Store(id="stored-image"),

        # ------------------------------------------
        # 2. Select model
        # ------------------------------------------

        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("2. Choose Model"),

                    model_dropdown,

                    dbc.Button(
                        "Predict",
                        id="predict-button",
                        color="success",
                        className="mt-3",
                        n_clicks=0,
                    ),
                ]
            ),
            className="mb-4",
        ),
        # ------------------------------------------
        # 3. Result
        # ------------------------------------------
        
        dbc.Row(
            [
                # Processed image
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Processed Image"),
                                html.Img(
                                    id="processed-image",
                                    style={
                                        "maxWidth": "80%",
                                        "height": "auto",
                                    },
                                    className="mt-3",
                                ),
                            ]
                        )
                    ),
                    width=6,
                ),
        
                # Prediction result
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Prediction"),
                                html.Div(
                                    id="prediction-result",
                                    className="mt-3",
                                ),
                            ]
                        )
                    ),
                    width=6,
                ),
            ],
            className="mb-4",
        ),
    ],
    fluid=True,
)

# --------------------------------------------------
# Callback 1: Upload and validate image
# --------------------------------------------------

@app.callback(
    Output("upload-status", "children"),
    Output("stored-image", "data"),
    Input("upload-image", "contents"),
)
def upload_image(contents):

    if contents is None:
        return (
            dbc.Alert(
                "Please upload an image.",
                color="warning",
            ),
            None,
        )

    try:
        # Split the data URL
        _, encoded = contents.split(",", 1)

        # Decode base64
        image_bytes = base64.b64decode(encoded)

        # Convert to NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        # Decode using OpenCV
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_GRAYSCALE,
        )
        
       
        
        # Check that OpenCV successfully decoded it
        if image is None:
            return (
                dbc.Alert(
                    "The uploaded file could not be read as an image.",
                    color="danger",
                ),
                None,
            )
            
        #Resize Image
        image = cv2.resize(image, (224,224), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert the image back to JPG and store it
        success, buffer = cv2.imencode(
            ".jpg",
            image,
        )

        if not success:
            return (
                dbc.Alert(
                    "Could not process the uploaded image.",
                    color="danger",
                ),
                None,
            )

        encoded_image = base64.b64encode(buffer).decode("utf-8")

        stored_image = f"data:image/png;base64,{encoded_image}"

        return (
            dbc.Alert(
                "Image uploaded successfully!",
                color="success",
            ),
            stored_image,
        )

    except Exception as e:
        return (
            dbc.Alert(
                f"Error uploading image: {str(e)}",
                color="danger",
            ),
            None,
        )


# --------------------------------------------------
# Callback 2: Predict
# --------------------------------------------------

@app.callback(
    Output("prediction-result", "children"),
    Output("processed-image", "src"),
    Input("predict-button", "n_clicks"),
    State("stored-image", "data"),
    State("model-select", "value"),
    prevent_initial_call=True,
)
def predict(n_clicks, stored_image, model):

    # Make sure an image has been uploaded
    if stored_image is None:
        return (
            dbc.Alert(
                "Please upload an image before making a prediction.",
                color="warning",
            ),
            None,
        )

    # ------------------------------------------
    # Decode stored image
    # ------------------------------------------

    _, encoded = stored_image.split(",", 1)

    image_bytes = base64.b64decode(encoded)

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_GRAYSCALE,
    )
    if image is None:
        return (
            dbc.Alert(
                "Could not read the stored image.",
                color="danger",
            ),
            None,
        )
   
    y_predict_proba = ha.check_for_pneumonia(image, model)
    selected_label = model_labels[model]
    confidence = np.round(y_predict_proba * 100, 2)
    output = f"Probablity of Pneumonia according to the {selected_label} model: {confidence}%"
    
    # ------------------------------------------
    # Encode processed image for display
    # ------------------------------------------

    success, buffer = cv2.imencode(
        ".jpg",
        image,
    )

    if not success:
        return (
            dbc.Alert(
                "Could not encode the processed image.",
                color="danger",
            ),
            None,
        )

    encoded_image = base64.b64encode(
        buffer
    ).decode("utf-8")

    image_src = (
        f"data:image/png;base64,{encoded_image}"
    )

    return (
        dbc.Alert(
            f"{output}",
            color = "danger" if y_predict_proba > 0.5 else "success",
        ),
        image_src,
    )


server = app.server

if __name__ == "__main__":
    app.run(debug=True)
# if __name__ == "__main__":
    # import webbrowser
    # from threading import Timer
    # url = "http://127.0.0.1:8050"
    # Timer(1, lambda: webbrowser.open(url)).start()
    # app.run(debug=True)
    