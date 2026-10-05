import numpy as np 
import pickle
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops, canny
from skimage.filters import rank
from skimage.morphology import disk
from skimage.measure import find_contours 
from scipy.stats import entropy, skew, kurtosis
import phase1 as p1
from pathlib import Path
BASE_DIR = Path.cwd()

def check_for_pneumonia(img, algo):   
    with open(BASE_DIR / f"{algo}.pkl", "rb") as f:
        model = pickle.load(f) 
    print(algo)
    if algo == "cnn":
        img = img.astype(np.float32) / 255.0
        if(len(img.shape)==2):
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
        y_pred_prob = model.predict(img).flatten()
    elif algo=="logreg":
        subset = model["features"]
        loaded_model = model["model"]
        features = extract_features(img)[subset]
        y_pred_prob = loaded_model.predict_proba(features)[:,1]
    elif algo == "svm":
        loaded_model = model["model"]
        loaded_scaler = model["scaler"]
        features = extract_features(img)
        # Apply and predict
        features_scaled = loaded_scaler.transform(features)
        y_pred_prob = loaded_model.predict_proba(features_scaled)[:,1]
    else:
        features = extract_features(img)
        y_pred_prob = model.predict_proba(features)[:,1]
    return(y_pred_prob)

def extract_features(img):
    ##From Phase 2
    df = pd.read_csv(BASE_DIR / "data_for_models.csv")
    X = df.drop(["ImageID", "TTV", "Original_Filepath", "Processed_Filepath", "Label"], axis=1)
    objects = X.select_dtypes(include="object").columns
    X[objects] = X[objects].astype("category")
    X = pd.get_dummies(X, columns=objects, drop_first=True)
    region_bins = {'top': [0.60955776, 0.97699754, 1.05721486, 1.57212119], 'mid': [0.8401788 , 1.00045872, 1.05395725, 1.52371745],'bot': [0.87177782, 1.13190838, 1.23029491, 1.69494614]}
    global_bin = [0.30459443, 0.4528695, 0.50903713, 0.65972394]
    features = p1.pixel_features(img) | p1.texture_features(img) | p1.edge_features(img) | p1.region_features(img, 0, 0.05)
    #print(features)
    features = pd.DataFrame([features])
    for region in ['top', 'mid', 'bot']:
        bins = region_bins[region]
        #print(bins)
        features[f'brightness_std_{region}'] = features[f'brightness_std_{region}'] / features['pixel_mean'] #normalise by overall exposure
        features[f'brightness_mean_{region}'] = features[f'brightness_mean_{region}']/features['pixel_mean'] #normalise by overall exposure
        if (features[f'brightness_mean_{region}'][0] <= bins[1]):
            #features[f'{region}_brightness_tier'] = 'Dark'  
            features[f'{region}_brightness_tier_Dark'] = 1
            features[f'{region}_brightness_tier_Medium'] = 0
            features[f'{region}_brightness_tier_Bright'] = 0
        elif features[f'brightness_mean_{region}'][0] <= bins[2] :
            #features[f'{region}_brightness_tier'] = 'Medium' 
            features[f'{region}_brightness_tier_Dark'] = 0
            features[f'{region}_brightness_tier_Medium'] = 1
            features[f'{region}_brightness_tier_Bright'] = 0
            
        else:
            #features[f'{region}_brightness_tier'] = 'Bright'
            features[f'{region}_brightness_tier_Dark'] = 0
            features[f'{region}_brightness_tier_Medium'] = 0
            features[f'{region}_brightness_tier_Bright'] = 1
        features = features.drop([f"is_bright_{region}"], axis=1)
    if features['pixel_mean'][0] <= global_bin[1] :
        #features['global_brightness_tier'] = 'Dark' 
        features[f'global_brightness_tier_Dark'] = 1
        features[f'global_brightness_tier_Medium'] = 0
        features[f'global_brightness_tier_Bright'] = 0
    elif features['pixel_mean'][0] <= global_bin[2] :
        #features['global_brightness_tier'] = 'Medium'
        features[f'global_brightness_tier_Dark'] = 0
        features[f'global_brightness_tier_Medium'] = 1
        features[f'global_brightness_tier_Bright'] = 0
    else:
        #features['global_brightness_tier'] = 'Bright'
        features['global_brightness_tier_Dark'] = 0
        features['global_brightness_tier_Medium'] = 0
        features['global_brightness_tier_Bright'] = 1
    if features["sym_classification"][0] == "not_symmetric":
       features["sym_classification_not_symmetric"] = 1
       features["sym_classification_roughly_symmetric"] = 0
       features["sym_classification_symmetric"] = 0
    elif features["sym_classification"][0] == "roughly_symmetric":
       features["sym_classification_not_symmetric"] = 0
       features["sym_classification_roughly_symmetric"] = 1
       features["sym_classification_symmetric"] = 0
    else:
       features["sym_classification_not_symmetric"] = 0
       features["sym_classification_roughly_symmetric"] = 0
       features["sym_classification_symmetric"] = 1
       
    features["contrast_cv"] = features["contrast_std"] / features["contrast"]
    features["pixel_skew_by_mean"] = features["pixel_skew"] / features["pixel_mean"]
    features = features[X.columns]
    return(features)
     
#path = "./full_processed/Val/normal/NORMAL2-IM-1427-0001.jpeg"
#img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)#

#y_p = check_for_pneumonia(img, "cnn")
#print(f"The probability is {y_p}")