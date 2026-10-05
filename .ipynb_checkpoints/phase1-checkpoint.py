import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, canny
from skimage.filters import rank
from skimage.morphology import disk
from skimage.measure import find_contours 
from scipy.stats import entropy, skew, kurtosis
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

def import_images(path):
    path = Path(path)
    data = []

    for filepath in path.rglob("*"):
        if filepath.is_file():
            image_id = filepath.name
            
            # Get the directory components
            subdirs = [part.lower() for part in filepath.parent.parts]

            if "test" in subdirs:
                TTV = "test"
            elif "train" in subdirs:
                TTV = "train"
            elif "val" in subdirs:
                TTV = "val"
            else:
                TTV = "None"

            # Determine label
            if "NORMAL" in filepath.name or "normal" in subdirs:
                label = "NORMAL"
            elif "bacteria" in filepath.name or "virus" in filepath.name:
                label = "PNEUMONIA"
            else:
                label = None

            data.append([image_id, TTV, label, filepath])

    return pd.DataFrame(
        data,
        columns=["ImageID", "TTV", "Label", "Filepath"]
    )

def validate_dataset(df):
    good = [] #for the images that are good
    bad = [] #for the images that dont meet QC

    valid_labels = {"pneumonia", "normal"}
    for idx, row in df.iterrows():
        img_id = row["ImageID"]
        ttv = str(row["TTV"]).strip().lower()
        label = str(row["Label"]).strip().lower()
        filepath = Path(str(row["Filepath"]))

        #Check file exists 
        if not filepath.exists():
            bad.append({"ImageID": img_id, "Reason": "Missing File", "Filepath": filepath})
            continue
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        #Check if image is corrupted
        if img is None or img.size == 0:
            bad.append({"ImageID": img_id, "Reason": "Corrupted Image", "Filepath": filepath})
            continue
        #Check for Label validity
        if label not in valid_labels:
            bad.append({"ImageID": img_id, "Reason": "Invalid Label", "Filepath": filepath})
            continue
        #check resolution
        width, height = img.shape
        resolution = width*height
        #check brightness
        brightness = np.mean(img)
        contrast  = np.std(img)
        good.append({
            "ImageID": img_id,
            "TTV": ttv,
            "Label": label,
            "filepath": filepath,
            "width": width,
            "height": height,
            "resolution": height*width,
            "brightness": brightness,
            "contrast": contrast
        })
    good_df = pd.DataFrame(good)
    bad_df = pd.DataFrame(bad)
    return good_df, bad_df
    
def remove_outliers_resize_and_save(df_clean, df_dirty, size=(224,224), output_dir="./processed_images"): 
    good = []
    bad = []
    
    if len(df_clean) < 5:
        print("not enough data for outliers")
        return(df_clean, df_dirty)
    features = ["width", "height", "resolution", "contrast", "brightness"]
    bounds = {}
    for feature in features : 
        p25, p75 = np.percentile(df_clean[feature], [25, 75])
        iqr = p75-p25
        #print(feature, p25, p75, iqr)
        lower = p25 - 1.5*iqr
        upper = p75 + 1.5*iqr
        bounds.update({feature: [upper, lower]})
    for idx, row in df_clean.iterrows():
        img_id = row["ImageID"]
        ttv = str(row["TTV"])
        label = str(row["Label"])
        filepath = Path(str(row["filepath"]))
        for feature in features:
            if row[feature] > bounds[feature][0] or row[feature] < bounds[feature][1]:
                bad.append({"ImageID": img_id, "Reason": f"Image has an outlying {feature}: {row[feature]}", "Filepath": filepath})
                break;
        else:
            try: 
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
                new_filename = f"{img_id}"
                processed_dir = Path(output_dir) / ttv / label
                processed_dir.mkdir(parents=True, exist_ok=True)
                new_filepath = processed_dir / new_filename 
                cv2.imwrite(new_filepath, img_resized)
                #img_normalized.imwrite(new_filepath, format="JPEG", quality=95, optimize=True)
                good.append({
                    "ImageID": img_id,
                    "TTV": ttv,
                    "Label": label,
                    "Original_Filepath": filepath,
                    "Processed_Filepath": new_filepath
                })
            except Exception as e:
                bad.append({"ImageID": img_id, "Reason": f"Processing Failed ({str(e)})", "Filepath": filepath})
    df_good = pd.DataFrame(good)
    df_bad = pd.concat([df_dirty, pd.DataFrame(bad)], ignore_index=True, axis=0)
    return(df_good, df_bad) 

def inspect_samples(clean_df, num_samples=6):
    """Displays a grid sample of processed grayscale JPEG images."""
    if clean_df.empty:
        print("No valid images available to inspect.")
        return

    sample_df = clean_df.sample(min(num_samples, len(clean_df))).reset_index(drop=True)

    fig, axes = plt.subplots(int(num_samples/3), 3, figsize=(16, 8))
    axes = axes.flatten()

    for idx, row in sample_df.iterrows():
        img = cv2.imread(row["Processed_Filepath"],cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
        # Display using grayscale colormap
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f"ID: {row['ImageID']}\nLabel: {row['Label']}\nSet: {row['TTV']}", fontsize=9)
        axes[idx].axis("off")

    for i in range(len(sample_df), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def load_image(filepath, size=(1024,1024)):
    img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    if img.shape != size:
         raise ValueError(
            f"Expected {size}, got {img.shape}"
        )
    return(img)
    
def pixel_features(img, threshold=0.01):
    #Threshold is used to ignore the back blackground which dilutes the signal
    features = {}
    img = img.astype(np.float32) / 255.0
    foreground = img[img>threshold]
    flat_pixels = np.ravel(img)
    features["pixel_mean"] = np.mean(img)
    features["pixel_std"] = np.std(img)
    features["foreground_min"] = np.min(foreground)
    features["foreground_max"] = np.max(foreground)
    features["pixel_skew"] = skew(flat_pixels)      # Pos = more dark pixels, Neg = bright opacities
    features["pixle_kurt"] = kurtosis(flat_pixels)   # Peakedness/tail weight of intensity distribution
    
    # Key Percentiles (Intensity spread)
    p25, p50, p75 = np.percentile(flat_pixels, [25, 50, 75])
    features["pixel_iqr"] = float(p75 - p25)  # Interquartile range (spread of core pixel value
    features["pixel_mode"] = float(p50)
    return(features)
    
def texture_features(img):
    features = {}
    #img = img.astype(np.uint8)
    img = (img//8).astype(np.uint8)
    glcm = graycomatrix(
        img, 
        distances=[1,2,4], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
        #levels=256, 
        levels = 32,
        symmetric=True, 
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # Filter out zero probabilities for log calculation
    shannon_entropy = entropy(hist, base=2)
    
    features["contrast"] = float(np.mean(contrast))
    features["contrast_std"] = float(np.std(contrast))
    features["homogeneity"] = float(np.mean(homogeneity))
    features["energy"] = float(np.mean(energy))
    features["entropy"] = float(shannon_entropy)
    return(features)
    
def edge_features(img, blur=0):
    features = {}
    #img = img.astype(np.float32) / 255.0 #scale pixels to [0,1] scale
    if(blur):
        img = cv2.GaussianBlur(img, (3, 3), 0)
    # Detect edges using Canny
    edges = cv2.Canny(img, 30, 100) 
    edge_pixels = np.count_nonzero(edges)
    total_pixels = img.size  # 224 x 224 = 50,176
    edge_density = edge_pixels / total_pixels
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    features["edge_density"] = edge_density
    features["contour_count"] = contour_count
    return(features)
    
def region_features(img, sixths=0, threshold=0.01):
    #threshold helps ignore the background
    features={}
    img = img.astype(np.float32) / 255.0
    mask = img > threshold
    ys,xs = np.where(mask)
    # Crop to the non-black region
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    img = img[y1:y2, x1:x2]
    mask = mask[y1:y2, x1:x2]
    h,w = img.shape
    if w % 2 != 0: 
        img = img[:, :-1] 
        mask = mask[:, :-1]
    h,w = img.shape
    brightness = np.mean(img[mask])
    contrast = np.std(img[mask])
    
    y3 = h//3
    mid_x = w // 2
    
    if (sixths):
        regions = {
            "top_l":   img[:y3, :mid_x],
            "top_r":  img[:y3, mid_x:],
            "mid_l":  img[y3:2*y3, :mid_x],
            "mid_r": img[y3:2*y3, mid_x:],
            "bot_l":   img[2*y3:, :mid_x],
            "bot_r":  img[2*y3:, mid_x:]
        }
    else:
        regions = {
            "top":   img[:y3, :],
            "mid":  img[y3:2*y3, :],
            "bot":   img[2*y3:, :],
        }
    
    for name, region in regions.items():

        pixels = region[region > threshold]
        
        if len(pixels) == 0:
            continue
        features[f"brightness_mean_{name}"] = float(np.mean(pixels))
        features[f"is_bright_{name}"] = "is_bright" if(np.mean(pixels) > 0.65) else "is_dim"
        features[f"brightness_std_{name}"] = float(np.std(pixels))

    # 3. Split Image and Mask down the Center Axis
    left = img[:, :mid_x]
    right = img[:, mid_x:]
    
    left_mask = mask[:, :mid_x]
    right_mask = mask[:, mid_x:]

    # Flip right side to align anatomically with left side
    right_flipped = cv2.flip(right, 1)
    right_mask_flipped = cv2.flip(right_mask.astype(np.uint8), 1).astype(bool)

    # 4. Joint Mask: Only compute metrics where BOTH sides have body tissue
    valid = left_mask & right_mask_flipped
    
    # 5. Calculate Symmetry strictly on Valid Body Pixels
    left_pixels = left[valid]
    right_pixels = right_flipped[valid]
    # Absolute Difference & MSE
    diff = left_pixels - right_pixels
    mean_abs_diff = np.mean(diff)
    relative_mse = np.mean(diff ** 2)/(brightness**2 + 1e-8)

    #Normalise
    left_mean = np.mean(left_pixels)
    left_std = np.std(left_pixels)
    right_mean = np.mean(right_pixels)
    right_std = np.std(right_pixels)
    

    left_norm = ( left_pixels - left_mean ) / (left_std + 1e-8) 
    right_norm = ( right_pixels - right_mean ) / (right_std + 1e-8)
    correlation = np.corrcoef(left_norm, right_norm)[0, 1]
    correlation = np.clip( correlation, -1.0, 1.0 ) 
    # Convert [-1, 1] -> [0, 1] 
    symmetry_score = (correlation + 1.0) / 2.0
    features['LR_abs_diff']= mean_abs_diff
    features['LR_mse']= relative_mse
    features['LR_correlation']= correlation
    features["symmetry_score"] = symmetry_score
    features['sym_classification'] = "highly_symmetric" if symmetry_score >= 0.85 else "roughly_symmetric" if symmetry_score>=0.65 else "not_symmetric"
    
        
    return(features)

def feature_eng(df):
    feature_rows= []
    for idx, row in df.iterrows():
        img = load_image(row["Processed_Filepath"], size=(224,224))
        blur=True
        features = pixel_features(img) | \
        texture_features(img) | \
        edge_features(img) | \
        region_features(img, 0, 0.05)
        feature_rows.append(features)
    return(pd.concat([df, pd.DataFrame(feature_rows, index=df.index)], axis=1))     