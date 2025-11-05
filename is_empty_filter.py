import cv2, numpy as np, joblib

# Load the pre-trained classifier
_model = joblib.load("data/cache/empty_filter_forest.pkl")

def is_empty(img_path: str, prob_thresh: float = 0.7) -> bool:
    """Return True if the image likely contains no ships."""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    ratio = np.sum(edges > 0) / edges.size
    variance = np.var(gray / 255.0)
    features = np.array([[ratio, variance]])
    prob_ship = _model.predict_proba(features)[0][1]
    return prob_ship < prob_thresh
