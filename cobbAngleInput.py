import requests
import math

# --------- CONFIGURATION ---------
# The following values should be set to match your Roboflow deployment.
API_KEY = "h9Nz8spjAJGNiHhshLda"              # Your Roboflow private API key (should be in env for production)
PROJECT = "scoliosis-dgqy2-6hnes"             # Project slug from the Roboflow dashboard URL
MODEL_VERSION = 1                             # Model version number (update to match your deployment)
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{MODEL_VERSION}"  # Full inference endpoint

# --------- GEOMETRY: Utility to fit a line through two points (returns angle in radians) ---------
def fit_line(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx)

# --------- Cobb Angle Calculation Logic ---------
def calculate_cobb_angle(boxes):
    """
    Finds the Cobb angle by computing the angle between lines
    (top 2 centers and bottom 2 centers) from bounding boxes.
    Requires at least 3 bounding boxes.
    """
    if len(boxes) < 3:
        return None
    # Compute center points for each detected vertebral box
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
    top, bottom = centers[:2], centers[-2:]   # First 2 and last 2 centers
    # Get angles (in radians)
    angle_top = fit_line(*top)
    angle_bottom = fit_line(*bottom)
    # Find absolute difference (in degrees)
    angle_diff = abs(math.degrees(angle_top - angle_bottom))
    # Normalize Cobb angle to <= 90 degrees
    cobb_angle = angle_diff if angle_diff <= 90 else 180 - angle_diff
    return round(abs(cobb_angle), 2)          # Return positive, rounded to 2 decimals

# --------- Classification of Scoliosis Severity ---------
def classify_severity(angle):
    """
    Classifies scoliosis cases by Cobb angle value.
    """
    if angle < 10:
        return "Normal"
    elif 10 <= angle < 25:
        return "Mild"
    elif 25 <= angle < 45:
        return "Moderate"
    else:
        return "Severe"

# --------- Bounding Box Extraction from Roboflow JSON Response ---------
def extract_boxes_from_predictions(predictions_json):
    """
    Converts Roboflow 'predictions' objects to (x1, y1, x2, y2) coordinates for boxes
    Only keeps results classified as 'spine'.
    """
    boxes = []
    for pred in predictions_json:
        if pred.get("class") == "spine":       # Only process objects classified as spine
            x, y = pred["x"], pred["y"]        # Centers
            w, h = pred["width"], pred["height"]
            # Convert center x/y and w/h to (x1, y1) (top-left) and (x2, y2) (bottom-right)
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
            boxes.append((x1, y1, x2, y2))
    return boxes

# --------- Main Inference and Cobb Angle Pipeline ---------
def analyze_image(file_obj):
    """
    Main entry point.
    Given a file-like object (from Streamlit upload), this function:
    - Sends to Roboflow's hosted model
    - Extracts bounding boxes for spines
    - Calculates Cobb angle and severity
    - Returns (angle, severity) if successful, or (None, error-message) on failure.
    """
    file_obj.seek(0)  # Ensure we're at the start of the file for upload
    files = {"file": (getattr(file_obj, "name", "image.jpg"), file_obj.read())}
    params = {"api_key": API_KEY}

    # ---------- DEBUG INSTRUMENTATION ----------
    print(f"POST {DETECT_URL}")                   # Print the endpoint being used
    print(f"Params: {params}")                    # Show any parameters, especially api_key
    print(f"Files: {list(files.keys())}, Name: {files['file'][0]}")  # File details
    # -------------------------------------------

    # ---- Roboflow Hosted Inference ----
    response = requests.post(DETECT_URL, params=params, files=files)

    # --------- More Debug Output ---------
    print(f"HTTP Status: {response.status_code}")
    print("Raw Response Text:", response.text)
    try:
        response_json = response.json()
        print("Parsed Response JSON:", response_json)
    except Exception as e:
        print("Failed to decode JSON. Error:", e)
        return None, f"Inference response not JSON: {e}"

    # Extract all predicted spine boxes
    predictions = response_json.get("predictions", [])
    boxes = extract_boxes_from_predictions(predictions)
    print(f"Extracted {len(boxes)} spine boxes: {boxes}")

    # Quality checks
    if not boxes:
        return None, "No spine boxes detected."
    boxes.sort(key=lambda b: b[1])  # Sort boxes by vertical position (top-to-bottom)

    # Cobb angle computation and severity class
    cobb_angle = calculate_cobb_angle(boxes)
    if cobb_angle is None:
        return None, "Not enough vertebrae detected to calculate Cobb angle."
    severity = classify_severity(cobb_angle)
    print(f"Result: Cobb Angle = {cobb_angle}°, Severity = {severity}")

    return cobb_angle, severity
