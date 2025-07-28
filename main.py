'''
import json
import re
from statistics import median

file_path = "/mnt/data/_annotations.valid.jsonl"

all_boxes = []
x1_values = []

with open("C:/Users/jeetk/Downloads/scoliosis.v2i.openai/_annotations.valid.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        assistant_message = obj["messages"][-1]["content"]  # Last message contains the boxes

        # Extract bounding boxes like <locXXXX>
        raw_boxes = re.findall(r"<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)> spine", assistant_message)

        # Convert to int and store
        boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in raw_boxes]
        all_boxes.append(boxes)

        for x1, y1, x2, y2 in boxes:
                    x1_values.append(x1)

print(x1_values[:10])

# Show sample output
#for i, boxes in enumerate(all_boxes[:2]):
#    print(f"Image {i+1}: {boxes}")
'''

from roboflow import Roboflow
import json
import re
import numpy as np
import math

file_path = r"C:\Users\jeetk\Downloads\scoliosis.v2i.openai\_annotations.valid.jsonl"

def calculate_cobb_angle(boxes):
    if len(boxes) < 3:
        return None  # Not enough vertebrae to calculate curvature

    # Step 1: Get center points of each bounding box
    centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]

    # Step 2: Pick top 2 and bottom 2 points to form lines
    top = centers[:2]
    bottom = centers[-2:]

    # Step 3: Fit lines to top and bottom vertebrae
    def fit_line(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.atan2(dy, dx)

    angle_top = fit_line(*top)
    angle_bottom = fit_line(*bottom)

    # Step 4: Cobb angle is the acute angle between the two lines
    angle_diff_rad = abs(angle_top - angle_bottom)
    angle_diff_deg = math.degrees(angle_diff_rad)
    cobb_angle_deg = angle_diff_deg if angle_diff_deg <= 90 else 180 - angle_diff_deg

    return round(cobb_angle_deg, 2)



# Main parsing
cobb_angles = []

with open(file_path, "r") as f:
    for line in f:
        obj = json.loads(line)
        image_name = obj.get("image", "unknown")  # Get the image filename
        assistant_message = obj["messages"][-1]["content"]


        # Get bounding boxes
        raw_boxes = re.findall(r"<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)> spine", assistant_message)
        boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in raw_boxes]

        # Sort top to bottom by y1
        boxes = sorted(boxes, key=lambda b: b[1])

        cobb = calculate_cobb_angle(boxes)
        if cobb is not None:
            if cobb < 10:
                severity = "Normal"
            elif 10 <= cobb < 25:
                severity = "Mild"
            elif 25 <= cobb < 45:
                severity = "Moderate"
            else:
                severity = "Severe"

            cobb_angles.append((cobb, severity))

# Output result
print("Sample Cobb angles:")
for i, (angle, severity) in enumerate(cobb_angles):
    print(f"Image {i+1}: Cobb angle = {abs(angle)}° — Severity: {severity}")
