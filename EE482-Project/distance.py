from ultralytics import YOLO
import os

# Load a model
model = YOLO('best.pt')  # Make sure this points to the correct model file

# chose one of the three methods [1,2,3]
method = 3

# Known reference dimensions and distance
reference_width_in_pixels_1 = 1008.1542  # The width of the object in pixels in your reference image
reference_distance_1 = 18  # The distance from the camera to the object in the reference image in centimeters
reference_width_in_pixels_2 = 583.8926  # The width of the object in pixels in your reference image
reference_distance_2 = 35  # The distance from the camera to the object in the reference image in centimeters
reference_width_in_pixels_3 = 349.0002  # The width of the object in pixels in your reference image
reference_distance_3 = 60  # The distance from the camera to the object in the reference image in centimeters

# Perform prediction
results = model(source='photos', conf=0.75 , save =True)

# Calculate and print estimated distance for each detection
for result in results:
    file_name = os.path.basename(result.path)  # Extract the base file name
    for box in result.boxes.xyxy:
        if box.numel() == 4:
            x1, y1, x2, y2 = box.tolist()  # Unpack the tensor to individual coordinates
            detected_width_in_pixels = x2 - x1
            detected_Tall_in_pixels = y2 - y1
            
            # First method
            if (method == 1):
                # Apply the formula
                 estimated_distance = (reference_width_in_pixels_2 * reference_distance_2) / detected_width_in_pixels
            
            # Second method
            if (method == 2):
                # Apply the formula
                estimated_distance1 = (reference_width_in_pixels_1 * reference_distance_1) / detected_width_in_pixels
                estimated_distance2 = (reference_width_in_pixels_2 * reference_distance_2) / detected_width_in_pixels
                estimated_distance3 = (reference_width_in_pixels_3 * reference_distance_3) / detected_width_in_pixels
                #average 
                estimated_distance = ( estimated_distance3 + estimated_distance2 + estimated_distance3)/(3)
            
            # Third method
            if (method == 3):
                # Apply the formula
                estimated_distance = ( detected_Tall_in_pixels*detected_Tall_in_pixels * (0.0000424514)) + (detected_Tall_in_pixels*(-0.1247301856))  + (114.4322942610)

            # Print out the file name and the estimated distance
            print(f"{file_name}: Estimated distance: {estimated_distance:.2f} cm")
            
        else:
            print("Unexpected box dimensions:", box)