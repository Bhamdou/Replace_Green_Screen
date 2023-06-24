import cv2
import numpy as np

# Load the foreground image with greenscreen and background.
foreground = cv2.imread('greenscreen.jpg')
background = cv2.imread('background.jpg')

# Resize images to match sizes
foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

# Convert the images to HSV color space.
hsv_img = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

# Define a range for the "green" in the greenscreen.
lower_green = np.array([36, 0, 0])
upper_green = np.array([86, 255, 255])

# Create a mask for the greenscreen area.
mask = cv2.inRange(hsv_img, lower_green, upper_green)

# Smooth the mask with a Gaussian filter
mask = cv2.GaussianBlur(mask, (3, 3), 0)

# Use the mask to extract the greenscreen part of the foreground image.
extract = cv2.bitwise_and(foreground, foreground, mask=mask)

# Invert the mask, to get the part of the image that's not the greenscreen.
mask_inv = cv2.bitwise_not(mask)

# Extract the background where the greenscreen area should be.
background_part = cv2.bitwise_and(background, background, mask=mask)

# Replace the greenscreen part of the foreground with the corresponding part from the background.
final_img = cv2.bitwise_or(extract, background_part)

cv2.imshow('final', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
