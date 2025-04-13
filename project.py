import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image using OpenCV
image =cv2.imread('image2.png')
# Convert the image from BGR to RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
# Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Display the original image using Matplotlib
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.axis('off')
plt.title('Original Image')
# Display the grayscale image using Matplotlib
plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
# Show the plot
plt.show()
# Save the grayscale image using OpenCV
cv2.imwrite('image_gray.jpg', image_gray)
# Reduce noise in the grayscale image using Gaussian blur
image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
# Apply thresholding to the blurred image
ret, image_thresh = cv2.threshold(image_blurred, 127, 255, cv2.THRESH_BINARY)
# Apply dilation to the thresholded image
kernel = np.ones((5, 5), np.uint8)
image_dilated = cv2.dilate(image_thresh, kernel, iterations=1)
plt.subplot(1, 2, 1)
plt.imshow(image_thresh, cmap='gray')
plt.axis('off')
plt.title('Thresholded Image')
plt.subplot(1, 2, 2)
plt.imshow(image_dilated, cmap='gray')
plt.axis('off')
plt.title('Dilated Image')
plt.show()
# Find contours in the dilated image
contours, hierarchy = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_contours = image.copy()
# Draw contours on the original image
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 3)
# Afficher le nombre de contours trouvés
print(f'Number of contours found: {len(contours)}')
plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Contours')
plt.show()
cv2.imwrite('image_contours.jpg', image_contours)




