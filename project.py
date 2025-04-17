import cv2
import numpy as np
import matplotlib.pyplot as plt
#charger l'image
img = cv2.imread('image1.jpg')
#convertir l'image en rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#convertir l'image en niveaux de gris
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#appliquer un flou gaussien pour réduire le bruit et lisser l'image
img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
#appliquer un seuillage (Otsu) pour binariser l'image
_, img_thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Light erosion to reduce noise
kernel = np.ones((5, 5), np.uint8)
img_gray = cv2.erode(img_gray, kernel, iterations=4)
# Morphological closing to fill small holes inside objects
closing = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
# ---- Start Watershed separation steps ----
# Distance transform
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
# Normalize for visualization (optional)
dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Threshold to obtain sure foreground regions
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Get sure background by dilating the closing result
sure_bg = cv2.dilate(closing, kernel, iterations=3)

# Unknown region (possible borders between coins)
unknown = cv2.subtract(sure_bg, sure_fg)

# Label the foreground markers
_, markers = cv2.connectedComponents(sure_fg)

# Add 1 to all labels so background is not 0
markers = markers + 1

# Mark the unknown region as 0
markers[unknown == 255] = 0

# Apply Watershed
img_ws = img.copy()
markers = cv2.watershed(cv2.cvtColor(img_ws, cv2.COLOR_RGB2BGR), markers)

# Draw boundaries (marked with -1 by watershed)
img_ws[markers == -1] = [255, 0, 0]  # Red borders

# Count detected objects (excluding background and boundary)
num_objects = len(np.unique(markers)) - 2  # -1 for background and -1 for boundary
# Count and label each coin
num_objects = 0
img_labeled = img.copy()
for i in range(2, markers.max() + 1):
    mask = np.uint8(markers == i)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            num_objects += 1
            cv2.putText(img_labeled, str(num_objects), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.drawContours(img_labeled, [cnt], -1, (0, 255, 0), 1)  # Draw contours in green
# ---- Display results ----

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.subplot(2, 3, 3)
plt.imshow(closing, cmap='gray')
plt.title('After Closing')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(dist_norm, cmap='jet')
plt.title('Distance Transform')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_ws)
plt.title('Watershed Result')
plt.axis('off')
plt.subplot(2, 3, 6)
plt.imshow(img_labeled)
plt.title(' Result')
plt.axis('off')
plt.tight_layout()
plt.figtext(0.5, 0.01, f"Detection complete: {num_objects} object found", wrap=True, horizontalalignment='center', fontsize=14, color='green')
plt.show()