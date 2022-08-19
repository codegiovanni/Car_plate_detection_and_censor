import cv2
import numpy as np

# Read the image
folder = 'images/'
file_name = 'image0.jpg'
full_path = folder + file_name
img = cv2.imread(full_path)
# cv2.imshow('Image', img)

# Grayscale the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', gray)

# Apply filter and find edges
bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(bfilter, 30, 200)
# cv2.imshow('Edged', edged)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Find number plate
roi = None
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
    if len(approx) == 4:
        roi = approx
        break

roi = np.array([roi], np.int32)
points = roi.reshape(4, 2)
x, y = np.split(points, [-1], axis=1)

(x1, x2) = (np.min(x), np.max(x))
(y1, y2) = (np.min(y), np.max(y))
number_plate = img[y1:y2, x1:x2]

# cv2.imshow('Number plate', number_plate)

# Blur the image
blurred_img = cv2.GaussianBlur(img, (51, 51), 30)
# cv2.imshow('Blurred', blurred_img)

# Create a mask for ROI and fill the ROI with white color
mask = np.zeros(img.shape, np.uint8)
cv2.fillPoly(mask, roi, (255, 255, 255))
# cv2.imshow('Mask', mask)

# Create a mask for everywhere except ROI and fill with white color
mask_inverse = np.ones(mask.shape).astype(np.uint8) * 255 - mask
# cv2.imshow('Inverse mask', mask_inverse)

# Combine all the masks and images
result = cv2.bitwise_and(blurred_img, mask) + cv2.bitwise_and(img, mask_inverse)
# cv2.imshow('result', result)

# Save and open the image
cv2.imwrite('output/' + str(file_name[:-4]) + '_censored.jpg', result)
print('Successfully saved')
saved = cv2.imread('output/' + str(file_name[:-4]) + '_censored.jpg')
cv2.imshow('Saved', saved)

if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
