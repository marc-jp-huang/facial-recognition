import cv2
import sys
dataroot = "data/haarcascades/"
# Get user supplied values
args = dict([arg.split('=', 1) for arg in sys.argv[1:]])
imagePath = args.get('--file','abba.png')
cascPath = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_default.xml")

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier(dataroot+'haarcascade_eye.xml')
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    #openCV3 not support anymore
    flags=0
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(
    	    roi_gray,
            scaleFactor=1.1,
            minNeighbors=5
    	)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
