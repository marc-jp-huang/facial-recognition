import cv2
import sys
dataroot = "data/haarcascades/"
# Get user supplied values
args = dict([arg.split('=', 1) for arg in sys.argv[1:]])
imagePath = args.get('--file','cat1.jpg')
cascPath = args.get('--cascade', "data/haarcascades/haarcascade_frontalcatface.xml")

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
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

cv2.imshow("Faces found", image)
cv2.waitKey(0)
