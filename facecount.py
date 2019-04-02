import cv2

faceClassifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

objImage=cv2.imread('2person.jpg')
cvtImage=cv2.cvtColor(objImage,cv2.COLOR_BGR2GRAY)
foundFaces=faceClassifier.detectMultiScale(cvtImage,scaleFactor=1.3,minNeighbors=2,minSize=(50,50))
for (x,y,w,h) in foundFaces:
    cv2.rectangle(objImage,(x,y),(x+w,y+h),(0,0,255),2)
detectResult=str(len(foundFaces))+' faces detected'
cv2.imshow(detectResult, objImage)
cv2.waitKey(0)
