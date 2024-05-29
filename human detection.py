import cv2

cap = cv2.VideoCapture('imagensTeste/feras.mp4')


human_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_fullbody.xml')
human_head_cascade = cv2.CascadeClassifier('xmlFiles/haarcascade_frontalface_default.xml')


while(True):
    # Captura o frame-by-frame
    ret, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_head_cascade.detectMultiScale(gray, 1.2, 5)
    
    # Resultado do frame
    for (x,y,w,h) in humans:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         
         

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
