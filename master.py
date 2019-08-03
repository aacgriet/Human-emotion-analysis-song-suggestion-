import numpy
import cv2
import pygame
import time
from sklearn.externals import joblib
pygame.mixer.init()
alg=joblib.load('model.pkl')
p=cv2.VideoCapture(0)
ret,a=p.read()
while ret:
    ret,a=p.read()
    
    a1=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    
    fd=cv2.CascadeClassifier(r"C:\Users\user1\Desktop\aiml\haarcascade_frontalface_default.xml")
    
    
    img2=a1.copy()
    
    faces=fd.detectMultiScale(a,1.3,5)
    if len(faces)==0:
        print('none')
    else:
        (x,y,w,h)=faces[0]
        cv2.rectangle(a,(x,y),(x+w,y+h),[0,0,255],3)
        img2=img2[y:y+h,x:x+w]
        ig3=img2/(img2.max())
        ig4=ig3.astype(numpy.float32)
        ig5=cv2.resize(ig4,(250,250))
        ig5=ig5.reshape(1,-1)
        y1=alg.predict(ig5)
        print(y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if(y1=='angry'):
            cv2.putText(a,'happy',(90,50), font, 2,(0,255,255),2,cv2.LINE_AA)
        elif(y1=='fear'):
            cv2.putText(a,'happy',(90,50), font, 2,(0,255,255),2,cv2.LINE_AA)
        elif(y1=='happy'):
            cv2.putText(a,'happy',(90,50), font, 2,(0,255,255),2,cv2.LINE_AA)
        elif(y1=='sad'):
            cv2.putText(a,'happy',(90,50), font, 2,(0,255,255),2,cv2.LINE_AA)
        else:
            cv2.putText(a,'surprised',(90,50),font,2,(0,0,255),2,cv2.LINE_AA)
            
       
        if(y1=='angry'):
            pygame.mixer.music.load(r"C:\Users\TEST01\Downloads\angry.mp3")
            pygame.mixer.music.play()
            pygame.mixer.music.fadeout(90000)
            time.sleep(90)
            #time can be changed according to our wish
        elif(y1=='fear'):
            pygame.mixer.music.load(r"C:\Users\TEST01\Downloads\fear.mp3")
            pygame.mixer.music.play()
            pygame.mixer.music.fadeout(90000)
            time.sleep(90)
        elif(y1=='happy'):
            pygame.mixer.music.load(r"C:\Users\TEST01\Downloads\happy.mp3")
            pygame.mixer.music.play()
            pygame.mixer.music.fadeout(90000)
            time.sleep(90)
        elif(y1=='sad'):
            pygame.mixer.music.load(r"C:\Users\TEST01\Downloads\sad.mp3")
            pygame.mixer.music.play()
            pygame.mixer.music.fadeout(90000)
            time.sleep(90)
        else:
            pygame.mixer.music.load(r"C:\Users\TEST01\Downloads\surprised.mp3")
            pygame.mixer.music.play()
            pygame.mixer.music.fadeout(90000)
            time.sleep(90)
    cv2.imshow('img',a)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):# Hit `q` to exit
        break
    
p.release()
cv2.waitKey(0) 

cv2.destroyAllWindows()