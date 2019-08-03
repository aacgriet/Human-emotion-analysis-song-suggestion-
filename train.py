
import cv2
import os
import numpy
mainfolder=r"C:\Users\TEST01\Desktop\Datasets"
def face_detect(img):
fd=cv2.CascadeClassifier(r"C:\Users\TEST01\Desktop\haarcascade_frontalface_default.xml")
   
    faces=fd.detectMultiScale(img,1.3,5)
    if len(faces)==0:
        return None
    img2=img.copy()
    (x,y,w,h)=faces[0]
    img2=img[y:y+h,x:x+w]
    return img2
def load_data(mainfolder):
    folders=os.listdir(mainfolder)
    angry_path=mainfolder+'\\'+folders[0]
    fear_path=mainfolder+'\\'+folders[1]
    happy_path=mainfolder+'\\'+folders[2]
    sad_path=mainfolder+'\\'+folders[3]
    surprised_path=mainfolder+'\\'+folders[4]
    angerfiles=os.listdir(angry_path)
    fearfiles=os.listdir(fear_path)
    happyfiles=os.listdir(happy_path)
    sadfiles=os.listdir(sad_path)
    surprisedfiles=os.listdir(surprised_path)
    faces=[]
    labels=[]
for file in angerfiles:
        img=cv2.imread(angry_path+'\\'+file,-1)
        face=face_detect(img)
        if face is not None:
            labels.append(0)#angry
            img3=face/(face.max())
            img4=img3.astype(numpy.float32)
            img5=cv2.resize(img4,(250,250))
            img5=numpy.ndarray.flatten(img5)
            faces.append(img5)
            
    for file in fearfiles:
        img=cv2.imread(fear_path+'\\'+file,-1)
        face=face_detect(img)
        if face is not None:
            labels.append(1)#fear
            img16=face/(face.max())
            img17=img16.astype(numpy.float32)
            img18=cv2.resize(img17,(250,250))
            img18=numpy.ndarray.flatten(img18)
            faces.append(img18)
            
    for file in happyfiles:
        img=cv2.imread(happy_path+'\\'+file,-1)
        face=face_detect(img)
        if face is not None:
            labels.append(2)#happy
            img10=face/(face.max())
            img11=img10.astype(numpy.float32)
            img12=cv2.resize(img11,(250,250))
            img12=numpy.ndarray.flatten(img12)
            faces.append(img12)
            
    for file in sadfiles:
        img=cv2.imread(sad_path+'\\'+file,-1)
        face=face_detect(img)
        if face is not None:
            labels.append(3)#sad
            img13=face/(face.max())
            img14=img13.astype(numpy.float32)
            img15=cv2.resize(img14,(250,250))
            img15=numpy.ndarray.flatten(img15)
            faces.append(img15)
            
    for file in surprisedfiles:
        img=cv2.imread(surprised_path+'\\'+file,-1)
        face=face_detect(img)
        if face is not None:
            labels.append(4)#surp
            img6=face/(face.max())
            img7=img6.astype(numpy.float32)
            img8=cv2.resize(img7,(250,250))
            img9=numpy.ndarray.flatten(img8)
            faces.append(img9)
    return faces,labels
faces,labels=load_data(mainfolder)
faces=numpy.array(faces)

   for i in range(len(labels)):
    if labels[i]=='angry':
        labels[i]=0
    elif labels[i]=='fear':
        labels[i]=1

    elif labels[i]=='happy':
        labels[i]=2
    elif labels[i]=='sad':
        labels[i]=3
    elif labels[i]=='surprised':
        labels[i]=4


from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts=train_test_split(faces,labels,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
alg=KNeighborsClassifier(n_neighbors=5)

alg.fit(xtr,ytr)

accuracy=alg.score(xts,yts)
print(accuracy)

accuracy1=alg.score(xtr,ytr)
print(accuracy1)

from sklearn.externals import joblib
joblib.dump(alg,'model.pkl')

