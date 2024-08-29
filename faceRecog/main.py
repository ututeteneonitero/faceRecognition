
import cv2

faceRef = cv2.CascadeClassifier('faceRef.xml')
camera = cv2.VideoCapture(0)

def faceDetection(frame):
    optimezed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceRef.detectMultiScale(optimezed_frame,scaleFactor=1.1, minSize=(500,500), minNeighbors=5)
    return faces

def drawer_box(frame):
    for x,y,w,h in faceDetection(frame):
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,198,121), 4)
    pass


def close_window():
    camera.release()
    cv2.destroyAllWindows()
    
    exit()

def main():
    while True :
        _,frame = camera.read()
        drawer_box(frame)
        cv2.imshow('face_recognition',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
            break
    
if __name__ == "__main__":
    main()