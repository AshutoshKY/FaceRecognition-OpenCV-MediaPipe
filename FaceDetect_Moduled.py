import cv2
import mediapipe as mp
import time

## Class for major Functions and easy Call
class FaceDetector():
    
    ## Defines Outline and Utilites and Confidence Parameters
    def __init__(self, minDetectCon = 0.75):
        
        self.minDetectCon = minDetectCon
        
        ## MediaPipe Functions/Classes
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectCon)
     
    ## To find Faces and Draw Rectangle by taking data from
    ## faceDetection utility inside which results.detection    
    def findFaces(self, img, draw=True):
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceDetection.process(imgRGB)
            #print(self.results)
            
            ##list
            boundBoxes = []
            
            if self.results.detections:
                for id,detection in enumerate(self.results.detections):
                    ## Prints id(1 if one people,1 if two people detected and so on)
                    ## Normalized values btw 0 and 1
                        # print(id, detection)
                    
                    ## Prints Detection-Score(% that a face is found) 
                        # print(detection.score)
                    ## Prints x,y position of detected Face
                        # print(detection.location_data.relative_bounding_box)
                    
                    ## Draws Square and Points Key features using built-in functions
                        # mpDraw.draw_detection(img,detection)
                    
                    faceDetectBoxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = img.shape
                    boundbox = int(faceDetectBoxC.xmin * iw), int(faceDetectBoxC.ymin * ih), \
                            int(faceDetectBoxC.width * iw), int(faceDetectBoxC.height * ih)
                    # if detection.score > 0.75:
                    boundBoxes.append([id,boundbox, detection.score])
                    if draw:
                        self.customDraw(img,boundbox)
                        cv2.putText(img, f'{int(detection.score[0]*100)}%',
                                (boundbox[0], boundbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255,0,255), 2)
            return img, boundBoxes
        
    ## To Draw Custom Rectangel around Face   
    def customDraw(self,img,boundbox, l=30, thick=3, rtthick=1):
        x,y,w,h = boundbox
        x1,y1 = x+w, y+h
        
        cv2.rectangle(img, boundbox, (255,0,255), rtthick)
        #Top_left x,y
        cv2.line(img, (x,y), (x+l,y), (255,0,255), thick)
        cv2.line(img, (x,y), (x,y+l), (255,0,255), thick)
        #Top-Right x1,y
        cv2.line(img, (x1,y), (x1-l,y), (255,0,255), thick)
        cv2.line(img, (x1,y), (x1,y+l), (255,0,255), thick)
        #Bottom-Left x,y1
        cv2.line(img, (x,y1), (x+l,y1), (255,0,255), thick)
        cv2.line(img, (x,y1), (x,y1-l), (255,0,255), thick)
        #Bottom-Right x1,y1
        cv2.line(img, (x1,y1), (x1-l,y1), (255,0,255), thick)
        cv2.line(img, (x1,y1), (x1,y1-l), (255,0,255), thick)
        
        return img

## Main Function Defined
def main():
    
    ## Captures Video from Source, 0 for camera, or src
    ## uses cap to gather data from cv2 video Capture
    cap = cv2.VideoCapture("./videos/vid4-2.mp4")
    # cap = cv2.VideoCapture(0)
    pTime=0 
    
    ## Creating a Object of class FaceDetector
    detector = FaceDetector()
    
    while True:
        ## cap Returns 2 Things bool and data
        ## success/error flag, and data from src which gets stored in img
        success, img = cap.read()
        
        ## Calling findFaced function from FaceDetector class
        img, boundboxes = detector.findFaces(img)
        if not success: break
        # print(boundboxes)
        
        ## FPS Calculation, cTime=current time, pTime = previous time
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        
        ## cv.putText(img, text, org-xy plane, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]	) -> img
        ## Defins what and how the text to be rendered in the Output Window
        cv2.putText(img, f'FPS: {int(fps)}',(10,50), cv2.FONT_HERSHEY_PLAIN,
                    2,(0,255,0), 2)
        
        ## imshow has 2 parameters - window_name, image
        ## window_name is string which represents the image to be displayed(Name of the windows which ops up for diaplying img)
        ## image is the image to be displayed
        cv2.imshow("Cam_Image", img)
        
        ## Waitkey(0) shows only a Still image untill key press
        ## while waitkey(1) will show a img for 1ms until closed
        ## as while loop is running so endless until src ends
        cv2.waitKey(1)
    
   
#Main function Called  
if __name__ == "__main__" :
    main()
    
## Comments for Self-Explanation

## Thanks to YouTubers, mainly Murtaza's Workshop