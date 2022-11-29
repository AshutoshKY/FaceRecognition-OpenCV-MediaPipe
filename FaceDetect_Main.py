import cv2
import mediapipe as mp
import time

## Captures Video from Source, 0 for camera, or src
## uses cap to gather data from cv2 video Capture
cap = cv2.VideoCapture("./videos/vid2.mp4")
# cap = cv2.VideoCapture(0)

pTime=0

## MediaPipe Functions/Classes
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

## cap Returns 2 Things bool and data
## success/error flag, and data from src which gets stored in img
while True:
    success, img = cap.read()
    if not success: break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
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
            cv2.rectangle(img, boundbox, (255,0,255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%',
                        (boundbox[0], boundbox[1]-20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255,0,255), 2)
    
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