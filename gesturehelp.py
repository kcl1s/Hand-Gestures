import mediapipe as mp
import cv2
import time

class mpHand:
    def start():
        mpHand.Hands=mp.solutions.hands.Hands(False,1,.5,.5)
        mpHand.Draw=mp.solutions.drawing_utils

    def getLM(img,doDraw):
        frame=img
        h,w,c=frame.shape
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=mpHand.Hands.process(frameRGB)
        myHands=[]
        handsType=[]
        if results.multi_hand_landmarks != None:
            for hand in results.multi_handedness:
                handsType.append(hand.classification[0].label)
            for HLMs in results.multi_hand_landmarks:
                myHand=[]
                if doDraw==True:
                    mpHand.Draw.draw_landmarks(frame,HLMs,mp.solutions.hands.HAND_CONNECTIONS)
                for LM in HLMs.landmark:
                    myHand.append((int(LM.x*w),int(LM.y*h)))
                myHands.append(myHand)              
        return myHands,handsType    #returns 21 x,y tuples and Left,Right for each hand
                                    #https://google.github.io/mediapipe/solutions/hands.html

class TrackFPS:           #Weighted average (low pass) filter for frames per second
    def start(dataWeight):
        TrackFPS.dw=dataWeight
        TrackFPS.state=0
    
    def getFPS():
        if TrackFPS.state==0:
            TrackFPS.average=0
            TrackFPS.tlast=time.time()
            TrackFPS.state = 1
        elif TrackFPS.state==1:
            TrackFPS.tDelta=time.time()-TrackFPS.tlast
            TrackFPS.average=1/TrackFPS.tDelta
            TrackFPS.tlast=time.time()
            TrackFPS.state = 2            
        else:
            TrackFPS.tDelta=time.time()-TrackFPS.tlast
            TrackFPS.fps=1/TrackFPS.tDelta
            TrackFPS.average=(TrackFPS.dw * TrackFPS.fps)+((1 - TrackFPS.dw) * TrackFPS.average)
            TrackFPS.tlast=time.time()
        return TrackFPS.average