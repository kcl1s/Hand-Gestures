import mediapipe as mp
import cv2
import time
import serial
import numpy as np
from  pycreate2 import Create2

class ardu:
    def start(port):
        ardu.sp = serial.Serial(port, 115200, timeout=0.1)
        ardu.ELtime = time.time()
        time.sleep(2)

    def ckRec():
        cmd= ' '
        intlistin=[0,]
        if (ardu.sp.in_waiting>0):
            rawin=ardu.sp.read_until()
            decodein=rawin.decode()
            if(decodein[0]=='E'):
                stripin=decodein[2:-2]
                listin=stripin.split(':')
                intlistin= [int(i) for i in listin]
                cmd=decodein[0]
        return cmd, intlistin

    def write(cmd):
        cmd = cmd.encode()
        ardu.sp.write(cmd)

    def echoloop(interval):
        if (time.time()-ardu.ELtime > interval):
            ardu.ELtime= time.time()
            ardu.write('E\n')

class rb:
    old_dir= ''
    def start(port):
        rb.SLtime= time.time()
        rb.b = Create2(port)
        rb.b.start()
        rb.b.safe()
        rb.b.drive_stop()

    def mv(dir,spd):
        if dir != rb.old_dir:
            rb.old_dir=dir
            mvsp = int(spd)
            print (dir,mvsp)
            if dir == 's':
                rb.b.drive_stop()
            if dir == 'f':
                rb.b.drive_direct(mvsp,mvsp)
            if dir == 'b':
                rb.b.drive_direct(mvsp*-1,mvsp*-1)
            if dir == 'l':
                rb.b.drive_direct(mvsp,mvsp*-1)
            if dir == 'r':
                rb.b.drive_direct(mvsp*-1,mvsp) 

    def sensorloop(interval):
        if (time.time()-rb.SLtime > interval):
            rb.SLtime= time.time()
            rb.sensorPkg = rb.b.get_sensors()
            print('Got Sensors')
        
class mpMesh:
    def start():
        mpMesh.Mesh=mp.solutions.face_mesh.FaceMesh(False,3,.5,.5)
        mpMesh.Draw=mp.solutions.drawing_utils

    def getMesh(frame,doDraw):
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,c=frame.shape
        results=mpMesh.Mesh.process(frameRGB)
        myMeshs=[]
        if results.multi_face_landmarks != None:
            for faceMesh in results.multi_face_landmarks:
                myMesh=[]
                if doDraw==True:
                    mpMesh.Draw.draw_landmarks(frame,faceMesh,mp.solutions.face_mesh.FACE_CONNECTIONS)
                for LM in faceMesh.landmark:
                    myMesh.append((int(LM.x*w),int(LM.y*h)))
                myMeshs.append(myMesh)
        return myMeshs
    
class mpFBB:
    def start():
        mpFBB.Face=mp.solutions.face_detection.FaceDetection()
        mpFBB.Draw=mp.solutions.drawing_utils

    def getFaceBB(frame,doDraw):
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,c=frame.shape
        results=mpFBB.Face.process(frameRGB)
        myFaceBBs=[]
        if results.detections != None:
            for face in results.detections:
                if doDraw==True:
                    mpFBB.Draw.draw_detection(frame,face)
                bBox=face.location_data.relative_bounding_box
                TLbb=(int(bBox.xmin*w),int(bBox.ymin*h))
                WHbb=(int(bBox.width*w),int(bBox.height*h))
                myFaceBBs.append((TLbb,WHbb))
        return myFaceBBs    #returns ((TLx,TLy),(width,height) for each face
                            #https://google.github.io/mediapipe/solutions/face_detection.html

class mpPose:
    def start():
        mpPose.Pose=mp.solutions.pose.Pose()
        mpPose.Draw=mp.solutions.drawing_utils

    def getPose(frame,doDraw):
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,c=frame.shape
        results=mpPose.Pose.process(frameRGB)
        myPose=[]
        if results.pose_landmarks:
            for plm in results.pose_landmarks.landmark:
                myPose.append((int(plm.x*w),int(plm.y*h)))
            if doDraw==True:
                mpPose.Draw.draw_landmarks(frame,results.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)
        return myPose   #returns list of 33 x,y tuples for 1 body
                        #https://google.github.io/mediapipe/solutions/pose.html

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

class cvColorTrack:
    firstClick=True
    #newClick=False
    def start():
       cvColorTrack.mHSV=np.zeros([1,1,3],dtype=np.uint8)

    def newClick(frame,x,y):
        mFrame = frame[y:y+1,x:x+1]
        cvColorTrack.mHSV=cv2.cvtColor(mFrame,cv2.COLOR_BGR2HSV)
        (h,s,v)=cvColorTrack.mHSV[0,0]
        cvColorTrack.Hmin=np.clip(h-15,0,179)
        cvColorTrack.Hmax=np.clip(h+15,0,179)
        cvColorTrack.Smin=np.clip(s-60,0,255)
        cvColorTrack.Smax=np.clip(s+60,0,255)
        cvColorTrack.Vmin=np.clip(v-60,0,255)
        cvColorTrack.Vmax=np.clip(v+60,0,255)
        cvColorTrack.firstClick=False

    def doColorTrack(frame):
        if cvColorTrack.firstClick==False:
            frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)           
            LB=np.array([cvColorTrack.Hmin,cvColorTrack.Smin,cvColorTrack.Vmin])                       
            UB=np.array([cvColorTrack.Hmax,cvColorTrack.Smax,cvColorTrack.Vmax])                      
            cMask=cv2.inRange(frameHSV,LB,UB)
            contours,junk=cv2.findContours(cMask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            return contours
            
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




