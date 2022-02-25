import cv2
print(cv2.__version__)
import numpy as np
import gesturehelp as cvh
import PySimpleGUI as sg
keyDist=((0,4),(0,8),(0,12),(0,16),(0,20),(4,8),(4,12),(4,16),(4,20),(8,12),
            (8,16),(8,20),(12,16),(12,20),(16,20),(0,17))
curE=[10000]*10
tolThresh=400
knownG= False
width=640
height=480
cvh.TrackFPS.start(.05)
cvh.mpHand.start()
sg.user_settings_filename(path='.')     # The settings file will be in the program folder programName.json
DMlist=sg.user_settings_get_entry('-DMs-') #If setting json file available import else use blank
if DMlist != None:
    DMs=np.array(DMlist)
else:
    DMs=np.zeros([10,16],dtype='int')
gNames=sg.user_settings_get_entry('-gNames-',[])
if gNames == []:
    gNames=['']*10

def jointDistance(hand):
    global keyDist
    global DMs
    for x in range(len(keyDist)):
        DMs[0][x]=int(np.sqrt((hand[keyDist[x][0]][0]-hand[keyDist[x][1]][0])**2 + 
                              (hand[keyDist[x][0]][1]-hand[keyDist[x][1]][1])**2))

def findError(knownGesture):
    global DMs
    error=0
    for i in range(15):
        error+= int(abs(DMs[0][i]*100/DMs[0][15] - DMs[knownGesture][i]*100/DMs[knownGesture][15]))
    return error

sg.theme('DarkGreen5')
Icam= sg.Image(filename='',size=(width,height),pad=0,enable_events= True, k='Icam')
Dlayout= [[sg.T('Unknown',font='Times 32',k='Tcur')],[sg.T('Gesture Names     (Check to Train)   ', font=16)]]
Dlayout+= [[sg.In(default_text =gNames[i],s=20,font=16,k='gesture'+str(i)),
            sg.CB('     ',font=16,enable_events=True,k='train'+str(i))] for i in range(1,10)]
layout= [[Icam,sg.Column(Dlayout)],
        [sg.Quit(font=16,size=6)]]
window=sg.Window('cv Gestures', layout,grab_anywhere_using_control = True,finalize=True)
for i in range(1,10): 
    if gNames[i] != '':
        window['train'+str(i)].update(value=True)
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)

while True:
    event, values = window.read(10)
    ignore,  frame = cam.read()
    hand,handLR=cvh.mpHand.getLM(frame,True)
    if hand:
        jointDistance(hand[0])
        if event.startswith('train') and values[event]:
            DMs[int(event[-1:])]=DMs[0]
            print (DMs[int(event[-1:])])
        else:
            for i in range(1,10):
                if values['train'+str(i)]:
                    curE[i]=findError(i)
                    window['train'+str(i)].update(text=curE[i])
                else:
                    window['train'+str(i)].update(text='')
            if min(curE)<tolThresh:
                inNum=curE.index(min(curE))
                inText=values['gesture'+str(inNum)]
            else:
                inText='Unknown' 
            window['Tcur'].update(value=inText)
            
    cv2.putText(frame,str(int(cvh.TrackFPS.getFPS())).rjust(3)+str(' FPS'),(0,50),
                cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),3)    
    if event in (sg.WIN_CLOSED, 'Quit'):
        for i in range(1,10):
            gNames[i]=window['gesture'+str(i)].get()
        sg.user_settings_set_entry('-gNames-',gNames)
        sg.user_settings_set_entry('-DMs-',DMs.tolist())
        break
    window['Icam'].update(data=cv2.imencode('.ppm', frame)[1].tobytes())
    
cam.release()
window.close()