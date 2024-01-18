#หา background ของวิดีโอ และ ทำการ ตรวจจับวัตถุที่เคลื่อนไหว

import cv2

#Download 'ExampleBGSubtraction.avi' from https://drive.google.com/file/d/1OD_A0wqN2Om2SusCztybu-_hMSUQuRt7/view?usp=sharing

cap = cv2.VideoCapture('C:\\Users\\Poommipat\\Documents\\exam_comvision\\computer_vision-master\\ExampleBGSubtraction.avi')

haveFrame,bg = cap.read()

while(cap.isOpened()):
    haveFrame,im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    diffc = cv2.absdiff(im,bg)          #เอามาลบกันแล้วหา abs()
    diffg = cv2.cvtColor(diffc,cv2.COLOR_BGR2GRAY)      #เปลี่ยนเป็น gray
    bwmask = cv2.inRange(diffg,50,255)             #เอามาทำ binary โดยกำหนด threshold 50-255

    cv2.imshow('diffc', diffc)
    cv2.moveWindow('diffc',10,10)
    cv2.imshow('diffg',diffg)
    cv2.moveWindow('diffg', 400, 10)
    cv2.imshow('bwmask', bwmask)
    cv2.moveWindow('bwmask', 800, 10)

cap.release()
cv2.destroyAllWindows()
