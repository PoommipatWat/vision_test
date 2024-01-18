#ทดลองอ่านวีดีโอทั่วไปแบบไม่สน fps

import cv2

#Download 'ExampleBGSubtraction.avi' from https://drive.google.com/file/d/1OD_A0wqN2Om2SusCztybu-_hMSUQuRt7/view?usp=sharing

cap = cv2.VideoCapture('C:\\Users\\Poommipat\\Documents\\exam_comvision\\computer_vision-master\\ExampleBGSubtraction.avi')

while(cap.isOpened()):
    haveFrame, im = cap.read()

    if (not haveFrame) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    cv2.imshow('video',im)

cap.release()
cv2.destroyAllWindows()
