# การเปิดภาพจากกล้องและแสดงผลเบื้องต้น

import cv2
print(cv2.__version__)

#เปิดกล้อง
cap = cv2.VideoCapture(0)

#กำหนดขนาดภาพ
CAP_SIZE = (1280,720)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_SIZE[0])
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_SIZE[1])

#อ่านค่า return im=ภาพที่อ่านได้ ret=ค่าความสำเร็จ
ret,im = cap.read()

# ขนาดภาพ และ dimension เช่น 3 คือ rgb
print(im.shape)
print(type(im))

# แสดงค่าสีของ pixel ตำแหน่งที่ 0,0 เป็น BGR
print(im[0,0])

# แสดงค่าสีของ pixel ตำแหน่งที่ 0,0 และ channel ที่ 0 คือ blue
print(im[0,0,0])

#ประเภทสีแบบ uint8 คือ 0-255 ต้องระวังการคำนวณ Overflow
print(type(im[0,0,0]))

#แสดงภาพ
cv2.imshow('camera',im)

#แสดงแต่ละ channel สี
#cv2.imshow('blue channel',im[:,:,0])      เอาทุกแถว, เอาทุกหลัก, เอาแค่ channel ที่ 0
#cv2.imshow('green channel',im[:,:,1])     เอาทุกแถว, เอาทุกหลัก, เอาแค่ channel ที่ 1
#cv2.imshow('red channel',im[:,:,2])       เอาทุกแถว, เอาทุกหลัก, เอาแค่ channel ที่ 2

#รอภาพไม่ให้ถูกปิด
cv2.waitKey()

#เลิกการเชื่อมต่อกล้อง เพื่อไม่ให้ภาพค้าง
cap.release()
