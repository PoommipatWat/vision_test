import numpy as np

# ตัวอย่างค่า x และ y ที่เป็นไปได้
x = -7
y = 7

# คำนวณ arctan2 จาก NumPy
angle_numpy = np.arctan2(y, x)

# คำนวณ arctan จากเครื่องคิดเลข
angle_calculator = np.arctan(y / x)

# แสดงผลลัพธ์
print("NumPy arctan2:", np.degrees(angle_numpy), "degrees")
print("Calculator arctan:", np.degrees(angle_calculator), "degrees")
