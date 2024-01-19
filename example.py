from all import *


imc = image_color()
imp = image_process()
cls = classification()

# ตัวอย่าง GLCM
test = np.array([[255,250,152,80,11,5],
                [250,240,200,160,89,52],
                [253,230,180,122,40,2],
                [175,225,190,148,34,12],
                [120,152,205,123,88,15],
                [32,82,153,166,44,4],
                [17,11,9,78,0,2]])

glcm = (cls.GLCM_find(test, 4,0,1, nor='normalized', sym='symmetric'))
print(f"GLCM : \n{glcm[0]}")
print(f"Maximum probability : {glcm[1]}")
print(f"Entropy : {glcm[2]}")
print(f"Energy : {glcm[3]}")
print(f"Contrast : {glcm[4]}")
print(f"Homogeneity : {glcm[5]}")
print(f"Correlation : {glcm[6]}")

# ตัวอย่าง pdf
x1 = np.array([0.34,0.15,0.45,0.28,0.13])
x2 = np.array([0.19,0.3,0.45,0.25,0.35])
print(f"pdf : {cls.pdf_find(0.14, x1) * cls.pdf_find(0.37, x2)}")

