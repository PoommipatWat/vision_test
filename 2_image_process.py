import numpy as np

def linear_shift_invariance(input_signal, filter_coefficients):
    filter_coefficients = filter_coefficients[::-1,::-1]

    output = np.zeros((len(input_signal), len(input_signal[0])))

    for i in range(len(input_signal)):
        for j in range(len(input_signal[0])):
            for m, k in enumerate(range(i-1, i+2)):
                for n, l in enumerate(range(j-1, j+2)):
                    if k >= 0 and l >= 0 and k < len(input_signal) and l < len(input_signal[0]):
                        output[i,j] += input_signal[k,l] * filter_coefficients[m,n]     
    return output

def derivative_first_x(input_signal):
    filter_coefficients = np.array([[0,1,0],
                                    [0,-1,0],
                                    [0,0,0]], dtype=np.float64)
    out = linear_shift_invariance(input_signal, filter_coefficients)
    out[-1] = np.zeros(len(input_signal[0]))
    return out

def derivative_first_y(input_signal):
    filter_coefficients = np.array([[0,0,0],
                                    [1,-1,0],
                                    [0,0,0]], dtype=np.float64)
    out = linear_shift_invariance(input_signal, filter_coefficients)
    out[:,-1] = np.zeros(len(input_signal))
    return out

def derivative_second_x(input_signal):
    filter_coefficients = np.array([[0,1,0],
                                    [0,-2,0],
                                    [0,1,0]], dtype=np.float64)
    return linear_shift_invariance(input_signal, filter_coefficients)

def derivative_second_y(input_signal):
    filter_coefficients = np.array([[0,0,0],
                                    [1,-2,1],
                                    [0,0,0]], dtype=np.float64)
    return linear_shift_invariance(input_signal, filter_coefficients)

def prewitt_x(input_signal):
    filter_coefficients = np.array([[1,1,1],
                                    [0,0,0],
                                    [-1,-1,-1]], dtype=np.float64)
    return linear_shift_invariance(input_signal, filter_coefficients)

def prewitt_y(input_signal):
    filter_coefficients = np.array([[1,0,-1],
                                    [1,0,-1],
                                    [1,0,-1]], dtype=np.float64)
    return linear_shift_invariance(input_signal, filter_coefficients)

def sobel_x(input_signal):
    filter_coefficients = np.array([[1,2,1],
                                    [0,0,0],
                                    [-1,-2,-1]], dtype=np.float64)
    return linear_shift_invariance(input_signal, filter_coefficients)

def sobel_y(input_signal):
    filter_coefficients = np.array([[1,0,-1],
                                    [2,0,-2],
                                    [1,0,-1]], dtype=np.float64)
    return linear_shift_invariance(input_signal, filter_coefficients)

def gradient(d_x, d_y):
    gra = np.zeros((len(d_x), len(d_x[0]), 2), dtype=np.float64)
    magnitude1 = np.zeros((len(d_x), len(d_x[0])), dtype=np.float64)
    magnitude2 = np.zeros((len(d_x), len(d_x[0])), dtype=np.float64)
    direction = np.zeros((len(d_x), len(d_x[0])), dtype=np.float64)    
    for i in range(len(d_x)):
        for j in range(len(d_x[0])):
            gra[i,j][0] = d_x[i,j]
            gra[i,j][1] = d_y[i,j]
            magnitude1[i,j] = np.sqrt(d_x[i,j]**2 + d_y[i,j]**2)
            magnitude2[i,j] = np.abs(d_x[i,j]) + np.abs(d_y[i,j])
            if d_x[i,j] == 0 and d_y[i,j] == 0:
                direction[i, j] = 0
            elif d_x[i,j] == 0:
                direction[i, j] = 90
            else:
                direction[i, j] = np.arctan(d_y[i, j]/ d_x[i, j])*180/np.pi
    return gra, np.round(magnitude1), np.round(magnitude2), np.round(direction)

def gaussian(n, std=1):      #ยิ่ง std มากจะเบลอมาก แล้วจะได้ขอบน้อย
    filter_matrix = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            filter_matrix[i,j] = np.exp(-((i-n//2)**2+(j-n//2)**2)/(2*std**2))
    return np.round(filter_matrix,2)

def nonmaxima_suppression(magnitude, direction):
    suppressed_magnitude = np.zeros_like(magnitude)
    magnitude = np.pad(magnitude, 1, 'constant', constant_values=0)
    height, width = magnitude.shape
    for i in range(1, height - 1):
        for j in range(1, width - 2):
            angle = direction[i-1, j-1]
            # Check the neighbors based on the gradient direction'
            if -22.5 <= angle <= 22.5 or (157.5 <= angle <= 180) or (-180 <= angle <= -157.5):
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            elif 22.5 < angle <= 67.5 or (-157.5 <= angle <= -112.5):
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
            elif 67.5 < angle <= 112.5 or (-112.5 <= angle <= -67.5):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 112.5 < angle <= 157.5 or (-67.5 <= angle <= -22.5):
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]

            # Suppress non-maximum values
            if magnitude[i, j] >= max(neighbors):
                suppressed_magnitude[i-1, j-1] = magnitude[i, j]
    return suppressed_magnitude

def hysteresis_thresholding(input_signal, lower_threshold, upper_threshold):
    #มี 2 ค่า threshold คือ low กับ high เนื่องจากแบบปกติทำแต่ high จะทำให้ขอบที่เป็นขอบจริงๆ หายไป เลยเอา low มาเพิ่มแล้วทำ neighbors ให้เป็นขอบจริงๆ
    strong_edge = np.zeros_like(input_signal)
    weak_edge = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        for j in range(len(input_signal[0])):
            if input_signal[i,j] >= upper_threshold:
                strong_edge[i,j] = 1
            elif input_signal[i,j] >= lower_threshold:
                weak_edge[i,j] = 1
    return strong_edge, weak_edge
        
def canny_edge(input_signal, blur, tl, th):
    first_step = linear_shift_invariance(input_signal, gaussian(5, blur))
    _,_,mag, dir = gradient(prewitt_x(first_step), prewitt_y(first_step))
    suppressed_magnitude = nonmaxima_suppression(mag, dir)
    strong_edge, weak_edge = hysteresis_thresholding(suppressed_magnitude, tl, th)      #ยังไม่ได้เทสผลลัพท์
    return strong_edge , weak_edge

def median_filter(input_signal, n): #ยังไม่ไดเทส
    output = np.zeros_like(input_signal)
    input_signal = np.pad(input_signal, n//2, 'constant', constant_values=0)
    for i in range(n//2, len(input_signal)-n//2):
        for j in range(n//2, len(input_signal[0])-n//2):
            output[i-n//2, j-n//2] = np.median(input_signal[i-n//2:i+n//2+1, j-n//2:j+n//2+1])
    return output

inp = np.array([[0,0,0,0,0,0],
                [0,0,0,255,255,255],
                [0,0,0,255,255,255],
                [0,0,0,255,255,255],
                [0,0,255,255,255,255],
                [0,255,255,255,255,255]])


inp = np.array([[0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,100,150,0,150,100,0,0,0],
                [0,0,100,200,200,150,200,200,100,0,0],
                [0,0,100,200,200,200,200,200,100,0,0],
                [0,0,100,200,200,200,200,200,100,0,0],
                [0,0,0,100,200,200,200,100,0,0,0],
                [0,0,0,0,100,200,100,0,0,0,0],
                [0,0,0,0,0,50,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0]])             

inp = np.array([[0,0,0,0,0,0],
                [0,0,7,0,0,0],
                [0,0,6,0,0,0],
                [0,0,2,6,0,0],
                [0,2,4,4,6,0],
                [0,2,7,7,7,0],
                [0,0,0,0,0,0]])

md = np.array([[0,0,3,2,0,0,0],
               [0,0,1,1,2,0,0],
               [0,0,0,0,47,4,0],
               [0,0,0,10,0,2,0],
               [0,75,65,15,0,0,0],
               [70,80,2,63,0,0,0],
               [0,67,53,55,0,0,0],
               [0,0,72,0,0,0,0]])

print(median_filter(md, 3))



