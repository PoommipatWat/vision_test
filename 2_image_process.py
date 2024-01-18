import numpy as np

def linear_shift_invariance(input_signal, filter_coefficients):
    filter_coefficients = filter_coefficients[::-1,::-1]

    output = np.zeros((len(input_signal), len(input_signal[0])), dtype=np.float64)

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
            direction[i,j] = np.arctan2(d_y[i,j], d_x[i,j])*180/np.pi
    return gra, np.round(magnitude1), np.round(magnitude2), np.round(direction)



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
dx = prewitt_x(inp)
dy = prewitt_y(inp)

gra, magnitude1, magnitude2, direction = gradient(dx,dy)
print(direction)

