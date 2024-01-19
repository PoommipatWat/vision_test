import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from matplotlib.table import Table

class image_color:
    def __init__(self):
        pass

    @staticmethod
    def rgb2hsv(r, g, b):
        M = max([r, g, b])
        m = min([r, g, b])
        if m == M:
            h = 0
        elif M == r:
            h = (g - b) / (M - m) * 60 % 360
        elif M == g:
            h = (b - r) / (M - m) * 60 + 120
        elif M == b:
            h = (r - g) / (M - m) * 60 + 240
        return [h, 1 - (m / M), M / 255]

    @staticmethod
    def hsv2rgb(h, s, v):
        if s == 0:
            r = g = b = int(v * 255)
        else:
            h /= 60.0
            i = int(h)
            f = h - i
            p = int(255 * (v * (1.0 - s)))
            q = int(255 * (v * (1.0 - s * f)))
            t = int(255 * (v * (1.0 - s * (1.0 - f))))
            v = int(v * 255)
            if i == 0:
                r, g, b = v, t, p
            elif i == 1:
                r, g, b = q, v, p
            elif i == 2:
                r, g, b = p, v, t
            elif i == 3:
                r, g, b = p, q, v
            elif i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
        return [r, g, b]

    @staticmethod
    def show_hsv(h, s, v):
        h = h % 360
        h_norm = h / 2
        s_norm = int(s * 255)
        v_norm = int(v * 255)
        hsv_image = np.array([[[h_norm, s_norm, v_norm]]], dtype=np.uint8)
        rgb_color = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        color_display = np.zeros((100, 100, 3), dtype=np.uint8)
        color_display[:, :] = rgb_color
        cv2.imshow('Color from HSV', color_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def show_rgb(r, g, b):
        rgb_color = np.array([[[b, g, r]]], dtype=np.uint8)
        color_display = np.zeros((100, 100, 3), dtype=np.uint8)
        color_display[:, :] = rgb_color
        cv2.imshow('Color from RGB', color_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def rgb2gray(r, g, b):
        return np.mean([r, g, b])

    @staticmethod
    def rgb2gray_luminance(r, g, b):
        return 0.299 * r + 0.587 * g + 0.114 * b

class image_process:
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
    def derivative_first_x(input_signal):
        filter_coefficients = np.array([[0,1,0],
                                        [0,-1,0],
                                        [0,0,0]], dtype=np.float64)
        out = image_process.linear_shift_invariance(input_signal, filter_coefficients)
        out[-1] = np.zeros(len(input_signal[0]))
        return out

    @staticmethod
    def derivative_first_y(input_signal):
        filter_coefficients = np.array([[0,0,0],
                                        [1,-1,0],
                                        [0,0,0]], dtype=np.float64)
        out = image_process.linear_shift_invariance(input_signal, filter_coefficients)
        out[:,-1] = np.zeros(len(input_signal))
        return out

    @staticmethod
    def derivative_second_x(input_signal):
        filter_coefficients = np.array([[0,1,0],
                                        [0,-2,0],
                                        [0,1,0]], dtype=np.float64)
        return image_process.linear_shift_invariance(input_signal, filter_coefficients)

    @staticmethod
    def derivative_second_y(input_signal):
        filter_coefficients = np.array([[0,0,0],
                                        [1,-2,1],
                                        [0,0,0]], dtype=np.float64)
        return image_process.linear_shift_invariance(input_signal, filter_coefficients)

    @staticmethod
    def prewitt_x(input_signal):
        filter_coefficients = np.array([[1,1,1],
                                        [0,0,0],
                                        [-1,-1,-1]], dtype=np.float64)
        return image_process.linear_shift_invariance(input_signal, filter_coefficients)

    @staticmethod
    def prewitt_y(input_signal):
        filter_coefficients = np.array([[1,0,-1],
                                        [1,0,-1],
                                        [1,0,-1]], dtype=np.float64)
        return image_process.linear_shift_invariance(input_signal, filter_coefficients)

    @staticmethod
    def sobel_x(input_signal):
        filter_coefficients = np.array([[1,2,1],
                                        [0,0,0],
                                        [-1,-2,-1]], dtype=np.float64)
        return image_process.linear_shift_invariance(input_signal, filter_coefficients)

    @staticmethod
    def sobel_y(input_signal):
        filter_coefficients = np.array([[1,0,-1],
                                        [2,0,-2],
                                        [1,0,-1]], dtype=np.float64)
        return image_process.linear_shift_invariance(input_signal, filter_coefficients)

    @staticmethod
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

    @staticmethod
    def gaussian(n, std=1):
        filter_matrix = np.zeros((n,n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                filter_matrix[i,j] = np.exp(-((i-n//2)**2+(j-n//2)**2)/(2*std**2))
        return np.round(filter_matrix,2)

    @staticmethod
    def nonmaxima_suppression(magnitude, direction):
        suppressed_magnitude = np.zeros_like(magnitude)
        magnitude = np.pad(magnitude, 1, 'constant', constant_values=0)
        height, width = magnitude.shape
        for i in range(1, height - 1):
            for j in range(1, width - 2):
                angle = direction[i-1, j-1]
                if -22.5 <= angle <= 22.5 or (157.5 <= angle <= 180) or (-180 <= angle <= -157.5):
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                elif 22.5 < angle <= 67.5 or (-157.5 <= angle <= -112.5):
                    neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
                elif 67.5 < angle <= 112.5 or (-112.5 <= angle <= -67.5):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 112.5 < angle <= 157.5 or (-67.5 <= angle <= -22.5):
                    neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]

                if magnitude[i, j] >= max(neighbors):
                    suppressed_magnitude[i-1, j-1] = magnitude[i, j]
        return suppressed_magnitude

    @staticmethod
    def hysteresis_thresholding(input_signal, lower_threshold, upper_threshold):
        strong_edge = np.zeros_like(input_signal)
        weak_edge = np.zeros_like(input_signal)
        for i in range(len(input_signal)):
            for j in range(len(input_signal[0])):
                if input_signal[i,j] >= upper_threshold:
                    strong_edge[i,j] = 1
                elif input_signal[i,j] >= lower_threshold:
                    weak_edge[i,j] = 1
        return strong_edge, weak_edge

    @staticmethod
    def canny_edge(input_signal, blur, tl, th):
        first_step = image_process.linear_shift_invariance(input_signal, image_process.gaussian(5, blur))
        _,_,mag, dir = image_process.gradient(image_process.prewitt_x(first_step), image_process.prewitt_y(first_step))
        suppressed_magnitude = image_process.nonmaxima_suppression(mag, dir)
        strong_edge, weak_edge = image_process.hysteresis_thresholding(suppressed_magnitude, tl, th)
        return strong_edge , weak_edge

    @staticmethod
    def median_filter(input_signal, n):
        output = np.zeros_like(input_signal)
        input_signal = np.pad(input_signal, n//2, 'constant', constant_values=0)
        for i in range(n//2, len(input_signal)-n//2):
            for j in range(n//2, len(input_signal[0])-n//2):
                output[i-n//2, j-n//2] = np.median(input_signal[i-n//2:i+n//2+1, j-n//2:j+n//2+1])
        return output

    @staticmethod
    def dilation(input_signal, kernel):
        return binary_dilation(input_signal, structure=kernel).astype(np.int32)

    @staticmethod
    def erosion(input_signal, kernel):
        return binary_erosion(input_signal, structure=kernel).astype(np.int32)

    @staticmethod
    def opening(input_signal, kernel):
        return image_process.dilation(image_process.erosion(input_signal, kernel), kernel)

    @staticmethod
    def clossing(input_signal, kernel):
        return image_process.erosion(image_process.dilation(input_signal, kernel), kernel)

    @staticmethod
    def plot_table(matrix):
        plt.imshow(matrix, cmap='binary', interpolation='none', extent=[-0.5, matrix.shape[1]-0.5, -0.5, matrix.shape[0]-0.5])
        plt.grid(True, which='both', linestyle='-', linewidth=1, color='black')
        plt.xticks(np.arange(-0.5, matrix.shape[1]-0.5, 1), np.arange(0, matrix.shape[1], 1))
        plt.yticks(np.arange(-0.5, matrix.shape[0]-0.5, 1), np.arange(0, matrix.shape[0], 1))
        plt.show()

class classification:
    def __init__(self):
        pass

    @staticmethod
    def split_matrix(matrix, n, m):
        parts = []
        rows_per_part = len(matrix) // n
        cols_per_part = len(matrix[0]) // m
        for i in range(0, len(matrix), rows_per_part):
            for j in range(0, len(matrix[0]), cols_per_part):
                parts.append(matrix[i:i+rows_per_part, j:j+cols_per_part])
        return np.array(parts)

    @staticmethod
    def feature_space_extraction(matrix, n, m):
        blocks = classification.split_matrix(matrix, n, m)
        features = []
        for block in blocks:
            features.append(np.sum(block == 0))
        return features

    @staticmethod
    def entropy_find(*samples):
        n = 0
        sample = []
        for metrix in samples:
            n += (metrix)
            sample.append(metrix)
        entropy = 0
        for i in range(len(sample)):
            entropy += (sample[i])/n * np.log2((sample[i])/n)
        entropy *= -1
        return entropy, n

    @staticmethod
    def child_find(*samples):
        n = np.sum(samples)
        child = 0
        for i in range(len(samples)):
            child += (samples[i])/n * np.log2((samples[i])/n)
        child *= -1
        return child

    @staticmethod
    def information_gain_find(entropy, childs, samples):
        n = np.sum(samples)
        inner = 0
        for i in range (len(childs)):
            inner += (childs[i] * samples[i])/n
        return entropy - inner

    @staticmethod
    def homeworks_information_gain(sample_green, sample_red, sample_purple):
        # find entropy
        entropy, n = classification.entropy_find(sample_green, sample_red, sample_purple)

        for X in range(1, 6):
            less_than_green = 0
            morethan_green = 0
            for i in sample_green:
                if i[0] < X:
                    less_than_green += 1
                else:
                    morethan_green += 1
            less_than_red = 0
            morethan_red = 0
            for i in sample_red:
                if i[0] < X:
                    less_than_red += 1
                else:
                    morethan_red += 1
            less_than_purple = 0
            morethan_purple = 0
            for i in sample_purple:
                if i[0] < X:
                    less_than_purple += 1
                else:
                    morethan_purple += 1

            less_than = [less_than_green, less_than_red, less_than_purple]
            morethan = [morethan_green, morethan_red, morethan_purple]

            child_one = 0
            child_two = 0

            n_less = sum(less_than)
            n_more = sum(morethan)

            for i in range(3):
                if less_than[i] != 0:
                    child_one += less_than[i]/n_less * np.log2(less_than[i]/n_less)
                if morethan[i] != 0:
                    child_two += morethan[i]/n_more * np.log2(morethan[i]/n_more)
            child_one *= -1
            child_two *= -1

            print(child_one, child_two)

            information_gain = entropy - (n_less/n * child_one + n_more/n * child_two)
            print(f"X1 > {X}, information gain: {information_gain}")

        for Y in range(1, 6):
            less_than_green = 0
            morethan_green = 0
            for i in sample_green:
                if i[1] < Y:
                    less_than_green += 1
                else:
                    morethan_green += 1
            less_than_red = 0
            morethan_red = 0
            for i in sample_red:
                if i[1] < Y:
                    less_than_red += 1
                else:
                    morethan_red += 1
            less_than_purple = 0
            morethan_purple = 0
            for i in sample_purple:
                if i[1] < Y:
                    less_than_purple += 1
                else:
                    morethan_purple += 1

            less_than = [less_than_green, less_than_red, less_than_purple]
            morethan = [morethan_green, morethan_red, morethan_purple]

            child_one = 0
            child_two = 0

            n_less = sum(less_than)
            n_more = sum(morethan)

            for i in range(3):
                if less_than[i] != 0:
                    child_one += less_than[i]/n_less * np.log2(less_than[i]/n_less)
                if morethan[i] != 0:
                    child_two += morethan[i]/n_more * np.log2(morethan[i]/n_more)
            child_one *= -1
            child_two *= -1

            print(child_one, child_two)

            information_gain = entropy - (n_less/n * child_one + n_more/n * child_two)
            print(f"X2 > {Y}, information gain: {information_gain}")

    @staticmethod
    def plot_points(sample_green, sample_red, sample_purple):
        # Extract x and y coordinates for each class
        green_x, green_y = zip(*sample_green)
        red_x, red_y = zip(*sample_red)
        purple_x, purple_y = zip(*sample_purple)

        # Plot the points with different colors for each class
        plt.scatter(green_x, green_y, color='green', label='Green')
        plt.scatter(red_x, red_y, color='red', label='Red')
        plt.scatter(purple_x, purple_y, color='purple', label='Purple')

        # Add labels and legend
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Sample Data Plot')

        # Add a legend
        plt.legend()

        # Add a frame around the plot
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        # Show the plot
        plt.show()

    @staticmethod
    def pdf_find(pb, sample):
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)

        return 1/(std * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((pb - mean)/std)**2)

    #   GLCM
    @staticmethod
    def qiantize_intensity_level(input_signal,n):
        split = np.linspace(0, 255, n + 1, dtype=int)[1:]
        output = np.zeros_like(input_signal)
        for i in range(len(input_signal)):
            for j in range(len(input_signal[0])):
                for ind, k in enumerate(split):
                    if input_signal[i,j] <= k:
                        output[i,j] = ind
                        break
        return output
    
    @staticmethod
    def sym_GLCM_find(input_signal):
        return input_signal + input_signal.T
    
    @staticmethod
    def maximum_probability_find(input_signal):
        return np.max(input_signal)
    
    @staticmethod
    def angular_second_moment(input_signal):
        return np.sum(input_signal**2)
    
    @staticmethod
    def contrast(input_signal):
        output = 0
        for i in range(len(input_signal)):
            for j in range(len(input_signal[0])):
                output += input_signal[i,j] * (i-j)**2
        return output
    
    @staticmethod
    def correlation(input_signal):
        out = 0
        inp = input_signal.copy()
        inpt = input_signal.T
        u1 = 0
        u2 = 0
        for i in (range(len(inp))):
            u1 += i * np.sum(inp[i])
            u2 += i * np.sum(inpt[i])
        std12 = 0
        std22 = 0
        for i in (range(len(inp))):
            std12 += (i - u1)**2 * np.sum(inp[i])
            std22 += (i - u2)**2 * np.sum(inpt[i])
        std1 = np.sqrt(std12)
        std2 = np.sqrt(std22)
        for i in range(len(inp)):
            for j in range(len(inp[0])):
                out += (i - u1) * (j - u2) * inp[i,j] / (std1 * std2)
        print(u1, u2, std1, std2)
        return out
    
    @staticmethod
    def homogeneity(input_signal):
        output = 0
        for i in range(len(input_signal)):
            for j in range(len(input_signal[0])):
                output += input_signal[i,j] / (1 + np.abs(i-j))
        return output
    
    @staticmethod
    def entropy(input_signal):
        output = 0
        for i in range(len(input_signal)):
            for j in range(len(input_signal[0])):
                if input_signal[i,j] != 0:
                    output += input_signal[i,j] * np.log2(input_signal[i,j])
        return -output

    @staticmethod
    def GLCM_find(input_signal, n, dx, dy, nor='normalized', sym='symmetric'):
        output = np.zeros((n, n))
        input_s = classification.qiantize_intensity_level(input_signal, n)
        for i in range(len(input_s)):
            for j in range(len(input_s[0])):
                if i + dx < len(input_s) and j + dy < len(input_s[0]):
                    output[input_s[i,j], input_s[i+dx, j+dy]] += 1
        if sym == 'symmetric':
            output = classification.sym_GLCM_find(output)
        if nor == 'normalized':
            output /= np.sum(output)
            output = np.round(output, 2)
        return output, classification.maximum_probability_find(output), classification.angular_second_moment(output), classification.contrast(output), classification.homogeneity(output), classification.entropy(output), classification.correlation(output)

cls = classification()

