import numpy as np
import matplotlib.pyplot as plt

def split_matrix(matrix, n, m):
    parts = []
    rows_per_part = len(matrix) // n
    cols_per_part = len(matrix[0]) // m
    for i in range(0, len(matrix), rows_per_part):
        for j in range(0, len(matrix[0]), cols_per_part):
            parts.append(matrix[i:i+rows_per_part, j:j+cols_per_part])
    return np.array(parts)

def feature_space_extraction(matrix, n, m):
    blocks = split_matrix(matrix, n, m)
    features = []
    for block in blocks:
        features.append(np.sum(block == 0))
    return features

ind = np.array([[1,1,1,1,1,1,1,1],
                [1,1,1,0,0,1,1,1],
                [1,1,0,0,0,1,1,1],
                [1,1,0,1,0,0,1,1],
                [1,0,0,1,1,0,1,1],
                [1,0,0,1,1,0,0,1],
                [0,0,0,0,0,0,0,0],
                [0,0,1,1,1,0,0,0],
                [0,0,1,1,1,1,0,0],
                [0,0,1,1,1,1,0,0],
                [0,0,1,1,1,1,0,0],
                [1,1,1,1,1,1,1,1]])

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

def child_find(*samples):
    n = np.sum(samples)
    child = 0
    for i in range(len(samples)):
        child += (samples[i])/n * np.log2((samples[i])/n)
    child *= -1
    return child

def information_gain_find(entropy, childs, samples):
    n = np.sum(samples)
    inner = 0
    for i in range (len(childs)):
        inner += (childs[i] * samples[i])/n
    return entropy - inner

def homeworks_information_gain():
    # Data
    sample_green = [[0.5, 5.5], [0.5, 3.5], [1.5, 4.5], [1.5, 3.3], [1.5, 3.7], [3.5, 2.5], [4.5, 2.7]]
    sample_red = [[0.5, 1.5], [1.5, 2.5], [2.5, 1.5], [2.5, 3.5], [5.5, 0.5], [4.5, 1.7], [4.5, 1.3]]
    sample_purple = [[3.5, 4.5], [4.5, 2.3], [4.5, 4.3], [4.5, 4.7], [4.5, 5.5], [5.5, 4.5], [5.5, 5.5]]
    # find entropy
    entropy, n = entropy_find(sample_green, sample_red, sample_purple)

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
        print(f"X1 > {X}, infotmation gain : {information_gain}")
        
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
        print(f"X2 > {Y}, infotmation gain : {information_gain}")


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

def pdf_find(pb, sample):
    mean = np.mean(sample)
    print(mean)
    std = np.std(sample, ddof=1)
    print(std**2)
    return 1/(std * np.sqrt(2 * np.pi)) * np.exp(-1/2 * ((pb - mean)/std)**2)


