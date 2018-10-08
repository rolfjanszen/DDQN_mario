import cv2
import numpy as np
from math import sqrt


def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
        if n % i == 0:
            if i == 1: print('Who would enter a prime number of filters')
            return (i, int(n / i))


#Takes 3D matrix (set of images) and creates 2D grid output
def visualize_filter(kernel, scale):

    kernel_sz = kernel.shape[2]
    (grid_Y, grid_X) =factorization(kernel_sz)

    dims_x = kernel.shape[0]
    dims_y = kernel.shape[1]
    spacer = 3

    grid = np.array([[0 ]  *(spacer +dims_y ) *grid_Y]*(spacer +dims_x ) *grid_X )
    pos_X = 0
    pos_Y = 0
    # grid_X -= 1

    for i in range(kernel_sz):

        if i % grid_X == 0 and i != 0:
            pos_Y += (dims_y +spacer)
            pos_X = 0

        print('pos ' ,pos_X ,pos_Y)
        normal_im = spread_bit_val(kernel[: ,: ,i])
        grid[pos_X:(pos_X +dims_x) ,pos_Y:(pos_Y +dims_y)] = normal_im
        print(' kernel ',kernel[: ,: ,i])
        pos_X += (dims_x +spacer)

    grid = np.array(grid, dtype=np.uint8)

    # grid = spread_bit_val(grid)
    grid_resize = cv2.resize(grid ,(0 ,0) ,fx=scale ,fy=scale)
    # cv2.imshow('grid' ,grid_resize)
    # cv2.waitKey(10)

    return grid_resize

def spread_bit_val(float_image):

    kern_max = np.max(float_image)
    kern_min = np.min(float_image)
    print(' kenr max min ',kern_min, kern_max)
    float_image -= kern_min
    float_image = float_image * (200 / (kern_max - kern_min))

    image = np.array(float_image, dtype=np.uint8)
    return  image