import cv2
import numpy as np
from numpy import linalg as LA


def perpend_angle(x, y):
    if x == 0:
        return np.pi / 2 + np.pi
    else:
        res = np.arctan(y / x)

    if x < 0 and y < 0:
        z = res + np.pi
    elif x < 0 and y > 0:
        z = res + np.pi
    elif x > 0 and y < 0:
        z = res + np.pi * 2
    else:
        z = res
    z += np.pi / 2
    if z > np.pi * 2:
        return z - np.pi * 2
    else:
        return z


def flow_neighbor(ang):
    if ang <= np.pi / 8 or ang >= np.pi * 15 / 8:
        return [-1, 0]
    elif np.pi / 8 <= ang <= 3 / 8 * np.pi:
        return [-1, 1]
    elif 3 / 8 * np.pi <= ang <= np.pi * 5 / 8:
        return [0, 1]
    elif 5 / 8 * np.pi <= ang <= np.pi * 7 / 8:
        return [1, 1]
    elif 7 / 8 * np.pi <= ang <= np.pi * 9 / 8:
        return [1, 0]
    elif np.pi * 9 / 8 <= ang <= np.pi * 11 / 8:
        return [1, -1]
    elif np.pi * 11 / 8 <= ang <= np.pi * 13 / 8:
        return [0, -1]
    elif np.pi * 13 / 8 <= ang <= np.pi * 15 / 8:
        return [-1, -1]
    else:
        return [-1, 0]


def guassian(x, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(- x * x / (2 * sigma * sigma))


def improved_filter(x, sigma, r):
    sig = sigma * 1.05
    return guassian(x, sigma) - guassian(x, sig)


def grad(gray, iterations=1, mu=5):
    gray_img = gray.astype(np.float) / 255.
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    if np.amax(sobel_mag) == 0:
        sobel_m = np.zeros(gray.shape)
    else:
        sobel_m = sobel_mag / np.amax(sobel_mag)

    tang_x = - sobel_y
    tang_y = sobel_x
    tang = np.stack((tang_x, tang_y), axis=2)
    size = tang.shape

    tang_norm = LA.norm(tang, axis=2)
    np.place(tang_norm, tang_norm == 0, [1])
    tang = tang / np.stack((tang_norm, tang_norm), axis=2)

    for iteration in range(iterations):
        tang_hor = np.zeros(size)
        for h in range(size[0]):
            for w in range(size[1]):
                total_weight = 0.
                for j in range(max(w - mu, 0), min(size[1], w + mu + 1)):
                    weight = (1 - abs(sobel_m[h][w] - sobel_m[h][j])) * np.dot(tang[h][w], tang[h][j]) / 2.0
                    total_weight += weight
                    tang_hor[h][w] += weight * tang[h][j]
                if total_weight != 0:
                    tang_hor[h][w] = tang_hor[h][w] / total_weight

        tang = np.zeros(shape=size)

        for h in range(size[0]):
            for w in range(size[1]):
                total_weight = 0.
                for i in range(max(0, h - mu), min(size[0], h + mu + 1)):
                    weight = (1 - abs(sobel_m[h][w] - sobel_m[i][w])) * np.dot(tang_hor[h][w], tang_hor[i][w]) / 2.0
                    total_weight += weight
                    tang[h][w] += weight * tang_hor[i][w]
                if total_weight != 0:
                    tang[h][w] /= total_weight

    tang_norm = LA.norm(tang, axis=2)
    np.place(tang_norm, tang_norm == 0, [1])
    tang = tang / np.stack((tang_norm, tang_norm), axis=2)
    return tang


def search_edge_iter(input_image, grad_img):
    size = input_image.shape
    H2 = input_image
    H1 = np.zeros(size)

    for h in range(size[0]):
        for w in range(size[1]):
            angle_rect = perpend_angle(grad_img[h][w][0], grad_img[h][w][1]) + np.pi / 2
            if angle_rect >= 2 * np.pi:
                angle_rect -= 2 * np.pi
            pix = flow_neighbor(angle_rect)

            for j in range(-3, 4):
                if 0 <= h + pix[0] * j < size[0] and 0 <= w + pix[1] * j < size[1]:
                    H1[h][w] += H2[h + pix[0] * j][w + pix[1] * j] * improved_filter(abs(j), 1.0, 0.9761)

    H2 = np.zeros(size)

    for h in range(size[0]):
        for w in range(size[1]):
            total_weight = 0.
            angle_rect = perpend_angle(grad_img[h][w][0], grad_img[h][w][1])
            pix = flow_neighbor(angle_rect)

            for j in range(-3, 4):
                if 0 <= h + pix[0] * j < size[0] and 0 <= w + pix[1] * j < size[1]:
                    weight = guassian(abs(j), 3.)
                    total_weight += weight
                    H2[h][w] += weight * H1[h + pix[0] * j][w + pix[1] * j]

            H2[h][w] /= total_weight

    return H2


def search_edge(image, grad_img, iterations=1):
    H2 = search_edge_iter(image, grad_img)
    indi1 = (H2 < 0).astype(np.int)
    indi2 = (1 + np.tanh(H2) < 0.5).astype(np.int)

    edges = (1 - indi1 * indi2) * 255
    modified_image = np.minimum(image, edges)
    for i in range(iterations - 1):
        edges = search_edge_iter(modified_image, grad_img)
        indi1 = (edges < 0).astype(np.int)
        indi2 = (1 + np.tanh(edges) < 0.5).astype(np.int)
        edges = (1 - np.multiply(indi1, indi2)) * 255
        modified_image = np.minimum(modified_image, edges)

    return edges


def intensity_weight(rgb1, rgb2, sigma):
    x = np.dot(rgb1 - rgb2, rgb1 - rgb2)
    k = 1 / (np.sqrt(2 * np.pi) * sigma)
    return k * np.exp(- x ** 2 / (2 * sigma * sigma))


def smooth_image(image, grad_img, iterations=1):
    size = image.shape
    H4 = image

    for i in range(iterations):
        H3 = np.zeros(size)

        for h in range(size[0]):
            for w in range(size[1]):
                total_weight = 0.
                ang = perpend_angle(grad_img[h][w][0], grad_img[h][w][1])
                pix = flow_neighbor(ang)
                for j in range(-5, 6):
                    if 0 <= h + pix[0] * j < size[0] and 0 <= w + pix[1] * j < size[1]:
                        weight = guassian(abs(j), 2.) * intensity_weight(H4[h][w],
                                                                         H4[h + pix[0] * j][w + pix[1] * j],
                                                                         150.)
                        H3[h][w] += H4[h + pix[0] * j][w + pix[1] * j] * weight
                        total_weight += weight
                if total_weight != 0:
                    H3[h][w] /= total_weight

        H4 = np.zeros(size)

        for h in range(size[0]):
            for w in range(size[1]):
                total_weight = 0.
                angle_perpend = perpend_angle(grad_img[h][w][0], grad_img[h][w][1]) + np.pi / 2
                if angle_perpend > 2 * np.pi:
                    angle_perpend -= 2 * np.pi
                pix = flow_neighbor(angle_perpend)
                for j in range(-5, 6):
                    if 0 <= h + pix[0] * j < size[1] and 0 <= w + pix[1] * j < size[1]:
                        weight = guassian(abs(j), 2.) * intensity_weight(H3[h][w],
                                                                         H3[h + pix[0] * j][w + pix[1] * j],
                                                                         50.)
                        H4[h][w] += weight * H3[h + pix[0] * j][w + pix[1] * j]
                        total_weight += weight
                if total_weight != 0:
                    H4[h][w] /= total_weight
                else:
                    H4[h][w] = H3[h][w]
    return H4