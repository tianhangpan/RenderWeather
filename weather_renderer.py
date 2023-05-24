import numpy as np
import cv2
import random
import skimage
from PIL import Image
from matplotlib import pyplot as plt


class WeatherRenderer:
    def __init__(self):
        pass

    @staticmethod
    def render_haze(img, depth):
        # img, depth: numpy ndarray format
        h, w = img.shape[: 2]

        # transmittance
        gauss_depth = cv2.GaussianBlur(depth, (2 * (2 * 5) + 1, 2 * (2 * 5) + 1), 5)
        beta = random.random() * 2 + 2.6

        tx = np.exp(-beta * gauss_depth)

        # atmospheric light
        a = 0.3 + 0.6 * random.random()
        a = np.ones([h, w, 3]) * a
        trans = (np.expand_dims(tx, axis=2)).repeat(3, axis=2)

        haze = img * trans + (1 - trans) * a

        return haze, trans, a

    @staticmethod
    def render_rain(img, theta, density, intensity):
        # img: numpy ndarray format
        h, w = img.shape[: 2]

        img = np.power(img, 2)

        # parameter seed gen
        s = 1.01 + random.random() * 0.2
        m = density * (0.2 + random.random() * 0.05)  # mean of gaussian, controls density of rain
        v = intensity + random.random() * 0.3  # variance of gaussian,  controls intensity of rain streak
        length = random.randint(1, 40) + 20  # len of motion blur, control size of rain streak

        # Generate proper noise seed
        dense_chnl = np.zeros([h, w, 1])
        dense_chnl_noise = skimage.util.random_noise(dense_chnl, mode='gaussian', mean=m, var=v)
        dense_chnl_noise = cv2.resize(dense_chnl_noise, dsize=(0, 0), fx=s, fy=s)
        pos_h = random.randint(0, dense_chnl_noise.shape[0] - h)
        pos_w = random.randint(0, dense_chnl_noise.shape[1] - w)
        dense_chnl_noise = dense_chnl_noise[pos_h: pos_h + h, pos_w: pos_w + w]

        # form filter
        m = cv2.getRotationMatrix2D((length / 2, length / 2), theta - 45, 1)
        motion_blur_kernel = np.diag(np.ones(length))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (length, length))
        motion_blur_kernel = motion_blur_kernel / length
        dense_chnl_motion = cv2.filter2D(dense_chnl_noise, -1, motion_blur_kernel)
        dense_chnl_motion[dense_chnl_motion < 0] = 0
        dense_streak = (np.expand_dims(dense_chnl_motion, axis=2)).repeat(3, axis=2)

        # Render Rain streak
        tr = random.random() * 0.05 + 0.04 * length + 0.2
        img_rain = img + tr * dense_streak
        img_rain[img_rain > 1] = 1
        actual_streak = img_rain - img

        return img_rain, actual_streak

    @staticmethod
    def get_float_ndimage(path):
        img = Image.open(path)
        img = np.array(img).astype(float) / 255
        return img


if __name__ == '__main__':
    img = WeatherRenderer.get_float_ndimage(r'/Users/pantianhang/python_data/datasets/nwpu/train/images/3106.jpg')
    img_rain, _ = WeatherRenderer.render_rain(img, 90, 0.15, 1.0)

    plt.imshow(img_rain)
    plt.show()
