import cv2
from skimage.feature import hog
from skimage import exposure


cap = cv2.VideoCapture(0)


if __name__ == '__main__':

    while (True):

        ret, frame = cap.read()

        fd, atual = hog(frame, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

        momento = exposure.rescale_intensity(atual, in_range=(0, 10))

        cv2.imshow("HOG", momento)

        k = cv2.waitKey(1)
        if k == 27:
            exit()
        if k == ord('q'):
            exit()

    cv2.destroyAllWindows()
