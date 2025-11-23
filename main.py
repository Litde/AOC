import cv2
import numpy as np
from itertools import product
from tqdm import tqdm

def check_circle_parameters(params, image, cdst):
    for minDist, param1, param2, minRadius, maxRadius in product(
        params['minDist'],
        params['param1'],
        params['param2'],
        params['minRadius'],
        params['maxRadius']
    ):
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            1,
            minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
        if circles is None:
            continue


        if circles is not None:
            print(f"Detected {len(circles[0])} circles with parameters: dp={1}, minDist={minDist}, param1={param1}, param2={param2}, minRadius={minRadius}, maxRadius={maxRadius}")

        cp_cdst = np.copy(cdst)

        for i in circles[0, :]:
            i = np.uint16(np.around(i))
            cv2.circle(cp_cdst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(cp_cdst, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('Detected Circles', cp_cdst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def testing_circles():
    img = cv2.imread('0000001.jpg')
    img = cv2.pyrDown(img)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    # img = cv2.pyrDown(img)
    edges = cv2.Canny(img, 100, 200)

    cv2.imshow('Edges', edges)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines_params = {
        'rho': 1,
        'theta': np.pi / 180,
        'threshold': 100,
        'minLineLength': 50,
        'maxLineGap': 10
    }

    circles_params = {
        'minDist': [20, 30, 40, 50],
        'param1': [80, 120, 150, 200],
        'param2': [20, 40, 60, 80, 100],
        'minRadius': [10, 12, 15, 18],
        'maxRadius': [22, 28, 32]
    }

    check_circle_parameters(circles_params, grayscale_img, img)


def main():
    img = cv2.imread('0000001.jpg')
    img = cv2.pyrDown(img)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)
    edges = cv2.Canny(img, 100, 200)
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    circles = cv2.HoughCircles(image=gray_scale_img, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=30,
                               param1=80, param2=20,
                               minRadius=18, maxRadius=28)

    to_paint = np.copy(img)

    if circles is not None:
        print(f"Detected {len(circles[0])} circles")
        for i in circles[0, :]:
            i = np.uint16(np.around(i))
            cv2.circle(to_paint, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(to_paint, (i[0], i[1]), 2, (0, 0, 255), 3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100,
                            minLineLength=50,
                            maxLineGap=10)

    if lines is not None:
        print(f"Detected {len(lines)} lines")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(to_paint, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Detected Circles and Lines', to_paint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()