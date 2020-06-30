import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey()
    cv.destroyAllWindows()


# plt显示彩色图片
def plt_show0(img):
    b, g, r = cv.split(img)
    img = cv.merge([r, g, b])
    plt.imshow(img)
    plt.show()


# plt显示灰度图片
def plt_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def digital_segmentation(image):

    image = cv.GaussianBlur(image, (3, 3), 0) #高斯去噪

    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY) #转灰度图像
    #cv.imshow("gray image", gray)


    gray = cv.medianBlur(gray, 5) #中值滤波去除椒盐噪声
    #cv.imshow("medianBlur image", gray)

    ret, binary = cv.threshold(gray,0,255,cv.THRESH_OTSU|cv.THRESH_BINARY_INV)  #获取二值化图像
    #cv.imshow('a',binary)


    kernelX = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 17在x方向膨胀力度更大 返回指定形状和尺寸的结构元素。cv2.MORPH_RECT返回得是个矩形
    #print(kernelX)
    binary1 = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernelX, iterations=3)

    #
    kerne2X = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # kernelY = cv.getStructuringElement(cv.MORPH_RECT, (1, 5))

    # 膨胀，腐蚀
    binary1 = cv.dilate(binary1, kerne2X)
    # 黑白反转 由于minist数据集是白底黑字
    bin = binary #(255-binary)*(1/255)
    #cv.imshow('b', bin)

    # binary = cv.erode(binary, kernelX)
    # # 腐蚀，膨胀
    # binary = cv.erode(binary, kernelY)
    # binary = cv.dilate(binary, kernelY)



    #print("thresold value:",ret)
    #cv.imshow("binary image",binary)

    contours,hireachy = cv.findContours(binary1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) #只检测外轮廓即可

    # image1 = image.copy()
    # cv.drawContours(image1, contours, -1, (0, 255, 0), 5)#绘制轮廓 查看轮廓是否正确识别
    # plt_show0(image1)

    numbers = []
    number_image = []
    areas = []
    i = 0
    for item in contours:
        i = i+1
        area = cv.contourArea(item)      #获取每个轮廓面积
        #print("contour area" + str(i) + ':',  area)

        x,y,w,h = cv.boundingRect(item)     #获取轮廓的外接矩形
        rate = min(w,h)/max(w,h)    #获取外接矩形宽高比，可以起到一定的筛选作用
        #print("rectangle rate:%s"%rate)
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)  # 根据轮廓外接矩形返回数据，画出外接矩形
        cv.imshow("measure_object", image)
        number = []
        number.append(x) # 将矩形轮廓得四点坐标添加到number
        number.append(y)
        number.append(w)
        number.append(h)
        numbers.append(number) #再将所有number添加到numbers

        area1 = []
        area1.append(x) # 将x坐标和轮廓area添加到area1
        area1.append(area)
        areas.append(area1) # 再将所有area1添加到areas


    numbers = sorted(numbers, key=lambda s: s[0], reverse=False) # 按照x方向顺序来进行保存分割数字
    #print(numbers)

    areas = sorted(areas, key=lambda s: s[0], reverse=False) #按照x得方向来保存每个分割数字得面积
    #print("areas",areas)
    areas = [i[1] for i in areas] # 获得二维列表得第一列 这一列是按顺序保存得面积

    min_areas_idx = areas.index(min(areas)) # 返回最小面积得索引
    print('小数点索引位置', min_areas_idx)
    i = 0
    for number in numbers:
        i = i + 1
        splite_image = bin[(number[1]-2):(number[1]-2) + (number[3]+2), (number[0]-2):(number[0]-2) + (number[2]+2)]
        number_image.append(splite_image)
        #plt_show0(number_image)
        cv.imwrite('./get_nums_img/' + str(i-1) + '.png', splite_image)
    print('手写数字得定位、分割、保存，已完成')
    return number_image , min_areas_idx # 返回分割图像列表和分割图像中最小面积得索引用于后期识别小数点得依据




if __name__ == '__main__':
    src = cv.imread("./test/tu2.png")  # 读取图片
    # cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)    #创建GUI窗口,形式为自适应
    # cv.imshow("input image",src)    #通过名字将图像和窗口联系

    # 手写数字定位分割
    number_image, min_areas_idx = digital_segmentation(src)



    cv.waitKey(0)   #等待用户操作，里面等待参数是毫秒，我们填写0，代表是永远，等待用户操作
    cv.destroyAllWindows()  #销毁所有窗口