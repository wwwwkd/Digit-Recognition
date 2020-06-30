# list = [0.1259, 0.0776, 0.1231, 0.0803, 0.1428, 0.0747, 0.1139, 0.0714, 0.1172,
#          0.0731]
# a = sum(list)
# print(a)
import cv2
pic = cv2.imread('./get_nums_img/5.png')
pic = cv2.resize(pic, (28, 28), interpolation=cv2.INTER_CUBIC)
pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
cv2.imshow('', pic)
cv2.waitKey(0)
cv2.destroyAllWindows()

import scipy.misc
import  cv2 as cv
import torch
# image_array = cv.imread('./test/tu1.png')
# image_array = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)
# image_array = image_array.fromarry(image_array)
# scipy.misc.save('outfile.jpg')

# x = cv.imread('./get_nums_img/' + str(0) + '.png')
# x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
# x = cv.resize(x, (28,28))
# x = torch.from_numpy(x)
# x = x.reshape(1, 1, 28, 28)
# x = x.type(torch.FloatTensor)
# print(x.type())
