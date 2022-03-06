import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

# 返回一个cv2 image的RGB或HSV color均值, HSV取值范围为0~360, 0~1, 0~1
def ColorMean(im, mode):
    if (mode == 'RGB') or (mode == 'rgb'): 
        b, g, r = cv2.split(im)
        return int(np.mean(r)), int(np.mean(g)), int(np.mean(b))
    else:
        if (mode == 'HSV') or (mode == 'hsv'): 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(im)
            return 2*np.mean(h), np.mean(s)/255, np.mean(v)/255
        else: print('Error: not a cv2 image type.')

# TY's color balance，接收一个cv2 image，返回一个cv2 image
def AutoColorBalance(im): 
    stander_r = 223
    stander_g = 223
    stander_b = 233
    b, g, r = cv2.split(im)
    r_mean, g_mean, b_mean = ColorMean(im, 'rgb')
    b = b/b_mean * stander_b
    b[b>255] = 255
    b[b<0] = 0
    g = g/g_mean * stander_g
    g[g>255] = 255
    g[g<0] = 0
    r = r/r_mean * stander_r
    r[r>255] = 255
    r[r<0] = 0
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

# 求解轮廓的最大内接圆，传入一个cv2 img和其所有的轮廓，返回一个np.array包含每个轮廓对应的最大内接圆圆心和半径
def FindMinInternalCircle(im, contours):
    circles=[]
    for c in contours:
        raw_dist = np.empty(im.shape, dtype=np.float32)
        for i in range(im.shape[0]) :
            for j in range(im.shape[1]) :
                raw_dist[i, j] = cv2. pointPolygonTest(c, (j, i), True)
        minVal, maxVal, minDistPt, maxDistPt = cv2.minMaxLoc(raw_dist)
        maxVal = abs(maxVal)
        radius = int(maxVal)
        center_of_circle = maxDistPt
        circles.append([center_of_circle, radius])
    return circles

# 接受一个cv2 image，返回mask, im_copy, num, big, mid, small
def THCounting(im):

    im_copy = np.copy(im)
    hight = im.shape[0]
    width = im.shape[1]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # 根据斑块的颜色特征设定阈值
    lower_threshold = np.array([0, 0, 0])
    upper_threshold = np.array([200, 180, 205])
    # upper_threshold = np.array([200, 180, 200])
    mask = cv2.inRange(im, lower_threshold, upper_threshold)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=6)
    # mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    # mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    # cv2.imwrite('mask.jpg', mask)

    # RETR_EXTERNAL 如果你选择这种模式的话，只会返回最外边的的轮廓，所有的子轮廓都会被忽略掉
    # RETR_TREE 则会给出轮廓层级关系
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num = 0 # TH总数
    fitEllipseError = 0
    area_mean = 3200*hight*width/(4080*3072)
    area_threshold2 = area_mean*1
    area_threshold1 = area_mean/5

    for i in range(len(contours)):
        area = 0 # 单个轮廓的面积
        try:
            if hierarchy[0][i][3] == -1: # 如果这个第i个轮廓没有父轮廓，则应该被加上
                area = area + cv2.contourArea(contours[i])
            elif hierarchy[0][i][3] != -1: # 如果这个第i个轮廓具有某个父轮廓，则应该被减去
                area = area - cv2.contourArea(contours[i])
            else: print('面积加减出现未知错误')
            ellipse = cv2.fitEllipse(contours[i]) # 寻找椭圆
            pt = (int(ellipse[0][0]) - 20,int(ellipse[0][1]) + 20) # 记录椭圆圆心坐标，-20和+20是为了将数字标记对齐轮廓中心

            # 如果是超大区域
            if area > area_threshold2:
                num = num + round(area/area_mean)
                cv2.putText(im_copy, str(int(round(area/area_mean))), pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                im_copy = cv2.ellipse(im_copy, ellipse,(255,255,255),2) # 在原始彩色图像上绘制椭圆

            # 如果是普通大小区域则num+1
            elif area_threshold1 < area < area_threshold2:
                num = num + 1
                cv2.putText(im_copy, '1', pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
                im_copy = cv2.ellipse(im_copy, ellipse,(255,255,255),2) # 在原始彩色图像上绘制椭圆

            # 如果是超小区域则忽略
            else: pass
                # mod = cv2.ellipse(mod, ellipse,(255,255,255),2) # 在原始彩色图像上绘制椭圆
        except:
            fitEllipseError += 1
    stringtotal = "THcounting:" + str(num)
    cv2.putText(im_copy, stringtotal, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

    return im_copy

# 给定输入和输出目录
filepath_input = "./Original images"
filepath_output = os.path.abspath(os.path.join(os.path.join(filepath_input, os.pardir), "Output"))

# 如果存在jpg目录则提前删除然后新建一个，如果不存在jpg目录则新建一个
if os.path.exists(filepath_output):
    shutil.rmtree(filepath_output)
    os.mkdir(filepath_output)
else: os.mkdir(filepath_output)

for root, dirs, files in os.walk(filepath_input, topdown=False): 
    # files.sort(key=lambda x:int(x[:-5])) # 根据文件名排序files中的文件顺序，这里结尾扩展名是.jpeg，所以是-5
    for name in tqdm(files):
        # If a jpg is present, alarming this.
        if os.path.isfile(os.path.splitext(os.path.join(filepath_output, name))[0] + ".jpg"):
            print("A jpg file already exists for %s" % name)
        else:
            outfile_name = os.path.splitext(os.path.join(filepath_output, name))[0] + ".jpg"
            if os.path.join(root, name)[-9:] == ".DS_Store": pass
            else:
                filesuffix = os.path.splitext(os.path.join(root, name))[1].lower()
                if (filesuffix == ".png") or (filesuffix == ".tif") or (filesuffix == ".jpg") or (filesuffix == ".jpeg"):
                    try:
                        im = cv2.imread(os.path.join(root, name)) # 读取图片
                        im = AutoColorBalance(im) # 标准化色差
                        mod = THCounting(im) # Abeta plaque counting
                        cv2.imwrite(outfile_name, mod) # 保存处理后的图片
                    except Exception: print("Cannot proccess")
                else: print("Cannot proccess")