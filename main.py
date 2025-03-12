import cv2
import numpy as np

def resize_image(img,NH,HW,c,scaleH,scaleW):#img=原圖 NH=新高度 HW=新寬度 c=通道數 scaleH=高度縮放比例 scaleW=寬度縮放比例
    # 初始化新圖像，使用零填充，資料型態為 np.uint8 (適用於圖片格式)
    New_img = np.zeros((NH, HW, c),dtype=np.uint8)
    for i in range(NH):
        for j in range(HW):
            Orig_x = j / scaleW #計算原圖中對應的坐標
            Orig_y = i / scaleH
            # 最近鄰插值(取整數部分作為最近鄰點)
            nn_x = int(Orig_x)
            nn_y = int(Orig_y)
            # 限制最近鄰點防止超出圖片邊界
            nn_x = min(max(nn_x, 0), w - 1)
            nn_y = min(max(nn_y, 0), h - 1)
            nearest_pixel = img[nn_y, nn_x]
            # 雙線性插值（只當需要進行計算時）
            x1, y1 = int(Orig_x), int(Orig_y)#取左上角座標
            x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)#取右下角座標，防止不會超出邊界
            # 取得四個鄰近像素的值，分別為左上、右上、左下、右下
            A = img[y1, x1]
            B = img[y1, x2]
            C = img[y2, x1]
            D = img[y2, x2]
            # 計算插值權重，分別為X軸左側、X軸右側、Y軸左側、Y軸右側
            x1_weight = x2 - Orig_x
            x2_weight = Orig_x - x1
            y1_weight = y2 - Orig_y
            y2_weight = Orig_y - y1
            # 計算雙線性插值的顏色值
            bilinear_pixel = (A * x1_weight * y1_weight +
                              B * x2_weight * y1_weight +
                              C * x1_weight * y2_weight +
                              D * x2_weight * y2_weight)
           # 綜合最近鄰插值與雙線性插值，取兩者平均值
            final_pixel = (nearest_pixel + bilinear_pixel) // 2  
            # 設定最終顏色
            New_img[i, j] = np.clip(final_pixel, 0, 255)
    return  New_img

def bilateral_filter(img):
    d,Scolor, Sspace=5,75,75 #卷積核、顏色權重標準差、空間權重標準差
    h, w = img.shape[:2] # 獲取圖像的高度、寬度和通道數
    radius = d // 2# 計算半徑
    output = np.zeros_like(img, dtype=np.uint8)# 初始化輸出圖像
    space_kernel = np.zeros((d, d), dtype=np.float32)# 計算高斯核
    for i in range(d):# 計算空間高斯權重
        for j in range(d):
            space_kernel[i, j] = np.exp(-((i - radius) ** 2 + (j - radius) ** 2) / (2 * Sspace ** 2))
    for i in range(h):# 遍歷每個像素
        for j in range(w):           
            Wsum = 0# 初始化權重和加權總和
            pixel_value_sum = np.zeros(3) # 用於存儲加權後的像素值RGB
            for m in range(-radius, radius + 1):# 遍歷窗口內的所有像素
                for n in range(-radius, radius + 1):
                    x = i + m #窗口內的索引
                    y = j + n
                    if x >= 0 and x < h and y >= 0 and y < w:# 確保不超出邊界      
                        Cdiff = np.linalg.norm(img[i, j] - img[x, y])# 計算顏色差異 (使用歐幾里得距離)
                        Ckernel = np.exp(-Cdiff ** 2 / (2 * Scolor ** 2))# 計算顏色權重 (顏色差異越小，權重越大)
                        space_kernel_val = space_kernel[m + radius, n + radius]# 取得空間高斯權重
                        weight = Ckernel * space_kernel_val# 計算總權重 (空間權重 × 顏色權重)
                        pixel_value_sum += weight * img[x, y]# 加權像素值
                        Wsum += weight
            output[i, j] = (pixel_value_sum / Wsum).astype(np.uint8) # 計算濾波後的像素值 (除以總權重進行正規化)
    return output

def addsign(img):
    sign_img=cv2.imread("sign.png") #讀取簽名字檔的圖片
    Sh,Sw=sign_img.shape[:2] #只拿取簽名檔的高跟寬
    for i in range(Sh):
        for j in range(Sw):
            # 檢查該像素是否為黑色 RGB=0,0,0
            if(sign_img[i][j][0]==0)and(sign_img[i][j][1]==0)and(sign_img[i][j][2]==0):
                #如果該像素是黑色，則將原圖(`img`) 上對應位置的像素設為黑色
                img[i][j][0]=0
                img[i][j][1]=0
                img[i][j][2]=0
    bilateral_filter(img)#對修改後的`img`圖片進行雙邊濾波
    return img

def bigger_picture(img,h,w,c,scale):
    NH,NW = int(h * scale), int(w * scale)
    bigimg=resize_image(img,NH,NW,c,scale,scale)
    bigimg=addsign(bigimg)
    cv2.imshow("bigger", bigimg)
    cv2.imwrite("../bigger_picture.png",bigimg)#將圖片儲存到指定位子

def smaller_picture(img,h,w,c,scale):
    NH,NW = int(h* scale), int(w * scale)
    smallimg=resize_image(img,NH,NW,c,scale,scale)
    cv2.imshow("smaller", smallimg)
    cv2.imwrite("../smaller_picture.png",smallimg)

def width_picture(img,h,w,c,scale):
    NH,NW = h, int(w * scale)
    widthimg=resize_image(img,NH,NW,c,1,scale)
    cv2.imshow("width", widthimg)
    cv2.imwrite("../width_picture.png",widthimg)

def height_picture(img,h,w,c,scale):
    NH,NW = int(h * scale), int(w )
    heightimg=resize_image(img,NH,NW,c,scale,1)
    cv2.imshow("height", heightimg)
    cv2.imwrite("../height_picture.png",heightimg)

# 讀取圖片
img = cv2.imread('image.jpg')#讀取圖片
cv2.imshow('Original',img)
h, w,c= img.shape#抓取圖片的長寬
bigger_picture(img,h,w,c,1.5)#放大圖片
smaller_picture(img,h,w,c,1/1.5)#縮小圖片
width_picture(img,h,w,c,1.5)#寬度放大圖片
height_picture(img,h,w,c,1.5)#長度放大圖片
# 顯示結果
cv2.waitKey(0)
cv2.destroyAllWindows()