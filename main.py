import cv2
import numpy as np

def resize_image(img,NH,HW,c,scaleH,scaleW):
    New_img = np.zeros((NH, HW, c),dtype=np.uint8)
    
    for i in range(NH):
        for j in range(HW):
            Orig_x = j / scaleW #計算原圖中對應的坐標
            Orig_y = i / scaleH

            # 最近鄰插值
            nn_x = int(Orig_x)
            nn_y = int(Orig_y)
            nn_x = min(max(nn_x, 0), w - 1)
            nn_y = min(max(nn_y, 0), h - 1)
            nearest_pixel = img[nn_y, nn_x]

            # 雙線性插值（只當需要進行計算時）
            x1, y1 = int(Orig_x), int(Orig_y)
            x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

            A = img[y1, x1]
            B = img[y1, x2]
            C = img[y2, x1]
            D = img[y2, x2]

            x1_weight = x2 - Orig_x
            x2_weight = Orig_x - x1
            y1_weight = y2 - Orig_y
            y2_weight = Orig_y - y1

            # 計算雙線性插值的顏色值
            bilinear_pixel = (A * x1_weight * y1_weight +
                              B * x2_weight * y1_weight +
                              C * x1_weight * y2_weight +
                              D * x2_weight * y2_weight)

            # 綜合最近鄰插值與雙線性插值
            final_pixel = (nearest_pixel + bilinear_pixel) // 2  # 平均兩者

            # 設定最終顏色
            New_img[i, j] = np.clip(final_pixel, 0, 255)

    return  New_img

def bilateral_filter(img):
    d,Scolor, Sspace=5,75,75
    h, w = img.shape[:2] # 獲取圖像的高度、寬度和通道數
  
    radius = d // 2# 計算半徑
   
    output = np.zeros_like(img, dtype=np.uint8)# 初始化輸出圖像

    space_kernel = np.zeros((d, d), dtype=np.float32)# 計算高斯核
    for i in range(d):
        for j in range(d):
            space_kernel[i, j] = np.exp(-((i - radius) ** 2 + (j - radius) ** 2) / (2 * Sspace ** 2))

    for i in range(h):# 遍歷每個像素
        for j in range(w):
            
            Wsum = 0# 初始化權重和加權總和
            pixel_value_sum = np.zeros(3)

            for m in range(-radius, radius + 1):# 遍歷窗口內的所有像素
                for n in range(-radius, radius + 1):
                    x = i + m
                    y = j + n
  
                    if x >= 0 and x < h and y >= 0 and y < w:# 確保不超出邊界
                        
                        Cdiff = np.linalg.norm(img[i, j] - img[x, y])# 計算顏色差異

                        Ckernel = np.exp(-Cdiff ** 2 / (2 * Scolor ** 2))# 計算顏色權重

                        space_kernel_val = space_kernel[m + radius, n + radius]# 計算空間權重

                        weight = Ckernel * space_kernel_val# 計算總權重

                        pixel_value_sum += weight * img[x, y]# 加權像素值
                        Wsum += weight

            output[i, j] = (pixel_value_sum / Wsum).astype(np.uint8) # 計算濾波後的像素值

    return output

def addsign(img):
    sign_img=cv2.imread("sign.png")
    Sh,Sw=sign_img.shape[:2]
    for i in range(Sh):
        for j in range(Sw):
            if(sign_img[i][j][0]==0)and(sign_img[i][j][1]==0)and(sign_img[i][j][2]==0):
                img[i][j][0]=0
                img[i][j][1]=0
                img[i][j][2]=0
    bilateral_filter(img)
    return img


def bigger_picture(img,h,w,c,scale):
    NH,NW = int(h * scale), int(w * scale)
    bigimg=resize_image(img,NH,NW,c,scale,scale)
    bigimg=addsign(bigimg)
    cv2.imshow("bigger", bigimg)
    cv2.imwrite("../bigger_picture.png",bigimg)

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