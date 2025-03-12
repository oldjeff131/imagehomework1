import cv2
import numpy as np

def rotate_image(img, theta):#原圖、旋轉角度
    h, w, c = img.shape# 圖像的尺寸
    theta=float(theta)# 確保角度是浮點數類型
    theta_rad = np.deg2rad(theta)# 將角度轉為弧度
    cos_theta = np.cos(theta_rad)# 計算旋轉矩陣
    sin_theta = np.sin(theta_rad)
    NW = int(np.abs(w * cos_theta) + np.abs(h * sin_theta))# 計算旋轉後的圖像大小
    NH = int(np.abs(h * cos_theta) + np.abs(w * sin_theta))
    R_img = np.zeros((NH, NW, c), dtype=np.uint8)# 創建一個新的空白圖像來存放旋轉後的結果
    Cx, Cy = w // 2, h // 2# 計算圖像中心
    NCx, NCy = NW // 2, NH // 2
    # 遍歷新圖像中的每個像素
    for i in range(NH):
        for j in range(NW):          
            x = j - NCx# 計算新圖像的像素對應到原圖的像素位置
            y = i - NCy     
            orig_x = cos_theta * x + sin_theta * y + Cx # 逆旋轉操作 (旋轉回去)
            orig_y = -sin_theta * x + cos_theta * y + Cy
            orig_x = int(np.clip(orig_x, 0, w - 1))# 確保原始圖像坐標在合法範圍內
            orig_y = int(np.clip(orig_y, 0, h - 1))
            R_img[i, j] = img[orig_y, orig_x]# 複製像素值
    return R_img
image = cv2.imread("image.jpg")# 讀取圖像
theta=input("請輸入一個0~360的數字：")
try: #防呆機制
    rotated_image = rotate_image(image, theta)# 旋轉圖像  
    cv2.imshow("Original Image", image)# 顯示原始圖像和旋轉後的圖像
    cv2.imshow(f"Rotated Image ({theta} degrees)", rotated_image)
    cv2.imwrite("../Rotated Image.png", rotated_image)
    cv2.waitKey(0)# 等待按鍵關閉
    cv2.destroyAllWindows()
except:
    print("輸入錯誤")
