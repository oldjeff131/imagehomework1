import cv2
import numpy as np

def inverse_mapping_resize(image, scale_factor):
    """
    使用逆向映射方式縮放影像
    :param image: 原始影像 (numpy array)
    :param scale_factor: 縮放比例
    :return: 縮放後的影像
    """
    original_height, original_width = image.shape[:2]
    
    # 計算新影像尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 建立新影像
    resized_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # 計算縮放比例
    scale_x = original_width / new_width
    scale_y = original_height / new_height
    
    for y in range(new_height):
        for x in range(new_width):
            # 反向映射到原影像的座標
            src_x = x * scale_x
            src_y = y * scale_y
            
            # 進行雙線性插值
            x0 = int(np.floor(src_x))
            x1 = min(x0 + 1, original_width - 1)
            y0 = int(np.floor(src_y))
            y1 = min(y0 + 1, original_height - 1)
            
            a = src_x - x0
            b = src_y - y0
            
            # 插值計算 (雙線性插值公式)
            top = (1 - a) * image[y0, x0] + a * image[y0, x1]
            bottom = (1 - a) * image[y1, x0] + a * image[y1, x1]
            pixel_value = (1 - b) * top + b * bottom
            
            resized_image[y, x] = pixel_value.astype(np.uint8)
    
    return resized_image

# 測試程式
if __name__ == "__main__":
    image = cv2.imread("./sign.jpg", 0)  # 讀取影像
    scale_factor = 1.5  # 設定縮放比例
    resized_image = inverse_mapping_resize(image, scale_factor)
    
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output.jpg", resized_image)