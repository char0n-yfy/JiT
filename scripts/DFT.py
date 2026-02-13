import cv2
import numpy as np
import matplotlib.pyplot as plt

def center_crop(img, crop_size=None):
    """
    对图像进行中心裁剪。
    :param img: 输入图像 (H, W, C)
    :param crop_size: 裁剪尺寸 (h, w) 或整数 (s, s)。
                      如果为 None，则裁剪为图像中心的最大正方形。
    :return: 裁剪后的图像
    """
    h, w = img.shape[:2]
    
    # 如果未指定尺寸，默认裁剪为最大正方形
    if crop_size is None:
        short_edge = min(h, w)
        dy = (h - short_edge) // 2
        dx = (w - short_edge) // 2
        return img[dy:dy+short_edge, dx:dx+short_edge]
    
    # 如果指定了尺寸
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size

    # 确保裁剪尺寸不超过原图
    crop_h, crop_w = min(crop_h, h), min(crop_w, w)
    
    # 计算中心位置
    dy = (h - crop_h) // 2
    dx = (w - crop_w) // 2
    
    return img[dy:dy+crop_h, dx:dx+crop_w]

def analyze_spectrum_with_crop(image_path, crop_size=None):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取路径 {image_path} 下的图像")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # --- 预处理：中心裁剪 ---
    # crop_size 可以是具体数值（如 224），也可以是 None（自动裁剪为正方形）
    cropped_img = center_crop(img_rgb, crop_size)
    print(f"原图尺寸: {img_rgb.shape}, 裁剪后尺寸: {cropped_img.shape}")
    
    # 2. 分离通道 (使用裁剪后的图像)
    channels = cv2.split(cropped_img)
    channel_names = ['Red', 'Green', 'Blue']
    
    magnitude_spectra = []
    
    # 3. 计算频谱
    for channel in channels:
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        magnitude_spectra.append(magnitude_spectrum)
    
    # 4. 可视化
    plt.figure(figsize=(15, 10))
    
    # --- 第一行：三个通道的独立频谱 ---
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(magnitude_spectra[i], cmap='gray')
        plt.title(f'{channel_names[i]} Channel Spectrum')
        plt.axis('off')
        
    # --- 第二行：合成频谱、平均频谱、裁剪后的原图 ---
    
    # 合成彩色频谱
    norm_spectra = []
    for mag in magnitude_spectra:
        m_min, m_max = mag.min(), mag.max()
        norm = (mag - m_min) / (m_max - m_min)
        norm_spectra.append(norm)
    color_spectrum = np.dstack(norm_spectra)
    
    plt.subplot(2, 3, 4)
    plt.imshow(color_spectrum)
    plt.title('Synthesized Color Spectrum')
    plt.axis('off')
    
    # 平均频谱
    avg_spectrum = np.mean(magnitude_spectra, axis=0)
    plt.subplot(2, 3, 5)
    plt.imshow(avg_spectrum, cmap='gray')
    plt.title('Averaged Spectrum')
    plt.axis('off')
    
    # *** 输出裁剪后的图像 ***
    plt.subplot(2, 3, 6)
    plt.imshow(cropped_img)
    plt.title(f'Center Cropped Image\n{cropped_img.shape}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 使用示例
# 1. 自动裁剪为最大正方形
analyze_spectrum_with_crop('test/COCO_test2014_000000000191.jpg') 

# 2. 指定裁剪大小 (例如 224x224)
# analyze_spectrum_with_crop('your_image.jpg', crop_size=224)