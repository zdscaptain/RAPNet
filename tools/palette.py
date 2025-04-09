import os
import numpy as np
from PIL import Image

# 定义调色板 (此调色板为两个颜色，黑色和白色)
palette = [0, 0, 0,  # 黑色
           0, 0, 255]  # 白色

# 原始图像文件夹路径
path = r"D:\code\u-mixformer-main\work_dirs\format_results"
# 保存处理后图像的文件夹路径
new = r'D:\code\u-mixformer-main\work_dirs\maskformer_format_results_rgb'

# 如果输出文件夹不存在，则创建它
if not os.path.exists(new):
    os.mkdir(new)

# 获取输入文件夹中的所有文件
li = os.listdir(path)


# 定义用于转换掩膜的函数
def colorize_mask(mask, palette):
    # 去除图像的单通道维度，将图像从 (H, W, 1) 变为 (H, W)
    mask = np.squeeze(mask, axis=2)

    # 计算调色板需要填充的零值，以使其长度达到 256 * 3
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    # 将NumPy数组转换为PIL图像对象
    new_mask = Image.fromarray(mask.astype(np.uint8))
    # 将图像转换为调色板模式
    new_mask = new_mask.convert('P')
    # 应用调色板
    new_mask.putpalette(palette)

    return new_mask


# 遍历输入文件夹中的每个文件
for i in li:
    # 构造文件的完整路径
    file = os.path.join(path, i)
    # 打开图像并转换为NumPy数组
    mask = np.array(Image.open(file))
    # 增加一个通道维度，从 (H, W) 变为 (H, W, 1)
    mask = np.expand_dims(mask, axis=2)
    # 调用colorize_mask函数将灰度图转换为带调色板的彩色图像
    colorized_mask = colorize_mask(mask, palette)
    # 保存处理后的图像到指定的输出文件夹中，并将文件扩展名改为 .png
    colorized_mask.save(os.path.join(new, i[:-3] + 'png'))

print("所有图像已成功转换并保存！")
