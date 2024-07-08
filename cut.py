from PIL import Image
import os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

def split_image_to_grid(image_path, output_dir, grid_size=(5, 5)):
    # 打开图像
    image = Image.open(image_path)
    image = image.transpose(Image.TRANSPOSE)

    # 如果图像有alpha通道，将其转换为RGB模式
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    img_width, img_height = image.size
    

    # 计算网格单元的宽度和高度
    cell_width = img_width // grid_size[0]
    cell_height = img_height // grid_size[1]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历网格，并保存每个网格单元为单独的图像
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 计算当前网格单元的坐标
            left = i * cell_width
            upper = j * cell_height
            right = left + cell_width
            lower = upper + cell_height

            # 裁剪图像
            grid_img = image.crop((left, upper, right, lower))

            # 构建输出文件名
            output_filename = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + f"_{i}{j}.png")

            # 保存裁剪后的图像为 * 格式
            grid_img.save(output_filename, 'PNG')

    # print("图像已成功划分并保存。")

def process_images_in_parallel(image_paths, output_dir, grid_size=(5, 5)):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 使用 multiprocessing.Pool 并行处理图像
    with Pool(4) as pool:
        # 使用 partial 函数来固定 split_image_to_grid 函数的部分参数
        func = partial(split_image_to_grid, output_dir=output_dir, grid_size=grid_size)
        for _ in tqdm(pool.imap_unordered(func, image_paths), total=len(image_paths)):
            pass


if __name__ == "__main__":
    # imgs = np.sort(glob.glob('/media/liushilei/DatAset/workspace/test/torch/data/nyc/temp_raw_tiff/*.jp2'))
    input_dir = '/media/liushilei/DatAset/workspace/test/torch/data/nyc/temp_raw_tiff'
    output_dir = "/media/liushilei/DatAset/workspace/test/torch/data/nyc/cut_data"  # 输出目录
    image_paths = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith(".jp2")]

    process_images_in_parallel(image_paths, output_dir, grid_size=(5, 5))
