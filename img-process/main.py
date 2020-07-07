import os
import cv2
import rotate
import classify
import segment_img


def main():
    """从文件夹中读取图片，依次将图片旋转，分类，单元格分割和行分割，将最终分割的结果保存到另一个文件夹内"""

    # directory setting
    data_dir = './data/'                      # 根目录
    original_dir = data_dir + 'original'      # 原图目录
    rotated_dir = data_dir + 'rotated'        # 旋转后图片目录
    classified_dir = data_dir + 'classified'  # 分类图片目录
    segmented_dir = data_dir + 'segmented'    # 分割的图片目录

    # 创建目录
    dir_list = [original_dir, rotated_dir, classified_dir, segmented_dir]
    for each_dir in dir_list:
        if not os.path.exists(each_dir):
            os.makedirs(each_dir)

    try:  # 获取文件名, 如果文件目录不存在或为空则提示将图片存入
        original_names = os.listdir(original_dir)
    except FileNotFoundError:
        print('PLEASE PUT BGS BOREHOLE DATA/IMAGES INTO "{}"'.format(original_dir))
        return
    else:
        if not os.listdir(original_dir):
            print('PLEASE PUT BGS BOREHOLE DATA/IMAGES INTO "{}"'.format(original_dir))
            return

    print('processing...')
    with open(segmented_dir + '/cell.txt', 'w') as f:
        for i, original_name in enumerate(original_names):
            if original_name.split('.')[-1] != 'png':  # 抛出系统隐藏文件异常
                continue
            original_image = cv2.imread(original_dir + '/{}'.format(original_name))

            """旋转"""
            rotated_image = rotate.rotation(original_image)
            # 保存旋转后的图片
            cv2.imwrite(rotated_dir + '/{}'.format(original_name), rotated_image)

            """分类"""
            image_type = classify.ctr_detect(rotated_image)
            # 将不同类别图片保存在不同目录下
            if not os.path.exists(classified_dir + '/' + image_type):
                os.makedirs(classified_dir + '/' + image_type)
            cv2.imwrite(classified_dir + '/' + image_type + '/{}'.format(original_name), rotated_image)

            """图片分割(单元格分割)"""
            if image_type == 'dashed-line':  # 虚线
                line_fit_image = segment_img.dots2line(rotated_image)      # 直线拟合
                table_fit_image = segment_img.fit_table(line_fit_image)    # 表格拟合
                coordinate_cells = segment_img.table_out(table_fit_image)  # 单元格提取
                threshold = 66
            elif image_type == 'tabular':  # 表格
                table_fit_image = segment_img.fit_table(rotated_image)     # 表格拟合
                coordinate_cells = segment_img.table_out(table_fit_image)  # 单元格提取
                threshold = 66
            elif image_type == 'non-tabular':  # 无表格
                table_fit_image = segment_img.fit_table(rotated_image)     # 表格拟合
                coordinate_cells = segment_img.table_out(table_fit_image)  # 单元格提取
                coordinate_cells = segment_img.get_n_img_cell(rotated_image, coordinate_cells)
                threshold = 40
            # 以图片名创建目录用与存储每个segmentation
            if not os.path.exists(segmented_dir + '/' + original_name.split('.')[0]):
                os.makedirs(segmented_dir + '/' + original_name.split('.')[0])

            # 写入每个cell
            for j, cell in enumerate(coordinate_cells):
                # 文件名+索引+textType+坐标 保存到txt文档
                y1, y2, x1, x2 = cell[1], cell[1] + cell[3], cell[0], cell[0] + cell[2]
                cell_name = original_name.split('.')[0] + '-' + '{:0>3d}'.format(j) + '.' + original_name.split('.')[-1]

                if segment_img.has_text(rotated_image[y1:y2, x1:x2], image_type) is True:  # 粗略判断单元格内有无文本
                    if cell[-1] <= threshold:
                        f.write(cell_name + ' SingleText ')
                    else:
                        f.write(cell_name + ' MultipleText ')
                else:
                    f.write(cell_name + ' NoneText ')
                y_extend = 8 # 向下多取8像素
                bottom_indent = cell[3] + y_extend if cell[1] + cell[3] + y_extend <= rotated_image.shape[0] else rotated_image.shape[0] - cell[1]  # 三元表达式防止越界
                cell[3] = bottom_indent
                f.write(str(tuple(cell)))
                f.write('\r\n')
                # 保存分割图
                seg_img = rotated_image[y1:(y2 + y_extend), x1:x2]
                img_name = segmented_dir + '/' + original_name.split('.')[0] + '/' + cell_name
                cv2.imwrite(img_name, seg_img)




if __name__ == '__main__':
    main()
