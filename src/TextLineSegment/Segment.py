import os
import cv2

import rotate
import classify
import segmentImg
import segmentLine


class FilePaths:
    "filenames and paths to data"
    data_dir = '../../data/'  # 根目录
    original_dir = data_dir + 'BGS-records'   # 原图路径
    line_seg_dir = data_dir + 'line-segment'  # 行分割路径
    word_seg_dir = data_dir + 'word-segment'  # 词分割路径
    # line_seg_dir = data_dir + 'line-segment-preprocessed'  # 行分割路径
    # word_seg_dir = data_dir + 'word-segment-preprocessed'  # 词分割路径


def main():
    """从文件夹中读取图片，依次将图片旋转，分类，单元格分割和行分割，将最终分割的结果保存到另一个文件夹内"""

    # 创建目录
    if not os.path.exists(FilePaths.line_seg_dir):
        os.makedirs(FilePaths.line_seg_dir)

    try:  # 获取文件名, 如果文件目录不存在或为空则提示将图片存入
        original_names = os.listdir(FilePaths.original_dir)
    except FileNotFoundError:
        print('PLEASE PUT BGS BOREHOLE DATA/IMAGES INTO "{}"'.format(FilePaths.original_dir))
        return
    else:
        if not os.listdir(FilePaths.original_dir):
            print('PLEASE PUT BGS BOREHOLE DATA/IMAGES INTO "{}"'.format(FilePaths.original_dir))
            return

    print('start segmenting...')
    with open(FilePaths.line_seg_dir + '/lines.txt', 'w') as f:
        for i, original_name in enumerate(original_names):
            if original_name.split('.')[-1] != 'png':  # 抛出系统隐藏文件异常
                continue
            # 以图片名创建目录用于存储每个segmentation
            if not os.path.exists(FilePaths.line_seg_dir + '/' + original_name.split('.')[0]):
                os.makedirs(FilePaths.line_seg_dir + '/' + original_name.split('.')[0])
            # 获取图片
            original_image = cv2.imread(FilePaths.original_dir + '/{}'.format(original_name))

            """旋转"""
            rotated_image = rotate.rotation(original_image)

            """分类"""
            image_type = classify.ctr_detect(rotated_image)

            """图片分割(单元格分割)"""
            if image_type == classify.image_type1:  # 虚线
                line_fit_image = segmentImg.dots2line(rotated_image)  # 直线拟合
                table_fit_image = segmentImg.fit_table(line_fit_image)  # 表格拟合
                coordinate_cells = segmentImg.table_out(table_fit_image)  # 获取单元格坐标
                h_threshold = 66
            elif image_type == classify.image_type3:  # 表格
                table_fit_image = segmentImg.fit_table(rotated_image)  # 表格拟合
                coordinate_cells = segmentImg.table_out(table_fit_image)  # 获取单元格坐标
                h_threshold = 40
            elif image_type == classify.image_type2:  # 无表格
                table_fit_image = segmentImg.fit_table(rotated_image)  # 表格拟合
                coordinate_cells = segmentImg.table_out(table_fit_image)  # 获取单元格坐标
                coordinate_cells = segmentImg.get_n_img_cell(rotated_image, coordinate_cells)
                h_threshold = 40

            # 将每个cell/line写入文档并保存
            name_index = 0
            for j, cell in enumerate(coordinate_cells):
                y1, y2, x1, x2 = cell[1], cell[1] + cell[3], cell[0], cell[0] + cell[2]
                cell_name = original_name.split('.')[0] + '-' + '{:0>3d}'.format(name_index) + '.' + \
                            original_name.split('.')[-1]
                if segmentImg.has_text(rotated_image[y1:y2, x1:x2], image_type) is True:  # 单元格内有文本
                    if segmentImg.has_multiple_text(cell[-1], h_threshold) is False:  # 单行文本
                        y_extend = 8  # 向下多取8像素
                        h_extend = cell[3] + y_extend if cell[1] + cell[3] + y_extend <= rotated_image.shape[0] else \
                            rotated_image.shape[0] - cell[1]  # 三元表达式防止越界
                        cell[3] = h_extend
                        """写入txt"""
                        f.write(cell_name + ' ' + str(tuple(cell)) + '\r\n')  # 格式: 文件名 + 坐标
                        cell_image = rotated_image[y1:(y1 + h_extend), x1:x2]
                        cell_image_filename = FilePaths.line_seg_dir + '/' + original_name.split('.')[0] + '/' + cell_name
                        """保存图片"""
                        cv2.imwrite(cell_image_filename, cell_image)
                        name_index += 1
                    else:  # 多行文本 进一步分割
                        """行分割"""
                        cell_image = rotated_image[y1:y2, x1:x2]
                        if (cell_image.shape[0] > 1000 and cell_image.shape[1] > 200) or (
                                cell_image.shape[1] > 600 and cell_image.shape[0] > 100):
                            if 197 < cell_image.shape[0] < 500 and cell_image.shape[1] > 1000:  # 表头
                                pass
                            elif 180 < cell_image.shape[0] < 200 and 630 < cell_image.shape[1] < 700:  # 小表头
                                cell_img_cpy = cell_image.copy()
                                cell_image = segmentLine.rm_vertical_line(cell_image)
                                cell_image = segmentLine.rm_horizontal_line(cell_image)
                                cell_image = segmentLine.rm_noise(cell_image)
                                line_ctrs = segmentLine.header_segment(cell_img_cpy, cell_image)
                                for line in line_ctrs:
                                    line_name = original_name.split('.')[0] + '-' + '{:0>3d}'.format(name_index) + '.' + \
                                                original_name.split('.')[-1]
                                    y1_l, y2_l, x1_l, x2_l = line[1], line[1] + line[3], line[0], line[0] + line[2]
                                    # 将分割图坐标更新为在整张图上的坐标
                                    line[0] = line[0] + x1
                                    line[1] = line[1] + y1
                                    """写入txt"""
                                    f.write(line_name + ' ' + str(tuple(line)) + '\r\n')
                                    """保存图片"""
                                    line_image = cell_img_cpy[y1_l:y2_l, x1_l:x2_l]
                                    line_image_filename = FilePaths.line_seg_dir + '/' + original_name.split('.')[
                                        0] + '/' + line_name
                                    cv2.imwrite(line_image_filename, line_image)
                                    name_index += 1
                            else:  # 文本图
                                cell_img_cpy = cell_image.copy()
                                cell_image = segmentLine.rm_vertical_line(cell_image)
                                cell_image = segmentLine.rm_horizontal_line(cell_image)
                                cell_image = segmentLine.rm_noise(cell_image)
                                line_text_imgs, line_text_ctrs = segmentLine.line_segment(cell_img_cpy, cell_image)
                                temporary_index = name_index
                                for ltc in line_text_ctrs:
                                    # 将分割图坐标更新为在整张图上的坐标
                                    ltc[0] = ltc[0] + x1
                                    ltc[1] = ltc[1] + y1
                                    """写入txt"""
                                    temporary_line_name = original_name.split('.')[0] + '-' + '{:0>3d}'.format(
                                        temporary_index) + '.' + original_name.split('.')[-1]
                                    f.write(temporary_line_name + ' ' + str(tuple(ltc)) + '\r\n')
                                    temporary_index += 1
                                for lti in line_text_imgs:
                                    """保存图片"""
                                    line_name = original_name.split('.')[0] + '-' + '{:0>3d}'.format(name_index) + '.' + \
                                                original_name.split('.')[-1]
                                    line_image_filename = FilePaths.line_seg_dir + '/' + original_name.split('.')[
                                        0] + '/' + line_name
                                    cv2.imwrite(line_image_filename, lti)
                                    name_index += 1
                        else:  # 数字图
                            area_th = 200 if cell_image.shape[0] > 1000 and cell_image.shape[1] > 30 else 400
                            cell_img_cpy = cell_image.copy()
                            cell_image = segmentLine.rm_vertical_line(cell_image)
                            cell_image = segmentLine.rm_horizontal_line(cell_image)
                            cell_image = segmentLine.rm_noise(cell_image)
                            line_ctrs = segmentLine.num_img_segment(cell_img_cpy, cell_image, area_th)
                            for line in line_ctrs:
                                line_name = original_name.split('.')[0] + '-' + '{:0>3d}'.format(name_index) + '.' + \
                                            original_name.split('.')[-1]
                                y1_l, y2_l, x1_l, x2_l = line[1], line[1] + line[3], line[0], line[0] + line[2]
                                # 将分割图坐标更新为在整张图上的坐标
                                line[0] = line[0] + x1
                                line[1] = line[1] + y1
                                """写入txt"""
                                f.write(line_name + ' ' + str(tuple(line)) + '\r\n')
                                """保存图片"""
                                line_image = cell_img_cpy[y1_l:y2_l, x1_l:x2_l]
                                line_image_filename = FilePaths.line_seg_dir + '/' + original_name.split('.')[0] + '/' + line_name
                                cv2.imwrite(line_image_filename, line_image)
                                name_index += 1
                else:  # 单元格内无文本 直接忽略
                    pass


if __name__ == '__main__':
    main()
