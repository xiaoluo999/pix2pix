import cv2
import os
import numpy as np
import glob

def DataAugmentation(input_dir,output_dir):
    if not os.path.isdir(input_dir):
        print("cannot find %s"%input_dir)
        return
    input_paths = glob.glob(os.path.join(input_dir, "*.jpg"))#获取所有图像路径
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.bmp"))
    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")
    for file_path in input_paths:
        src_image = cv2.imread(file_path)
        file_name = file_path.strip().split('\\')[-1]
        # 左右翻转
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_left_right_flip.jpg"
        flip_leftright__image =  cv2.flip(src_image,1)
        cv2.imwrite(save_path,flip_leftright__image)

        # 上下翻转
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_up_down_flip.jpg"
        flip_updown_image = cv2.flip(src_image,0)
        cv2.imwrite(save_path,flip_updown_image)

        # 旋转10度
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_10_rotate.jpg"
        rows = src_image.shape[0]
        cols = src_image.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
        rotate_10_image = cv2.warpAffine(src_image, M, (cols, rows))
        cv2.imwrite(save_path,rotate_10_image)

        # 旋转-10度
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_350_rotate.jpg"
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -10, 1)
        rotate_350_image = cv2.warpAffine(src_image, M, (cols, rows))
        cv2.imwrite(save_path, rotate_350_image)

        # 旋转90度
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_90_rotate.jpg"
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        rotate_90_image = cv2.warpAffine(src_image, M, (cols, rows))
        cv2.imwrite(save_path, rotate_90_image)

        #旋转180度
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_180_rotate.jpg"
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        rotate_180_image = cv2.warpAffine(src_image, M, (cols, rows))
        cv2.imwrite(save_path, rotate_180_image)

        # 旋转270度
        save_path = output_dir + "\\" + file_name.split('.')[0] + "_270_rotate.jpg"
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 270, 1)
        rotate_270_image = cv2.warpAffine(src_image, M, ( cols, rows))
        cv2.imwrite(save_path, rotate_270_image)

#要求input_rgb_dir图像为24位，input_gray_dir为8位，out_merge_dir为四张图像，r、g、b、gray
def create_merge(input_rgb_dir,input_gray_dir,out_merge_dir):
    if os.path.exists(input_rgb_dir)==False:
        print("cannot find %s"%input_rgb_dir)
    if os.path.exists(input_gray_dir) == False:
        print("cannot find %s" % input_gray_dir)
    if os.path.exists(out_merge_dir) == False:
        os.makedirs(out_merge_dir)
    for file_name in os.listdir(input_rgb_dir):
        rgb_file_path = os.path.join(input_rgb_dir,file_name)
        input_gray_path = os.path.join(input_gray_dir,file_name)
        w = h = 256
        src_image = cv2.imread(rgb_file_path)
        label_image =cv2.imread(input_gray_path)
        merge_image = np.zeros(shape=[h, w * 4, 1])
        merge_image[:, :w, 0] = src_image[:, :, 0]
        merge_image[:, w:2 * w, 0] = src_image[:, :, 1]
        merge_image[:, 2 * w:3 * w, 0] = src_image[:, :, 1]
        merge_image[:, 3 * w:4 * w, 0] = label_image[:, :, 0]

        save_path = os.path.join(out_merge_dir,file_name)
        cv2.imwrite(save_path, merge_image)

def create_label(inupt_file_path,src_dir,label_dir):
    if not os.path.exists(inupt_file_path):
        print("cannot find : %s"%inupt_file_path)
        return
    if not os.path.isdir(src_dir):
        os.makedirs(src_dir)#用于递归创建目录
    if not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    with open(inupt_file_path) as file:
        content = file.readlines()
        for line in content:
            oneline_list = line.strip().split(' ')
            file_name = oneline_list[0].split('\\')[-1]
            image_path = oneline_list[0]

            src_image = cv2.imread(image_path)
            h = src_image.shape[0]
            w = src_image.shape[1]

            #生成label图像
            label_image = np.zeros(shape=[h, w, 1], dtype=np.uint8)
            mask = np.zeros((h + 2, w + 2), np.uint8)
            x1 = int(oneline_list[1])
            y1 = int(oneline_list[2])
            x2 = int(oneline_list[3])
            y2 = int(oneline_list[4])
            x3 = int(oneline_list[5])
            y3 = int(oneline_list[6])
            x4 = int(oneline_list[7])
            y4 = int(oneline_list[8])
            points_list = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            cv2.polylines(label_image, [points_list], True, (255))
            center_x = int((x1 + x2 + x3 + x4) / 4.0)
            center_y = int((y1 + y2 + y3 + y4) / 4.0)
            cv2.floodFill(label_image, mask, (center_x, center_y), 255)

            # 全部归一化
            w = 256
            h = 256
            src_image = cv2.resize(src_image, dsize=(w, h))
            label_image = cv2.resize(label_image, dsize=(w, h))
            label_image = np.resize(label_image, new_shape=(w, h, 1))

            #保存图像
            image_path = os.path.join(src_dir,file_name)
            cv2.imwrite(image_path, src_image)
            image_path = os.path.join(label_dir,file_name)
            cv2.imwrite(image_path, label_image)
    # cv2.waitKey(0)

# if __name__ =="__main__":
#     #1.生成图像和label文件夹
#     #file_path = "E:\project\标注工具\mark_tool_release\mark_tool_release\pts_out_file.txt"
#     #create_label(file_path,'.\\test\\src_image','.\\test\\label_image')
#     #2.对input和label文件数据增强
#     DataAugmentation(r'.\test\src_image',r'.\test\src_image')
#     DataAugmentation(r'.\test\label_image', r'.\test\label_image')
#     #3.生成合并图像
#     create_merge(r'.\test\src_image',r'.\test\label_image',r'.\test\merge_image')
