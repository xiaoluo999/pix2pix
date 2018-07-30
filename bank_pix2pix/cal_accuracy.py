#计算准确率
import os
import cv2
import glob

def cal_iou(output_image, target_image):
    output_im = cv2.imread(output_image)
    target_im = cv2.imread(target_image)
    rows = output_im.shape[0]
    cols = output_im.shape[1]
    out_image_num = 0
    count = 0

    #并集面积
    for i in range(rows):
        for j in range(cols):
            if output_im[i][j][0] == 255 or target_im[i][j][0] == 255:
                out_image_num = out_image_num + 1
    #交集面积
    for i in range(rows):
        for j in range(cols):
            if output_im[i][j][0] == 255 and target_im[i][j][0] == 255:
                count = count + 1

    print("out_image_num: ", out_image_num)
    # print("target_iamge_num", target_iamge_num)
    print("count: ", count)
    if out_image_num!=0:
        iou = float(count) / float(out_image_num )
    else:
        iou = 1
    return iou


def cal_iou_batch(output_images_dir, target_images_dir):
    if not os.path.exists(output_images_dir):
        print("output iamges dir is not exist:", output_images_dir)
    if not os.path.exists(target_images_dir):
        print("target images dir is not exist:", target_images_dir)
    output_images_paths = glob.glob(os.path.join(output_images_dir, '*-outputs.png'))
    #    print("output_images_paths: ",output_images_paths)
    total_iou = 0
    for path in output_images_paths:
        name, _ = path.split(os.path.sep)[-1].split('-')
        target_images_path = os.path.join(target_images_dir, name + '-targets.png')
        #        print("target_images_path: ",target_images_path)
        if not os.path.exists(target_images_path):
            print("target iamge is not exist: ", target_images_path)
        per_iou = cal_iou(path, target_images_path)
        total_iou = total_iou + per_iou
        print("per_iou: ", per_iou)
    result_iou = total_iou / len(output_images_paths)
    print("iou is : ", result_iou)
    #30000次训练集测试结果，直接交集/并集=0.9175458220345062
    #40000次训练集测试结果，0.9127897406909379
if __name__ == "__main__":
    path = 'C:\\a.123.py'
    basename =os.path.basename(path)
    temp = os.path.splitext(basename)
    output_images_dir = "E:\\images\\images"
    target_images_dir = "E:\\images\\images"
    cal_iou_batch(output_images_dir, target_images_dir)