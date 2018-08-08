import tensorflow as tf
import cv2
import os
import glob
import numpy as np

#获取连通区域的外界矩形，以列表形式返回
def get_cc(im_bin):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(im_bin, connectivity,cv2.CV_32S)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    #output[2]为shape为(n,5)的ndarray，5个值分别为x,y,width,height,width*height
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    max_area_index = 0
    min_thresh = 15 * 15
    cc_list = []
    #从下标1开始，0为图像的大小
    for i in range(1, len(stats)):
        if stats[i][4] > min_thresh:
            cc_list.append(stats[i][:4])#list中存放着一维的ndarray
    return cc_list


def sort_roi_lit(roi_list):
    # first sort by y
    #改为按面积排序，由大到小
    roi_list = sorted(roi_list, key=lambda x: x[2]*x[3],reverse = True )
    return roi_list[0]

def get_roi_list(mask, log=False):
    ret, im_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cc_list = get_cc(im_bin)
    cc_list = sort_roi_lit(cc_list)
    return cc_list

if __name__ =="__main__":
    input_dir = r"E:\project\idcard_frontal_target_location\data\idcard-frontal-2346\idcard-frontal-2346"
    graph_path = r".\ckt\export.meta"
    saver = tf.train.import_meta_graph(graph_path)
    with tf.Session() as sess:
        saver.restore(sess, r"./ckt/export")
        input = sess.graph.get_tensor_by_name('Placeholder:0')
        output = sess.graph.get_tensor_by_name('strided_slice:0')
        if not os.path.exists(input_dir):
            raise Exception("cannot find %s"%input_dir)
        image_list = glob.glob(os.path.join(input_dir,"*.jpg"))
        if(len(image_list)==0):
            image_list = glob.glob(os.path.join(input_dir, "*.png"))
        if (len(image_list) == 0):
            image_list = glob.glob(os.path.join(input_dir, "*.bmp"))
        if (len(image_list)==0):
            raise Exception("cannot find image file")

        for file_path in image_list:
            # base_dir = r"E:\project\idcard_frontal_target_location\data\idcard-frontal-2346\idcard-frontal-2346"
            # file_name = os.path.basename(file_path)
            # src = cv2.imread(os.path.join(base_dir,file_name), 1)
            src = cv2.imread(file_path, 1)
            if not isinstance(src,np.ndarray):
                continue
            if src.ndim == 3 and src.shape[2] ==3:
                src = cv2.resize(src,dsize=(256,256))
                bin_image = sess.run(output, feed_dict={input: src})
                x,y,width,height = get_roi_list(bin_image)
                crop = src[y:y+height,x:x+width]
                cv2.imshow("src",src)
                cv2.imshow("bin", bin_image)
                cv2.imshow("crop",crop)
                cv2.waitKey(0)
        print("test finished")
