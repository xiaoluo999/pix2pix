import numpy as np
import cv2
import sys

platform = sys.platform
if platform == 'win32':
    from matplotlib import pyplot as plt


def binarize(im, show=False):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, dst = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show:
        cv2.imshow('bin', dst)
        cv2.waitKey(0)
    return dst


def binarize_small(im, height_persentage=1, show=False):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    width = im_gray.shape[1]
    height = im_gray.shape[0]
    if show:
        print(im_gray.shape)
    dh = int(height * height_persentage)
    dw = dh
    nw = int(width // dw)
    nh = int(height // dh)
    ret, dst = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dst_block = np.copy(dst)
    for w in range(nw):
        for h in range(nh):
            ret, dst_block[dh * h:dh * (h + 1), dw * w:dw * (w + 1)] = cv2.threshold(
                im_gray[dh * h:dh * (h + 1), dw * w:dw * (w + 1)], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, dst_block[:, -dh:] = cv2.threshold(im_gray[:, -dh:], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if show:
        cv2.imshow('bin', dst)
        cv2.imshow('bin_block', dst_block)
        cv2.waitKey(0)
    return dst_block


def image_close(im, width, height, show=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    closed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    if show:
        cv2.imshow('closed', closed)
        cv2.waitKey(0)
    return closed


def image_open(im, width, height, show=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    opened = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    if show:
        cv2.imshow('opened', opened)
        cv2.waitKey(0)
    return opened


def get_boundray(histogram, space):
    space = int(space)
    max_density = 0
    max_index = 0
    for i in range(len(histogram) - space):
        density_tmp = sum(histogram[i:i + space])
        if density_tmp > max_density:
            max_density = density_tmp
            max_index = i
    return max_index


def cc_analyze(im_bin, show=False):
    connectivity = 4
    # Perform the operation
    output = cv2.connectedComponentsWithStats(im_bin, connectivity, cv2.CV_16U)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    if show:
        cv2.imshow('cc', np.array(labels, dtype=np.uint8))
        cv2.waitKey(0)


def get_roi_with_perspective(im_path, target, border=0.1):
    target = np.float32(target).reshape([4, 2])
    target_left_top = target[0]
    target_right_top = target[1]
    target_right_bottom = target[2]
    target_left_bottom = target[3]
    target_top = (target_right_top - target_left_top)[0]
    target_bottom = (target_right_bottom - target_left_bottom)[0]
    target_left = (target_left_bottom - target_left_top)[1]
    target_right = (target_right_bottom - target_right_top)[1]
    border = border
    new_target_left_top = target[0] + np.array([-target_top * border, -target_left * border])
    new_target_right_top = target[1] + np.array([target_top * border, -target_right * border])
    new_target_right_bottom = target[2] + np.array([target_bottom * border, target_right * border])
    new_target_left_bottom = target[3] + np.array([-target_bottom * border, target_left * border])

    new_target_top = target_top + target_top * border * 2
    new_target_bottom = target_bottom + target_bottom * border * 2
    new_target_left = target_left + target_left * border * 2
    new_target_right = target_right + target_right * border * 2

    # new_width=(new_target_top+new_target_bottom)/2
    # new_height=(new_target_left+new_target_right)/2
    new_width = 560
    new_height = 400
    src = np.float32([new_target_left_top, new_target_right_top, new_target_right_bottom, new_target_left_bottom])

    dst = np.float32([0, 0, new_width, 0, new_width, new_height, 0, new_height]).reshape([4, 2])

    im = cv2.imread(im_path)
    perspective_mat = cv2.getPerspectiveTransform(src, dst)
    im_perspective = cv2.warpPerspective(im, perspective_mat, (int(new_width), int(new_height)))

    in_point = target.reshape([1, 4, 2])
    out_point = cv2.perspectiveTransform(in_point, perspective_mat)
    out_point = np.array(out_point, dtype=np.float32).reshape([-1])
    return im_perspective, out_point


def get_roi_with_border(im, target, border=0.5):
    log = False
    if log:
        print(target)
    width = im.shape[1]
    height = im.shape[0]

    points = np.array(target, dtype=int).reshape([4, 2])
    p_x_min = points.min(0)[0]
    p_y_min = points.min(0)[1]
    p_x_max = points.max(0)[0]
    p_y_max = points.max(0)[1]

    left = p_x_min
    right = p_x_max
    top = p_y_min
    bottom = p_y_max
    target_width = right - left
    target_height = bottom - top

    x_border = int(target_width * border)
    y_border = int(target_height * border)

    new_left = left - x_border
    if new_left < 0:
        new_left = 0
    new_top = top - y_border
    if new_top < 0:
        new_top = 0
    new_right = right + x_border
    if new_right > width:
        new_right = width
    new_bottom = bottom + y_border
    if new_bottom > height:
        new_bottom = height

    im = im[:, new_left:new_right]
    im = im[new_top:new_bottom, :]

    shift = np.array([new_left, new_top], dtype=int)
    for i in range(len(points)):
        points[i] = points[i] - np.array(shift)

    new_target_points = np.array(points, dtype=np.float32).reshape([-1])

    return im, new_target_points, shift


def get_clear_sample(im, target):
    target = np.float32(target).reshape([4, 2])
    target_left_top = target[0]
    target_right_top = target[1]
    target_right_bottom = target[2]
    target_left_bottom = target[3]
    target_top = (target_right_top - target_left_top)[0]
    target_bottom = (target_right_bottom - target_left_bottom)[0]
    target_left = (target_left_bottom - target_left_top)[1]
    target_right = (target_right_bottom - target_right_top)[1]
    border = 0.2
    new_target_left_top = target[0] + np.array([-target_top * border, -target_left * border])
    new_target_right_top = target[1] + np.array([target_top * border, -target_right * border])
    new_target_right_bottom = target[2] + np.array([target_bottom * border, target_right * border])
    new_target_left_bottom = target[3] + np.array([-target_bottom * border, target_left * border])

    new_target_top = target_top + target_top * border * 2
    new_target_bottom = target_bottom + target_bottom * border * 2
    new_target_left = target_left + target_left * border * 2
    new_target_right = target_right + target_right * border * 2

    new_width = (new_target_top + new_target_bottom) / 2
    new_height = (new_target_left + new_target_right) / 2
    # new_width=560
    # new_height=400
    src = np.float32([new_target_left_top, new_target_right_top, new_target_right_bottom, new_target_left_bottom])

    dst = np.float32([0, 0, new_width, 0, new_width, new_height, 0, new_height]).reshape([4, 2])

    perspective_mat = cv2.getPerspectiveTransform(src, dst)
    im_perspective = cv2.warpPerspective(im, perspective_mat, (int(new_width), int(new_height)))

    in_point = target.reshape([1, 4, 2])
    out_point = cv2.perspectiveTransform(in_point, perspective_mat)
    out_point = np.array(out_point, dtype=np.float32).reshape([-1])
    return im_perspective, out_point


def get_original_coordinate(target, shift, scale):
    target = np.array(target, dtype=np.float32).reshape([-1, 2])
    for i in range(len(target)):
        target[i] = target[i] / scale
        target[i] = target[i] - shift
    return target


def shift_image_point(im, target):
    log = False
    width = im.shape[1]
    height = im.shape[0]

    points = np.array(target, dtype=int).reshape([4, 2])
    p_x_min = points.min(0)[0]
    p_y_min = points.min(0)[1]
    p_x_max = points.max(0)[0]
    p_y_max = points.max(0)[1]

    left_border = p_x_min
    right_border = width - p_x_max
    top_border = p_y_min
    bottom_border = height - p_y_max

    x_shift = -left_border // 4
    y_shift = 0

    im_shift = im
    BLACK = [0, 0, 0]
    if y_shift < 0:
        im_shift = im_shift[-y_shift:, :]
        im_shift = cv2.copyMakeBorder(im_shift, 0, -y_shift, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    else:
        im_shift = im_shift[:height - y_shift, :]
        im_shift = cv2.copyMakeBorder(im_shift, y_shift, 0, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    if x_shift < 0:
        im_shift = im_shift[:, -x_shift:]
        im_shift = cv2.copyMakeBorder(im_shift, 0, 0, 0, -x_shift, cv2.BORDER_CONSTANT, value=BLACK)
    else:
        im_shift = im_shift[:, :width - x_shift]
        im_shift = cv2.copyMakeBorder(im_shift, 0, 0, x_shift, 0, cv2.BORDER_CONSTANT, value=BLACK)

    points_shift = points + np.array([[x_shift, y_shift]] * 4, dtype=int)

    if log:
        cv2.imshow('shift', im_shift)
        cv2.waitKey(0)
    return im_shift, points_shift.reshape([-1])


def get_target_template():
    return np.array([[0, 0], [FLAGS.image_width, 0], [FLAGS.image_width, FLAGS.image_height], [0, FLAGS.image_height]],
                    dtype=np.float32)


def get_roi(im, target, mode='4points', log=False):
    if mode == '4points':
        width = im.shape[1]
        height = im.shape[0]

        points = np.array(target, dtype=int).reshape([4, 2])
        p_x_min = points.min(0)[0]
        p_y_min = points.min(0)[1]
        p_x_max = points.max(0)[0]
        p_y_max = points.max(0)[1]

        left = p_x_min
        right = p_x_max
        top = p_y_min
        bottom = p_y_max
        im = im[:, left:right]
        im = im[top:bottom, :]

        new_target = [[0, 0], [FLAGS.image_width, 0], [FLAGS.image_width, FLAGS.image_height], [0, FLAGS.image_height]]
        new_target = np.array(new_target, dtype=np.float32).reshape([-1])

        return im, new_target
    elif mode == '2points':
        rect = target
        top = rect[0, 1]
        left = rect[0, 0]
        width = rect[1, 0] - rect[0, 0]
        height = rect[1, 1] - rect[0, 1]
        im_roi = im[top:top + height, left:left + width]
        return im_roi
    elif mode == 'box':
        rect = target
        left = rect[0]
        if left < 0:
            left = 0
        top = rect[1]
        if top < 0:
            top = 0
        width = rect[2]
        if left+width >= im.shape[1]:
            width = im.shape[1]-left-1
        height = rect[3]
        if top+height >= im.shape[0]:
            height = im.shape[0]-top-1
        #im_roi = im[top:top + height, left:left + width]
        im_roi = im[top:top + height, left:left + width]
        im_mask = np.zeros(im.shape,dtype=np.uint8)
        im_mask[top:top + height,left:left + width]=255
        return im_roi,im_mask


def resize(im, width, height):
    x_shift = 0
    y_shift = 0
    if im.shape[0] * im.shape[1] == 0:
        print('image width or height is zero...\n\n')
    if (im.shape[1] / im.shape[0] > width / height):
        border = int(im.shape[1] * height / width - im.shape[0])
        if border % 2 == 0:
            im = cv2.copyMakeBorder(im, border // 2, border // 2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            im = cv2.copyMakeBorder(im, border // 2, border // 2 + 1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y_shift = border // 2
    else:
        border = int(im.shape[0] * width / height - im.shape[1])
        if border % 2 == 0:
            im = cv2.copyMakeBorder(im, 0, 0, border // 2, border // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            im = cv2.copyMakeBorder(im, 0, 0, border // 2, border // 2 + 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x_shift = border // 2
    scale = width / im.shape[1]
    im = cv2.resize(im, (width, height))
    return (x_shift, y_shift), scale, im


def show_result(im_draw, original_points, predicted_points):
    original_points = np.array(original_points, dtype=int).reshape([-1, 2])
    predicted_points = np.array(predicted_points, dtype=int).reshape([-1, 2])
    for p in range(len(original_points)):
        cv2.circle(im_draw, (original_points[p][0], original_points[p][1]), 2, (0, 255, 0), 2)
        cv2.circle(im_draw, (predicted_points[p][0], predicted_points[p][1]), 2, (255, 0, 0), 2)
    cv2.imshow('result', im_draw)
    cv2.waitKey(0)


def show_conv(conv, title='conv', swap=None):
    conv = np.array(np.array(conv, dtype=np.float32) * 255, dtype=np.uint8)
    shape = conv.shape
    chanel = shape[-1]
    for i in range(chanel):
        im_draw = conv[0, :, :, i]
        if swap:
            im_draw = im_draw.swapaxes(0, 1)
        cv2.imshow(title + '{}'.format(i), im_draw)
        cv2.waitKey(0)


def show_conv_plot(conv, title='conv', swap=None, show=False):
    conv = np.array(np.array(conv, dtype=np.float32) * 255, dtype=np.uint8)
    shape = conv.shape
    chanel = shape[-1]
    raw = 6
    col = chanel // raw + 1
    fig = plt.figure(title)
    for i in range(chanel):
        im_draw = conv[0, :, :, i]
        if swap:
            im_draw = im_draw.swapaxes(0, 1)
        ax = fig.add_subplot(raw, col, i + 1)
        ax.imshow(im_draw, cmap="gray")
        # ax.imshow(im_draw)
        ax.set_title(i + 1)
        plt.axis("off")
    if show:
        # plt.show(block=True)
        plt.show()


def show_rect(im, rect):
    rect = np.array(rect, np.int16).reshape([-1, 2])
    cv2.rectangle(im, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), [255, 0, 0], 2)
    cv2.imshow('rect', im)
    cv2.waitKey(0)


def show_points(im, points):
    points = np.array(points, np.int16).reshape([-1, 2])
    for i in points:
        cv2.circle(im, (points[i][0], points[i][1]), [255, 0, 0], 2)
    cv2.imshow('rect', im)
    cv2.waitKey(0)


def show_image(im, title='title', wait=0):
    cv2.imshow(title, im)
    cv2.waitKey(wait)
