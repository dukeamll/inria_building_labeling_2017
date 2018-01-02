import os
import numpy as np
import matplotlib.pyplot as plt
from dataReader import patch_extractor


def make_output_file(label, colormap):
    try:
        colormap = {0: 0, 1: 255, 2: 122}
        encode_func = np.vectorize(lambda x, y: y[x])
    except:
        colormap = {0:0, 1: 255, 2: 122}
        encode_func = np.vectorize(lambda x, y: y[x])
    return encode_func(label, colormap)


def decode_labels(label, num_images=10):
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    label_colors = {0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255)}
    for i in range(n):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
        outputs[i] = pixels
    return outputs


def decode_labels_binary(label, colormap, num_images=None):
    label_binary = label[:, :, :, 0]
    n, h, w = label_binary.shape
    if num_images is not None:
        n = num_images
    outputs = np.zeros((n, h, w), dtype=np.uint8)
    encode_func = np.vectorize(lambda x, y: y[x])

    for i in range(n):
        outputs[i, :, :] = encode_func(label_binary[i, :, :], colormap)

    return outputs


def get_pred_labels(pred):
    if len(pred.shape) == 4:
        n, h, w, c = pred.shape
        outputs = np.zeros((n, h, w, 1), dtype=np.uint8)
        for i in range(n):
            outputs[i] = np.expand_dims(np.argmax(pred[i,:,:,:], axis=2), axis=2)
        return outputs
    elif len(pred.shape) == 3:
        outputs = np.argmax(pred, axis=2)
        return outputs


def pad_prediction(image, prediction):
    _, img_w, img_h, _ = image.shape
    n, pred_img_w, pred_img_h, c = prediction.shape

    if img_w > pred_img_w and img_h > pred_img_h:
        pad_w = int((img_w - pred_img_w) / 2)
        pad_h = int((img_h - pred_img_h) / 2)
        prediction_padded = np.zeros((n, img_w, img_h, c))
        pad_dim = ((pad_w, pad_w),
                   (pad_h, pad_h))

        for batch_id in range(n):
            for channel_id in range(c):
                prediction_padded[batch_id, :, :, channel_id] = \
                    np.pad(prediction[batch_id, :, :, channel_id], pad_dim, 'constant')
        prediction = prediction_padded
        return prediction
    else:
        return prediction


def image_summary(image, truth, prediction, img_mean=np.array((109.629784946, 114.94964751, 102.778073453), dtype=np.float32)):
    truth_img = decode_labels(truth, 10)

    prediction = pad_prediction(image, prediction)

    pred_labels = get_pred_labels(prediction)
    pred_img = decode_labels(pred_labels, 10)

    return np.concatenate([image+img_mean, truth_img, pred_img], axis=2)


def get_output_label(result, image_dim, input_size, colormap, overlap=0,
                     output_image_dim=None, output_patch_size=None, make_map=True, soft_pred=False):
    if output_image_dim is not None and output_patch_size is not None:
        image_pred = patch_extractor.un_patchify_shrink(result, image_dim, output_image_dim,
                                                        input_size, output_patch_size, overlap=overlap)
    else:
        image_pred = patch_extractor.un_patchify(result, image_dim, input_size, overlap=overlap)
    if soft_pred:
        labels_pred = image_pred
    else:
        labels_pred = get_pred_labels(image_pred)
    if make_map:
        output_pred = make_output_file(labels_pred, colormap)
        return output_pred
    else:
        return labels_pred


def iou_metric(truth, pred, truth_val=255):
    truth = truth / truth_val
    pred = pred / truth_val
    truth = truth.flatten()
    pred = pred.flatten()
    intersect = truth*pred
    return sum(intersect == 1) / \
           (sum(truth == 1)+sum(pred == 1)-sum(intersect == 1))


def set_full_screen_img():
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()


def make_task_img_folder(parent_dir):
    task_fold_name = os.path.basename(os.getcwd())
    if not os.path.exists(os.path.join(parent_dir, task_fold_name)):
        os.makedirs(os.path.join(parent_dir, task_fold_name))
    return os.path.join(parent_dir, task_fold_name)


def get_task_img_folder(local_dir=False):
    if local_dir:
        IMG_DIR = r'/home/lab/Documents/bohao/research/figs'
        TASK_DIR = r'/home/lab/Documents/bohao/research/tasks'
    else:
        IMG_DIR = r'/media/ei-edl01/user/bh163/figs'
        TASK_DIR = r'/media/ei-edl01/user/bh163/tasks'

    return make_task_img_folder(IMG_DIR), make_task_img_folder(TASK_DIR)


def barplot_autolabel(ax, rects, margin=None):
    if margin is None:
        margin = (ax.get_ylim()[1] - ax.get_ylim()[0])/20/ax.get_ylim()[1]
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2, (1+margin)*height,
                '{:.2f}'.format(height), ha='center', va='bottom')
