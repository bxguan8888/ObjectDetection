import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import mxnet as mx
import cv2
import os, sys

OUTPUT_ROOT = "/Users/boxuanguan/Documents/python/object_detection_temp/drive_video/yunnan/labeled_images/"


def decodeBox(yolobox, w_size, h_size, x_dscale, y_scale):
    i, j, cx, cy, w, h, cls1, cls2, cls3, cls4 = yolobox
    cxt = j * x_dscale + cx * x_dscale
    cyt = i * y_scale + cy * y_scale
    wt = w * w_size
    ht = h * h_size
    cls_idx = np.argmax([cls1, cls2, cls3, cls4])
    cls_prob = [cls1, cls2, cls3, cls4][cls_idx]
    return [cxt, cyt, wt, ht, cls_idx, cls_prob]


def return_canonical_label(img, label, img_name):
    result = []

    # assert label.shape == (7,7,9)
    # Orignal Img (360, 640, 3)
    h_size = img.shape[0]
    w_size = img.shape[1]
    x_dscale = w_size / 7.0
    y_scale = h_size / 7.0

    ilist, jlist = np.where(label[:, :, 0] > 0.1)

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(np.uint8(img))

    for i, j in zip(ilist, jlist):
        cx, cy, w, h, cls1, cls2, cls3, cls4 = label[i, j, 1:]
        obj_prob = label[i, j, :1]
        cxt, cyt, wt, ht, cls, cls_prob = decodeBox([i, j, cx, cy, w, h, cls1, cls2, cls3, cls4], w_size, h_size, x_dscale,
                                          y_scale)
        # Create a Rectangle patch
        rect = patches.Rectangle((cxt - wt / 2, cyt - ht / 2), wt, ht, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        class_name = "unknown"
        class_id = -1
        if cls == 0:
            class_id = 1
            class_name = "car"
        elif cls == 1:
            class_id = 2
            class_name = "pedestrian"
        elif cls == 2:
            class_id = 3
            class_name = "cyclist"
        elif cls == 3:
            class_id = 20
            class_name = "traffic lights"
        # result.append([max(cxt - wt/2, 0), h_size - min(cyt + ht/2, h_size), min(cxt + wt / 2, w_size), h_size - max(cyt - ht/2, 0), class_id])
        plt.text(x=int(max(cxt - wt/2, 0)), y=int(max(cyt - ht / 2, 0)), s="%s:%.3f" % (str(class_name), cls_prob*obj_prob), bbox=dict(facecolor='red', alpha=0.5))
        result.append([max(cxt - wt/2, 0), max(cyt - ht / 2, 0), min(cxt + wt / 2, w_size), min(cyt + ht/2, h_size), class_id])
    plt.savefig(OUTPUT_ROOT+img_name) # save image
    # plt.show()
    plt.close('all')

    return result

def get_images_ndarray(images_name):
    images_ndarray = []
    i = 0
    for image_name in images_name:
        images_ndarray.append(plt.imread(image_name))
        i+=1
    return images_ndarray


def read_image_files(path, num_img_to_process=-1):
    dirs = os.listdir(path)
    images = []
    for file in dirs:
        if file.endswith(".jpg"):
            images.append(path+file)
    images.sort()
    print len(images)
    return images[:num_img_to_process]

def get_resized_images_ndarray(images_ndarray, shape):
    # shape example: (224, 224)
    resized_images_ndarray = []
    for image_ndarray in images_ndarray:
        resized_images_ndarray.append(cv2.resize(image_ndarray, shape))
    return resized_images_ndarray

def get_image_iter(images_ndarray):
    img_nd_array = []
    for image_ndarray in images_ndarray:
        img_nd = mx.nd.array(ctx=mx.cpu(0), source_array=image_ndarray.transpose((2, 0, 1)))
        img_nd_array.append(img_nd)
    image_iter = mx.io.NDArrayIter(data=mx.ndarray.stack(*img_nd_array), data_name='data', batch_size=32)
    return image_iter

def generate_test_result_json(images_ndarray, pred, images_name):
    num_images = pred.shape[0]
    for i in range(0, num_images):
        file_name = images_name[i].split("/")[-1]
        print "processing %d image '%s'" % (i, file_name)
        print "{%s: %s}" % (file_name, str(return_canonical_label(images_ndarray[i], pred[i], file_name)))



if __name__ == '__main__':
    # path = "/Users/boxuanguan/Documents/python/object_detection_temp/Autonomous-Yolo/DATA/testing/"
    path = "/Users/boxuanguan/Documents/python/object_detection_temp/drive_video/yunnan/images/"
    model_path = "/Users/boxuanguan/Documents/BitTiger/ObjectDetection/YOLO/models/train_val_version/drive_full_detect"
    images_name = read_image_files(path, -1)
    print len(images_name)
    print images_name[0]

    images_ndarray = get_images_ndarray(images_name)
    resized_images_ndarray = get_resized_images_ndarray(images_ndarray, (224, 224))
    image_iter = get_image_iter(resized_images_ndarray)

    sym, args_params, aux_params = mx.model.load_checkpoint(model_path, 69)
    logit = sym.get_internals()['logit_output']
    mod = mx.mod.Module(symbol=logit, context=mx.cpu(0))
    mod.bind(image_iter.provide_data)
    mod.init_params(allow_missing=False, arg_params=args_params, aux_params=aux_params,
                    initializer=mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'))
    out = mod.predict(eval_data=image_iter, num_batch=32)

    pred = (out.asnumpy() + 1) / 2

    print pred.shape
    # print return_canonical_label(resized_images_ndarray[0], pred[0])
    # print return_canonical_label(images_ndarray[0], pred[0])

    generate_test_result_json(images_ndarray, pred, images_name)
