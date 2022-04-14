import time
import os
import argparse
import glob
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument(
    '--flist', default='', type=str,
    help='The filenames of image to be processed: input, mask, output.')

parser.add_argument(
    '--mlist', default='', type=str,
    help='The filenames of masks to be processed: input, mask, output.')

parser.add_argument(
    '--image_height', default=-1, type=int,
    help='The height of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--image_width', default=-1, type=int,
    help='The width of images should be defined, otherwise batch mode is not'
    ' supported.')
parser.add_argument(
    '--checkpoint_dir', default='', type=str,
    help='The directory of tensorflow checkpoint.')


if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(0)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = InpaintCAModel()
    input_image_ph = tf.placeholder(
        tf.float32, shape=(1, args.image_height, args.image_width*2, 3))
    output = model.build_server_graph(FLAGS, input_image_ph)
    output = (output + 1.) * 127.5
    output = tf.reverse(output, [-1])
    output = tf.saturate_cast(output, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(
            args.checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')

    print(args.flist)

    for imagefile in glob.glob(args.flist):

        print(imagefile)

        # with open(args.flist, 'r') as f:
        #     lines = f.read().splitlines()
        t = time.time()
        # for line in lines:
        # for i in range(100):

        
        image_name = imagefile.split("/")[-1]
        mask_name = image_name.replace("ss_","")
        print("mask_name========= ",  mask_name)
        mask = "/content/masks/" + "mask_" + mask_name
        print("mask========= ",  mask)
        out = mask.split("/")[-1]
        out = "/content/outputs/" + out

        # print("image==", image )
        # print("mask_name==", mask_name )
        # print("mask==", mask )
        # print("out==", out )

        # image, mask, out = line.split()
        base = os.path.basename(mask)

        image = cv2.imread(imagefile)
        mask = cv2.imread(mask)
        image = cv2.resize(image, (args.image_width, args.image_height))
        mask = cv2.resize(mask, (args.image_width, args.image_height))
        # cv2.imwrite(out, image*(1-mask/255.) + mask)
        # # continue
        # image = np.zeros((128, 256, 3))
        # mask = np.zeros((128, 256, 3))

        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 4
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = sess.run(output, feed_dict={input_image_ph: input_image})
        print('Processed: {}'.format(out))
        cv2.imwrite(out, result[0][:, :, ::-1])

    # print('Time total: {}'.format(time.time() - t))
