# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 08月 19日 星期日 20:38:18 CST
# ***
# ************************************************************************************/
import sys
import image
import torch
import argparse
import glob
import os


def dehaze_filter(r, device):
    model = image.DehazeFilter(r)
    model.to(device)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=str, default="hazeimgs/*.jpg", help="input image")
    parser.add_argument('--output', type=str, default="output", help="output directory")
    args = parser.parse_args()

    # Create directory to store results
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = dehaze_filter(3, device)

    image_filenames = glob.glob(args.input)

    for index, filename in enumerate(image_filenames):
    	print("Dehazing {} ... ".format(filename))

    	img = image.open(filename)

    	input_tensor = image.to_tensor(img)
    	output_tensor = model(input_tensor)

    	oimg = image.grid_image([input_tensor, output_tensor], nrow = 2)
    	oimg.save(args.output + "/dehaze_" + os.path.basename(filename))
