# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 08月 19日 星期日 20:38:18 CST
# ***
# ************************************************************************************/
import sys
import image
import torch


def dehaze_filter(device, img, r=3):
    model = image.DehazeFilter(r)
    model.to(device)
    t = image.to_tensor(img)
    t = model(t)
    return image.from_tensor(t)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = image.open(sys.argv[1])
    img = dehaze_filter(device, img, 5)
    img.show()
