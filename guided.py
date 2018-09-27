# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 08月 19日 星期日 20:38:18 CST
# ***
# ************************************************************************************/
import sys
import image
import torch


def guided_filter(device, i, p, r=3, e=0.01):
    model = image.GuidedFilter(r, e)
    model.to(device)
    ti = image.to_tensor(i)
    tp = image.to_tensor(p)
    t = model(ti, tp)

    return image.from_tensor(t)


def self_guided_filter(device, img, r=5, e=0.01):
    model = image.GuidedFilter(r, e)
    model.to(device)

    t = image.to_tensor(img)
    t = model.self_guided(t)

    return image.from_tensor(t)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img = image.open(sys.argv[1])
    img = self_guided_filter(device, img, 10, 0.01)

    # img = guided_filter(device, s, c, 3, 0.01)

    img.show()
