# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, 2018年 08月 19日 星期日 20:38:18 CST
# ***
# ************************************************************************************/

from PIL import Image, ImageDraw
from torchvision import transforms

import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def open(filename):
    return Image.open(filename).convert('RGB')


def to_tensor(image):
    """
    return 1xCxHxW tensor
    """
    transform = transforms.Compose([transforms.ToTensor()])
    t = transform(image)
    return t.unsqueeze(0).to(device)


def from_tensor(tensor):
    """
    tensor format: 1xCxHxW
    """
    transform = transforms.Compose([transforms.ToPILImage()])
    return transform(tensor.squeeze(0).cpu())


# def draw_box(image, x1, y1, x2, y2):
#     draw = ImageDraw.Draw(image)
#     # fill = (0,0,255)
#     draw.rectangle((x1, y1, x2, y2))
#     del draw


# def draw_text(image, x, y, text):
#     draw = ImageDraw.Draw(image)
#     draw.text((x, y), text, fill=(255, 0, 0))
#     del draw


class GaussFilter(nn.Module):
    """
    3x3 Guassian filter
    """

    def __init__(self):
        super(GaussFilter, self).__init__()
        self.conv = nn.Conv2d(
            3, 3, kernel_size=3, padding=1, groups=3, bias=False)

        # self.conv.bias.data.fill_(0.0)
        self.conv.weight.data.fill_(0.0625)
        self.conv.weight.data[:, :, 0, 1] = 0.125
        self.conv.weight.data[:, :, 1, 0] = 0.125
        self.conv.weight.data[:, :, 1, 2] = 0.125
        self.conv.weight.data[:, :, 2, 1] = 0.125
        self.conv.weight.data[:, :, 1, 1] = 0.25

    def forward(self, x):
        x = self.conv(x)
        return x


class GuidedFilter(nn.Module):
    """
    Guided filter with r, e
    """
    def __init__(self, r, e):
        super(GuidedFilter, self).__init__()
        self.radius = r
        self.eps = e

    def box_sum(self, mat, r):
        """
            # Ai = Si+r - Si-r-1
            ===> i + r < n, i-r-1 >= 0
            ===> [0, r + 1), [r + 1, n - r), [n - r, n]
        """
        height, width = mat.size(0), mat.size(1)
        assert 2 * r + 1 <= height
        assert 2 * r + 1 <= width

        dmat = torch.zeros_like(mat)

        mat = torch.cumsum(mat, dim=0)
        dmat[0:r + 1, :] = mat[r:2 * r + 1, :]
        dmat[r + 1:height -
             r, :] = mat[2 * r + 1:height, :] - mat[0:height - 2 * r - 1, :]
        for i in range(height - r, height):
            dmat[i, :] = mat[height - 1, :] - mat[i - r - 1, :]

        dmat = torch.cumsum(dmat, dim=1)
        mat[:, 0:r + 1] = dmat[:, r:2 * r + 1]
        mat[:, r + 1:width -
            r] = dmat[:, 2 * r + 1:width] - dmat[:, 0:width - 2 * r - 1]
        for j in range(width - r, width):
            mat[:, j] = dmat[:, width - 1] - dmat[:, j - r - 1]
        return mat

    def box_filter(self, x, N):
        """
        x format is 1xCxHxW, here C = 3
        """
        y = torch.zeros_like(x)
        for i in range(x.size(1)):
            y[0][i] = self.box_sum(x[0][i], self.radius).div(N)
        return y

    def forward(self, i, p):
        N = torch.ones_like(i[0][0])
        N = self.box_sum(N, self.radius)
        mean_i = self.box_filter(i, N)
        mean_p = self.box_filter(p, N)
        mean_pi = self.box_filter(p * i, N)
        mean_ii = self.box_filter(i * i, N)

        cov_ip = mean_pi - mean_p * mean_i
        cov_ii = mean_ii - mean_i * mean_i

        a = cov_ip / (cov_ii + self.eps)
        b = mean_p - a * mean_i

        q = a * i + b
        q.clamp_(0, 1)

        return q

    def self_guided(self, p):
        N = torch.ones_like(p[0][0])
        N = self.box_sum(N, self.radius)
        mean_p = self.box_filter(p, N)
        mean_pp = self.box_filter(p * p, N)

        cov_pp = mean_pp - mean_p * mean_p

        a = cov_pp / (cov_pp + self.eps)
        b = mean_p - a * mean_p

        q = a * p + b
        q.clamp_(0, 1)

        return q


class DehazeFilter(nn.Module):
    """
    Dehaze filter with r
    """
    def __init__(self, r=7):
        super(DehazeFilter, self).__init__()
        self.radius = r
        self.maxpool = nn.MaxPool2d(2 * r + 1, stride=1, padding=r)

    def min_filter(self, x):
        """
        suppose x is : HxW, y ==> 1x1xHxW
        """
        y = x.unsqueeze(0).unsqueeze(0)
        y = y * (-1.0)
        y = self.maxpool(y)
        y = y * (-1.0)
        return y.squeeze(0).squeeze(0)

    def dark_channel(self, x):
        rgb = x[0]
        dc, _ = torch.min(rgb, dim=0)
        # dc size: HxW
        dc = self.min_filter(dc)
        return dc

    def atmos_light(self, dc, x):
        # dc -- HxW
        sorted, _ = dc.view(-1).sort(descending=True)
        index = int(dc.size(0) * dc.size(1) / 1000)
        thres = sorted[index].item()
        mask = dc.ge(thres)

        a = torch.zeros(3)
        for i in range(3):
            rgb = x[0][i]
            dx = torch.masked_select(rgb, mask)
            a[i] = torch.mean(dx)

        # RGB atmos light
        avg = 0.299 * a[0].item() + 0.587 * a[1].item() + 0.114 * a[2].item()
        a[0] = a[1] = a[2] = avg

        return a[0].item(), a[1].item(), a[2].item()

    def forward(self, x):
        """
        I = J*t + A*(1-t), here I = x, target is J
        t = 1.0 - omega*min_filter(Ic/Ac) for c = R, G, B, here w = 0.95
        J = (Ic - Ac)/t + Ac
        """
        omega = 0.95

        dc = self.dark_channel(x)

        # atmos light
        a_r, a_g, a_b = self.atmos_light(dc, x)

        # t -- from 1x3xHxW--> HxW
        t = torch.zeros_like(x)
        t[0][0] = x[0][0] / a_r
        t[0][1] = x[0][1] / a_g
        t[0][2] = x[0][2] / a_b
        t = self.dark_channel(t)
        t = 1 - omega * t

        refined_t = torch.zeros_like(x)
        refined_t[0][0] = t
        refined_t[0][1] = t
        refined_t[0][2] = t

        model = GuidedFilter(60, 0.0001)
        model.to(device)
        refined_t = model(x, refined_t)

        refined_t.clamp_(min=0.1)

        y = torch.zeros_like(x)
        y[0][0] = (x[0][0] - a_r) / refined_t[0][0] + a_r
        y[0][1] = (x[0][1] - a_g) / refined_t[0][1] + a_g
        y[0][2] = (x[0][2] - a_b) / refined_t[0][2] + a_b
        y.clamp_(0, 1)

        return y
