# coding=utf-8

# /************************************************************************************
# ***
# ***    File Author: Dell, Mon 07 Aug 2023 11:52:55 PM CST
# ***
# ************************************************************************************/
import sys
import image
import torch
from torchvision import transforms as T
from PIL import Image

if __name__ == '__main__':
    image_filename= sys.argv[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = image.SideWindowFilter(1, 10)
    model = model.eval()
    model = model.to(device)

    input_image = Image.open(image_filename).convert("RGB")
    input_tensor = T.ToTensor()(input_image)
    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_tensor = output_tensor.cpu()

    output_image = T.ToPILImage()(output_tensor.squeeze(0))
    output_image.show()
