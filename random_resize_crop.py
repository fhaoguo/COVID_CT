from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

img = Image.open('./dianwei.jpg')
# RandomResizedCrop 将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为指定大小
print('原图大小:', img.size)
data1 = transforms.RandomResizedCrop(224)(img)
print('随机裁剪后的大小：', data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title('Original')
plt.subplot(2,2,2),plt.imshow(data1),plt.title('Transform 1')
plt.subplot(2,2,3),plt.imshow(data2),plt.title('Transform 2')
plt.subplot(2,2,4),plt.imshow(data3),plt.title('Transform 3')
plt.show()

# 以输入图的中心点为中心点做指定size的crop操作
img1 = transforms.CenterCrop(224)(img)
img2 = transforms.CenterCrop(224)(img)
img3 = transforms.CenterCrop(224)(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title('Original')
plt.subplot(2,2,2),plt.imshow(img1),plt.title('Transform 1')
plt.subplot(2,2,3),plt.imshow(img2),plt.title('Transform 2')
plt.subplot(2,2,4),plt.imshow(img3),plt.title('Transform 3')
plt.show()

# 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
img1 = transforms.RandomHorizontalFlip()(img)
img2 = transforms.RandomHorizontalFlip()(img)
img3 = transforms.RandomHorizontalFlip()(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title('Original')
plt.subplot(2,2,2),plt.imshow(img1),plt.title('Transform 1')
plt.subplot(2,2,3),plt.imshow(img2),plt.title('Transform 2')
plt.subplot(2,2,4),plt.imshow(img3),plt.title('Transform 3')
plt.show()