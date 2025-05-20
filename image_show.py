import matplotlib.pyplot as plt
import numpy as np
import torchvision


class ImageShow:

    def __init__(self, trainloader, classes, batch_size: int=4):
        self.trainloader = trainloader
        self.classes = classes
        self.batch_size = batch_size

    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def show_batch(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = next(dataiter)

        # show images
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join(f'{self.classes[labels[j]]:5s}' for j in range(self.batch_size))) # tuple comprehension