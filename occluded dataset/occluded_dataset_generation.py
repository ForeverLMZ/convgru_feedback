import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

# Set up the transforms for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
mnist_train = datasets.MNIST(root='/home/mila/m/mashbayar.tugsbayar/datasets', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='/home/mila/m/mashbayar.tugsbayar/datasets', test=True, download=True, transform=transform)

# Set the occlusion size and generate the occluded MNIST dataset
occlusion_size = 15
occluded_mnist_train = []
occluded_mnist_test = []

#training set
for i in range(len(mnist_train)):
    img, target = mnist_train[i]
    img = np.array(img)
    h, w = img.shape[1:]
    occluded_img = img.copy()
    x = np.random.randint(0, w-occlusion_size)
    y = np.random.randint(0, h-occlusion_size)
    occluded_img[:, y:y+occlusion_size, x:x+occlusion_size] = 0
    occluded_mnist_train.append((torch.Tensor(occluded_img), target))
    
#testing set
for i in range(len(mnist_test)):
    img, target = mnist_test[i]
    img = np.array(img)
    h, w = img.shape[1:]
    occluded_img = img.copy()
    x = np.random.randint(0, w-occlusion_size)
    y = np.random.randint(0, h-occlusion_size)
    occluded_img[:, y:y+occlusion_size, x:x+occlusion_size] = 0
    occluded_mnist_test.append((torch.Tensor(occluded_img), target))

# Save the occluded MNIST dataset
torch.save(occluded_mnist_train, '/home/mila/m/mingze.li/occluded dataset/occluded_mnist_train.pt')
torch.save(occluded_mnist_test, '/home/mila/m/mingze.li/occluded dataset/occluded_mnist_test.pt')
