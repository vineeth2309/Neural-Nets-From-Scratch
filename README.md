# Neural-Nets-From-Scratch



### **Description:**

* Repository to implement Neural Network architectures from scratch using numpy.
* The repository contains classes for (Forward and Backward pass):
  1. **<u>Activation Functions</u>**: 
     * Sigmoid
     * Tanh
     * Relu
     * Leaky Relu
     * Softmax
  2. **<u>Linear Layer:</u>**
     * Weight and bias initialization
     * Activation function
     * Learning Rate
     * <u>To Add</u>:
       1. Different methods of weight initialization
       2. Different Optimization methods
  3. <u>**Loss Functions**</u>:
     * Mean Square Error
     * Root Mean Square Error
     * Cross Entropy
  4. <u>**Tensor**</u>:
     * Class to store data, gradients, shape, and whether the tensor requires gradient compute. 



### **Running the Repo:**

```bash
$ python3 Neural_Net.py
```

* The **Main** class creates a dummy dataset using the sklearn: **make_blobs** function, and converts them to tensor form. 

* The **Main** class contains functions for converting between class integer representation, and **one hot encoding**.

* The **Neural Network** class can be modified to change the model architecture using the structure followed in the code.

* <u>To Add:</u>

  1. Batch Training

  2. Computational Graph 

     

### **Project Requirements**:

1) Python3 / Python2

2) numpy

3) sklearn



