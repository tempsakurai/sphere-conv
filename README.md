## Visual media report (Miao Cao, IST-ICE, AizawaLab)

This is an implementation of ECCV 2018 paper: SphereNet: Learning Spherical Representations for Detection and Classification in Omnidirectional Images. 

## Main idea

This paper proposes SphereNet, a distortion-aware convolutional neural network to achieve better accuracy on omnidirectional images. 

Unlike perspective images, omnidirectional images cannot be projected on a plane without introducing distortion. Most of previous methods focus on change the representation of  omnidirectional images to reduce distortion, then apply conventional CNN. 

This paper gives a novel idea: instead of change the distorted images, is changes the convolutional kernel to a sphere convolutional kernel , which wrap around the sphere by using sphere sampling pattern. 

<div  align="center">    
<img src="https://github.com/tempsakurai/sphere-conv/blob/master/images/spherenet.png" width = 300 />
</div>

The sphere sampling pattern adjusts the sampling grid locations of the convolutional filters based on the geometry of the spherical image representation. 

Sphere convolution can be concluded by 3 steps: Calculate kernel relative offsets, calculate kernel absolute positions, and sample by positions and apply convolution. 

First, we get the kernel points relative offsets on tangent plane by gnomonic projection, then use center point coordinate and inverse gnomonic projection to calculate sampling locations' coordinate. Then we use these distorted sampling pattern instead of grid pattern to apply convolution. 

Although conventional CNN has a great power in general recognition task, sometimes directly applying it to an equivariance-required problem is not a good idea. 

In this case, this paper considered about the equivariance of equirectangular images and changed basic convolution operation, which gives better performance under the specific problem.

## Implementation

The main idea of sphere convolution is implemented in `model/sphere.py`. 

The sphere convolution implementation can be divided into 3 steps:

1. Calculate kernel relative offsets
2. Calculate kernel absolute positions
3. sample by positions and apply convolution



### Calculate relative offsets

First step is to get relative offsets of each kernel point to the center point. Suppose the step size on the sphere is Δθ and Δφ, Then the sampling locations on sphere can be defined as:

<div  align="center">    
<img src="https://github.com/tempsakurai/sphere-conv/blob/master/images/sampling.png" width = 300 />
</div>


Then use gnomonic projection to calculate filter locations on the tangent planes corresponding to these sampling areas:
<div  align="center">    
<img src="https://github.com/tempsakurai/sphere-conv/blob/master/images/gnomonic-projection.png" width = 700 />
</div>

Then we get the relative positions to the center point:
<div  align="center">    
<img src="https://github.com/tempsakurai/sphere-conv/blob/master/images/kernel-pattern.png" width = 500 />
</div>

>  Note that the relative position of the center point will not be used because we know the coordinate of center point already. So in the implementation I set the value to (0.5, 0.5) to avoid  *invalid value encountered in true_divide* error.



### Calculate absolute positions

Next step is to calculate absolute sampling positions (the coordinate on the sphere) by project the tangent plane back to the sphere. 

To compute the corresponding coordinate on sphere, he inverse gnomovic projection is used:
<div  align="center">    
<img src="https://github.com/tempsakurai/sphere-conv/blob/master/images/inverse-gnomonic.png" width = 700 />
</div>
where `rho=np.sqrt(x**2+y**2)` and `v=arctan(rho)`

```python
    rho = np.sqrt(x**2+y**2)
    v = arctan(rho)
    # inverse gnomonic projection
    phi= arcsin(cos(v) * sin(center_phi) + y * sin(v) * cos(center_phi) / rho)
    theta = center_theta + arctan(x * sin(v) / (rho * cos(center_phi) * cos(v) - y * sin(center_phi) * sin(v)))
```

Then convert radian coordinates (θ, φ) to pixel coordinates (x, y).



### Sample by positions and apply convolution

For each pixel, we compute a (3, 3, 2) kernel sampling pattern by the above method. The values stored in every sampling pattern is 9 coordinates for filter sampling. The sampling patterns are distorted depending on their locations, and can cross the left and right boundary.

```python
    # img_x, img_y: coordinates on equirectangular image
    img_x = ((theta + pi) * w / pi / 2 - offset) % w  # cross equirectangular image boundary 
    img_y = (pi / 2 - phi) * h / pi - offset
```

<div  align="center">    
<img src="https://github.com/tempsakurai/sphere-conv/blob/master/images/kernel.png" width = 300 />
</div>

Then use this sampling pattern to sample and apply convolution.


### Trouble in the implementation

At first, I confused Δθ, Δφ and θ,φ coordinates. Δθ and Δφ is the relative distance at the equator and they are not related to coordinates. θ and φ is the radian coordinates of equirectangular images and they They have linear relationship with x and y respectively.



## Experiment

### Spherical Image Classification

The experiment of omnidirectional MNIST dataset classification used both sphere CNN and conventional CNN to train and test the accuracy. In OmniMNIST dataset, MNIST digits are placed on tangent planes of the image sphere and an equirectangular image of the scene is rendered at a resolution of 60 × 60 pixels.

![image](https://github.com/tempsakurai/sphere-conv/blob/master/omni-mnist.png)
The network architecture for all models consist of two blocks of convolution and max-pooling, followed by a fully-connected layer and use 32 filters in the first and 64 filters in the second layer. 
The fully connected layer has 10 output neurons and uses a softmax activation function.

SphereCNN and CNN models are trained with Adam, learning rate of 0.0001 and batches of size 128 for 100 epochs.
Here is the accuracy curves of two models:

![image](https://github.com/tempsakurai/sphere-conv/blob/master/acc.jpg)

Experiment result shows that Sphere CNN performs better than conventional CNN for omnidirectional images. 
