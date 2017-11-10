## Project: Follow Me

---

[//]: # (Image References)

[intro]: ./docs/misc/sim_screenshot.png

[vgg_fcn]:   ./docs/misc/fcn.png
[vgg_fcn8s]: ./docs/misc/VGG_structure.png

[model]: ./docs/misc/archi.png

[train1]: ./docs/misc/sim_pattern_1.png
[train2]: ./docs/misc/sim_pattern_2.png
[train_curve]:   ./docs/misc/training_curve.png
[train_curve_f]: ./docs/misc/training_curve_final.png

[model_hero_near]: ./docs/misc/hero_near.png
[model_hero_far]:  ./docs/misc/hero_far.png
[model_people]:    ./docs/misc/people.png

### Introduction
A deep, fully convolutional neural network was trained for semantic segmentation. This net allowed a drone to recognise, track and follow a specific person (amongst others) in a relatively busy (Unity 3D) environment.

![params][intro]

### Network Architecture
#### Overall Structure
The network architecture was inspired by the so-called "FCN-8s" network [1], which itself was adapted from the so-called "VGG16" network [2]. This architecture was essentially simplified (with minor modifications) until it could run on my humble PC.

##### VGG16
The VGG network architectures were designed for image classification. They were primarily created to evaluate the effect of network depth on classification accuracy, and basically showed that deeper is better. These deep networks (16-19 layers) were made feasible by using very small convolution filters (3x3).

The advantage of using e.g. two stacked 3x3 conv layers instead of one 5x5 layer (which has the same receptive field, given no spacial pooling) is that the former has two non-linear rectification layers while the latter has only one. More rectification layers mean a more discriminative decision function. Additionally, the former has less parameters, assuming the same number of channels. This effectively regularises the 5x5 filters by forcing a decomposition through the 3x3 filters. Similar reasoning applies to three stacked 3x3 layers vs one 7x7 layer and so forth.

One of the VGG nets (and my final net) makes use of 1x1 conv layers. These layers effectively increase the non-linearity of the decision function without altering receptive fields. They can also be used to reduce dimensionality of previous layers.

VGG stands for Visual Geometry Group, a research lab at Oxford.

##### FCN-8s
The authors of FCN-8s tackled semantic segmentation by transforming fully connected layers at the outputs of popular classification architectures (like the VGG nets) into convolutional layers. This results in so-called fully convolutional networks, which output heat-maps instead of classifications. The transformation is achieved by replacing the fully connected layers with a 1x1 conv layer, which has the added advantage of allowing input images of any size.

![params][vgg_fcn]

To connect the coarse heat maps to individual pixels for refined prediction, the heat-maps were upsampled using so-called deconvolution layers (so-called because they aren't actually performing deconvolution). Deconvolution layers could be simple interpolations (e.g. bilinear interpolation), transposed convolutions (AKA àtrous convolutions) or they could be learned. In FCN-8s and its siblings, the layers were initialised as bilinear operations but were allowed to learn.

To further refine the predictions, the final layer was combined with lower layers using skip connections. This allows the net to make local predictions that respect global structure.

![params][vgg_fcn8s]

#### Final Network
The encoder section of the final net mimics the VGG shape, aiming for depth by using three separate stacks of two 3x3 conv layers, one 1x1 conv layer and one 2x2 max pooling layer. All conv layer stride lengths are one, so that downsampling exclusively happens at the max pool layers. A final 1x1 conv layer is added with 3 channels (target person, other people and background) to create the coarse heat-map.

The decoder section uses bilinear interpolation layers for upsampling and refining the heat map. There is one interpolation layer for each encoding max pool layer, so that the refined heat-map is the same size as the original image. Skip connections from each of the stacked 1x1 conv layers are also added.

Finally, a fully connected soft-max layer right at the end turns the refined heat-map into pixel-wise classifications.

![params][model]

### Data Acquisition
#### Simulator
Character paths were set so as to capture the target from a range of angles, with and without crowds present.

Example patrol patterns:

![params][train1]

![params][train2]

#### Data Augmentation
The simulator frequently crashed, making data gathering tedious. To augment the data set, all images were flipped left to right, doubling the number of total images.

### Hyperparameter Tuning
A range of learning rates were tested: 1e-2, 1e-3, 1e-4 and 5e-5. The lowest rate ended up with the best accuracy (eventually).

A batch size of 50 was used. My graphics card couldn't handle more than that, and significantly less made the gradient updates too erratic.

To save time the first 10 epochs were trained at 1e-3. Thereafter another 50 epochs were trained at 5e-5 to refine the model as much as possible (50 epochs is where the validation loss seemed to flatten out).

Training curve over 100 epochs:

![params][train_curve]

Training curve for last 50 epochs:

![params][train_curve_f]

Steps per epoch (for training and validation) were set at roughly the number of images divided by the batch size.

### Results
The model does well for non-target people (middle is ground truth, right is model output):

![params][model_people]

And does well when the target is close to the camera:

![params][model_hero_near]

But struggles a bit when the target's further away:

![params][model_hero_far]

Final weighted Intersection over Union is 0.43518.

### Future Enhancements
Ultimately the model was limited by my hardware. I investigated AWS but had some issues proving to Amazon that I'm not a robot.

There are a few semantic segmentation architectures out there [3]. One of interest is DeepLab [4], which claims the current state of the art. DeepLab emphasises àtrous convolutions, and uses conditional random fields to improve localisation accuracy.

Additionally, more data is always worth gathering (particularly of the target at a distance) and there are many ways to further augment the existing data set [5].

This architecture could potentially be used to identify other objects (e.g. animals, cars) if provided with enough training data of those objects. The original VGG was trained as a general object classification net after all. More objects would probably require more parameters (and hence beefier hardware) though.

### References
[1] [J. Long, E. Shelhamer and T. Darrel. Fully convolutional networks for semantic segmentation. CV, abs/1411.4038, 2015.](https://arxiv.org/abs/1411.4038)

[2] [K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014.](https://arxiv.org/abs/1409.1556)

[3] [A. Garcia-Garcia, S. Orts-Escolano, S. Oprea, V. Villena-Martinez and J. Garcia-Rodriguez. A Review on Deep Learning Techniques Applied to Semantic Segmentation. CV, abs/1704.06857, 2017.](https://arxiv.org/abs/1704.06857)

[4] [L. Chen, G. Papandreou, I. Kokkinos, K. Murphy and A. L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. CV, abs/1606.00915, 2017.](https://arxiv.org/abs/1606.00915)

[5] [J. Wang and L. Perez. The Effectiveness of Data Augmentation in Image Classification using Deep Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
