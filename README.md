# Classification of bottle photos
## Description of the problem and data
### A general description of the field covered by the data
The aim of the project is to create a computer system that will be able to automatically recognize and classify images of bottles based on their visual features. Deep learning techniques are used here, specifically neural networks, which are able to effectively extract and analyze patterns in images.  
To achieve this goal, the project requires an appropriate training dataset that contains images of bottles of different types, in different contexts and with different variants. This dataset is used to train a neural network that will be able to recognize bottles based on features such as shape, color, label and texture.  
During training, the neural network is presented with the training data set and its weights are modified iteratively to minimize classification error. This process involves backpropagation, where the error is reversed through the network and the weights are adjusted depending on this error. By repeatedly presenting training data and modifying the weights, the neural network gradually improves its ability to recognize bottles.
After completing the training and achieving satisfactory results, the neural network model is evaluated on a set of test data that was not used in the training process. This allows you to evaluate the overall bottle classification performance of the model.  
In summary, the bottle image classification project using neural networks aims to create a system capable of automatically recognizing and classifying bottles based on their visual features.  
### Number of records
For the project we used 25,000 photos of bottles, divided into five categories depending on what is in the photo. The labels are: beer, plastic bottle (water), soda, water (water bottle), wine.
### Basic statistics
The photos were sharp, characterized by great clarity and detail. All photos had a resolution of 512 by 512.
### Clear definition of what problem is being solved
The goal of the project is to teach a computer to recognize patterns and understand the content of images in order to automatically identify them. Therefore, the problem was image classification, which is the process of assigning categories or labels to images based on their visual characteristics.
## Data processing
### Number of missing records and how to solve this problem, normalization/standardization of input data
All data was properly prepared, nothing was missing or needed to be modified. Therefore, no steps have been taken here.
### Transformations performed on the data
The only data transformation was downscaling the images, done:
```python
img_gen = ImageDataGenerator(rescale=1./255)
```
### Method of dividing into a training set and a testing set
At the beginning, about 30 photos were set aside - they will not be used in the teaching process and will be used for later practice. From the remaining photos from each label, 100 photos of bottles belonging to a certain label were selected and called the testing set. The remaining photos are a training set.

## Description of the neural networks used
### Network type and architecture
Since we were dealing with the problem of image classification, we used convolutional neural networks (CNN), which are specially designed neural networks used to process spatial data, such as images. Their architecture is based on the use of convolutional layers, pooling layers and fully connected layers. All parts are included in the script:
```python
model = Sequential()
# Convolution 1
model.add(Conv2D(32, kernel_size=(4,4), activation = "relu", input_shape = (128, 128, 3)))
model.add(AvgPool2D(pool_size = (4,4)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Convolution 2
model.add(Conv2D(64, kernel_size=(3,3), activation = "relu"))
model.add(AvgPool2D(pool_size = (3,3)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Convolution 3
model.add(Conv2D(128, kernel_size=(3,3), activation = "relu"))
model.add(AvgPool2D(pool_size = (3,3)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Convolution 4
model.add(Conv2D(256, kernel_size=(2,2), activation = "relu"))
model.add(AvgPool2D(pool_size = (1,1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Convolution 5
model.add(Conv2D(16, kernel_size=(1,1), activation = "relu"))
model.add(AvgPool2D(pool_size = (1,1)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Flatten & Linear Fully Connected Layers
model.add(Flatten())
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(5, activation = "softmax"))
```
### Description of layers
The basic component is convolutional layers. These layers perform a convolution operation, which involves moving the filter (convolution kernel) over the image and then computing the dot product between the filter elements and the corresponding pixels in the image. This process allows for the extraction of local features such as edges, textures and patterns by recognizing local patterns present in the image.  
Pooling layers are another important element in the CNN architecture. The pooling layer works on the output of the convolutional layer and reduces the spatial dimensionality of the data by, for example, using the pool size operation, which selects the maximum value from a specific area. Thanks to this, the concatenation layer reduces the number of network parameters and also introduces invariance to small translations in the input data.  
After the convolution and pooling layers, we applied fully connected layers. These layers aim to perform classification or regression based on the features extracted by the convolutional layers. Each neuron in a fully connected layer is connected to all neurons in the previous layer, which provides the ability to model more complex feature relationships.   

### ReLU activation function
We used ReLU as the activation function, which is one of the most frequently used activation functions in neural networks, including convolutional neural networks (CNN). The ReLU function is simple and effective.   
Mathematically, the ReLU function is defined as: f(x) = max(0, x), where x is the input value and f(x) is the output value after applying the ReLU function.   
The ReLU function is non-linear and involves passing non-negative values without any change, while for negative values the function returns zero. In other words, if the input value is greater than or equal to zero, ReLU returns that value, and if it is less than zero, then it returns zero.   
In addition to using ReLU, we also used the softmax activation function - particularly useful in classification applications, where the model is intended to assign input data to one of many classes.   
### Softmax activation function
The softmax function works by transforming the results into probability values. Each element of the input vector z is transformed to a positive value and then divided by the sum of the transformed values for all elements of the vector. As a result, the sum of all transformed values is equal to 1, which allows these values to be interpreted as probabilities of belonging to particular classes. The action can be illustrated by the formula:   
$softmax(z_i ) = \frac{e^{z_i }}{∑z_j}\  $  
Where zi means the i-th element of the input vector z, and the sum in the denominator means summing over all elements of the vector.   
### How to train the network – Adam
In the project, we used the Adam (Adaptive Moment Estimation) learning algorithm, which combines the features of the AdaGrad (Adaptive Gradient) and RMSProp (Root Mean Square Propagation) algorithms. The first is an optimization algorithm that adjusts the learning rate for each weight based on previous gradients. AdaGrad uses a faster learning rate for rare features (weights) and a slower learning rate for frequent features, which can improve convergence in case of uneven data distribution. The second one is a different optimization algorithm that adjusts the learning rate for each weight by taking into account the average values of gradients over time. RMSProp helps to better adapt to the curvature of the error surface and prevents oscillation around the local minimum.
### Loss function
Since the project concerned multi-class classification, we used the Categorical Crossentropy function, which is popular in this application. It was used in conjunction with the softmax activation function on the last layer of the neural network.  
Categorical Crossentropy compares the true labels (classes) of the input data with the model predictions. The loss is calculated, which is a measure of the discrepancy between these two probability distributions.
## Discussion of the results and conclusions
### Network results summarized in a table, divided into training and testing sets
|    epoch     |     precision    |      precision           |     accuracy    |         accuracy        |
|--------------|------------------|-----------------|-----------------|-----------------|
|              |     train        |     test        |     train       |     test        |
|     1        |     0,541745     |     0,719708    |     0,379341    |     0,425397    |
|     2        |     0,77115      |     0,746032    |     0,568032    |     0,573413    |
|     3        |     0,802091     |     0,80436     |     0,679475    |     0,735714    |
|     4        |     0,842154     |     0,913643    |     0,749822    |     0,802778    |
|     5        |     0,870345     |     0,904072    |     0,803562    |     0,84127     |
|     6        |     0,888346     |     0,91887     |     0,82854     |     0,819048    |
|     7        |     0,89516      |     0,936544    |     0,857435    |     0,876984    |
|     8        |     0,909301     |     0,861254    |     0,869145    |     0,890079    |
|     9        |     0,910329     |     0,868601    |     0,886999    |     0,905556    |
|     10       |     0,924648     |     0,824561    |     0,903206    |     0,918651    |
|     11       |     0,926005     |     0,931348    |     0,909395    |     0,901984    |
|     12       |     0,93472      |     0,953508    |     0,915539    |     0,885714    |
|     13       |     0,935428     |     0,947804    |     0,923019    |     0,93254     |
|     14       |     0,94025      |     0,948073    |     0,921995    |     0,925       |
|     15       |     0,942426     |     0,948169    |     0,928807    |     0,936905    |
|     16       |     0,941808     |     0,954868    |     0,931389    |     0,943651    |
|     17       |     0,947039     |     0,949362    |     0,936687    |     0,928571    |
|     18       |     0,950847     |     0,957527    |     0,937578    |     0,939286    |
|     19       |     0,948353     |     0,9509      |     0,943321    |     0,925397    |
|     20       |     0,954477     |     0,952188    |     0,945191    |     0,916667    |

Based on the data in the table, you can create line charts that show changes in the values of two metrics: precision and accuracy.  
![](/image/graph1.jpg )  
![](/image/graph2.jpg )  
### Why were precision and accuracy values selected for analysis?
Precision is one of the basic metrics used in machine learning to assess the quality of binary and multiclass classifiers. It measures the proportion of correctly classified positive instances among all instances that were classified as positive. The precision metric focuses on the quality of the predictions for the positive class. The higher the precision value, the fewer false positive classifications occur among the instances classified as positive. In other words, it measures how often the classifier correctly identifies actual positive cases.  
In turn, classification is performed to obtain information about what is in the graphic. Therefore, to see how satisfactory the result is, you can use accuracy, which is one of the most frequently used metrics in machine learning to evaluate the effectiveness of a classification model. It measures the percentage of correctly classified cases out of all cases. Accuracy is an intuitive metric for evaluating a model because it measures how well the model does at classifying the data. The higher the accuracy value, the better the model performs.  
However, keep in mind that accuracy may not be a dangerous metric in some situations, especially when dealing with unbalanced class problems. If the data is unbalanced and one class dominates the others, the model can achieve high accuracy by predicting only the dominant class. In such cases, other metrics such as precision, sensitivity (recall) or F1-score may be more appropriate to assess the performance of the model. It is also worth noting that accuracy does not take into account class-specific misprediction information. It may happen that the model achieves high accuracy but makes significant errors for specific classes. Therefore, it is worth analyzing other metrics to get a more complete picture of the classification model's performance.  
### Determine whether the results are satisfactory with justification
The results obtained and presented above are quite promising, so we were tempted to try it on our own collection of photos. We tested the script on a pseudo-randomly selected ten out of 29 photos. Each photo contains one of three beer bottles: a green one without a label, a brown bottle with a traditional German shape and a beer bottle with a Krusovice brewery label. The results are as follows:  
![](/image/demo1.jpg )  
![](/image/demo2.jpg )  
According to the script, the photos include:
* beer 4 times
* wine 2 times
* carbonated drink (soda) 2 times
* water 2 times  

From this run we can calculate an accuracy of 40%, which seems quite low. However, it is even satisfying. No other drink has been falsely detected more times than beer, or even three times. Therefore, if you present the script with several consecutive images of the same thing, it is possible that it will correctly recognize the object in the image. What's more, it's nice that the program works so well on our photos, which were blurry and underexposed, unlike those on which it learned.  
We subjected the data to additional analysis - we used all photos from the zip file downloaded from the Kaggle website. Based on them, we created a confusion matrix.  
![](/image/matrix.jpg )  
Confusion matrix shows most of the values on the diagonal, which is understandable - the predicted results are very often consistent with the actual ones, which is confirmed by high accuracy.
### Conclusions, further proposals for the development of the project.
The script works satisfactorily well - it achieves high accuracy when run on bright, clear images, but also performs quite well on blurry photos taken with a laptop camera. It allows you to recognize what is in the image and which bottle when there are a large number of similar photos.  
Development of the program could be considered. An interesting idea would be to use a camera on a laptop. The user would stand in front of it with a bottle and the script would recognize what kind of bottle it was. As the photos above showed, it is possible for the script to handle such images.
