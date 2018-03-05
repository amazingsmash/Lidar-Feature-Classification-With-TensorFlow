# LiDAR Feature Classification With TensorFlow


Small TF example used to classify airborne LiDAR using TensorFlow Deep Neural Network models.

Pre-processing and deature extraction from the pointcloud data in folder "DATASET" is performed in Matlab code via voxeling algorithms.

## Normal workflow:

1. Execute "generateDataForTensorFlow.m" on Matlab to generate "TF_Data.mat" containing sample points for training, testing and predicting on the TensorFlow model.

2. Execute "sampleDNNClassifier.py" with Python3 environment (numpy and tensorflow needed) enabling the desired stages.

3. Run "plotTensorFlowClassification.m" on Matlab should show the results of the binary classification as predicted by the DNN model.
