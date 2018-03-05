
import numpy as np
import scipy.io
import tensorflow as tf

# Importing
file = scipy.io.loadmat('TF_Data.mat')
# print(file)
samples = file["samples"]  # numpy matrix
trainSamples = file["trainSamples"]  # numpy matrix
testSamples = file["testSamples"]  # numpy matrix

n_features = samples.shape[1] - 1

###########

# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[n_features])]

# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 30],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=2,
    dropout=0.1,
    model_dir="model_backup"
)

if True:  # Training
    print("Train %d samples with %d features." %
          (trainSamples.shape[0], n_features))

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": trainSamples[:, :-1]},
        y=trainSamples[:, -1],
        num_epochs=None,
        batch_size=100,
        shuffle=True
    )

    classifier.train(input_fn=train_input_fn, steps=40000)

if True:  # Testing
    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": testSamples[:, :-1]},
        y=testSamples[:, -1],
        num_epochs=1,
        shuffle=False
    )

    # Evaluate accuracy
    evaluation = classifier.evaluate(input_fn=test_input_fn)
    accuracy_score = evaluation["accuracy"]
    print("\nTest Accuracy: {0:f}%\n".format(accuracy_score * 100))

if True:  # Predicting

    # Saving results
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": samples[:, :-1]},
        num_epochs=1,
        shuffle=False
    )

    print("Predicting class of %d samples." % samples.shape[0])
    predictions = classifier.predict(input_fn=predict_input_fn)
    predicted_classes = [p["classes"] for p in predictions]

    predicted_classes_array = np.array(
        [0 if p == b'0' else 1 for p in predicted_classes])
    predicted_classes_array = predicted_classes_array[:, None]  # vertical

    pointsPredicted = np.hstack((samples[:, 0:3], predicted_classes_array))

    scipy.io.savemat('TF_results.mat', {"pointsPredicted": pointsPredicted})
