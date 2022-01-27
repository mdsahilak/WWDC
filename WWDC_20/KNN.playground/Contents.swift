import Foundation
import PlaygroundSupport
/*:
# Machine Learning
## In this Playground, we'll explore a simple Machine Learning Algorithm called KNearestNeighbours and test it on a classic ML dataset called the Iris Dataset.
### Exploring Machine Learning with Swift is an intresting area to pursue, due to increasing interest in swift in the machine learning space expressed by projects like Swift4Tensorflow.
- - -
*/
//: First, Lets explore the dataset.
//: - Note: The Iris Dataset The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

// The data and targets for the data are available like so.
IrisDataset.data
// Each array in IrisDataset.data represents the values for the features of the iris flower that we are trying to classify.
// These are the feature names in that order.
IrisDataset.featureNames


// Targets refer to the numeric value that represents the species of Iris that we are trying to classify.
IrisDataset.targets

// We are trying to classify the datapoints as one of these species of iris.
IrisDataset.targetNames

//: ##  Lets represent our classifier by building a simple swift struct.
/*:
 A  KNearest Neighbour Classifer  classifies an input by plotting the  n  number of feature values of an item in a dataset in n dimensional space. When the classifier is asked to classfiy an unknown item, it plots this item in the same way as well. The model then computes the nearest item from the training set that is present in this n dimensional space, hence the name Nearest Neighbbour Classifier.
*/
struct ScrappyKNN {
    
    // Data Points for Training
    var dataTrain: [[Double]] = []
    // Target Labels for Training
    var targetsTrain: [Double] = []
    
    // Accepting Input data and Fitting it to the model
    mutating func fit(trainingData: [[Double]], trainingTargets: [Double]) {
        self.dataTrain = trainingData
        self.targetsTrain = trainingTargets
    }
    
    // Method to calculate the nearest neighbour from training for a prediction
    private func nearestNeighbour(for pos: [Double]) -> Double {
        // initialise a random best distance and best index
        var bestDistance = calculateEucDist(pos, dataTrain[0])
        var bestIndex = 0
        
        // find the closest data point from the training set and returns it.
        // The distance between 2 points is calculated using euclidean distance formula
        for i in 1..<dataTrain.count{
            let distance = calculateEucDist(pos, dataTrain[i])
            if distance < bestDistance {
                bestDistance = distance
                bestIndex = i
            }
        }
        return targetsTrain[bestIndex]
    }
    
    // The prediction method.
    // Takes an input dataset and returns all predicted output labels.
    func predict(testingData: [[Double]]) -> [Double] {
        var predictions: [Double] = []
        
        // For every element in the testing data, it calculates its nearest neighbour in the training set
        for row in testingData {
            let label = nearestNeighbour(for: row)
            predictions.append(label)
        }
        
        return predictions
    }
    
}

/*: Now that we have our model, we need to train the model and make it predict our inputs.
 Before we can go ahead, we need to divide our dataset. We need to set aside a bunch from the dataset so that we can test them and verify the accuracy of the classifier. It is important that we make sure that we do not train the model on these datapoints as the model has to be tested on data that it has never seen before, which is the whole essence of testing.
- Note: I provided a helper method to do just that within the IrisDataset Model. If you would like to see how it is implemented, it is available within the playground sources in Dataset.swift with explanation.
*/
// This method gives us four variables that represent the training data and targets as well as the testing data and targets as a tuple.

let (trainingData, trainingTargets, testingData, testingTargets) = IrisDataset.trainTestSplit(numOfTestItems: 50)

//:  - Note: The iris dataset contains data for 150 flowers, so we'll train the data on 100 items and then test the classifier on the remaining 50.
/*:  ##  Lets Train and Test the classifier */
// initialise the classifier
var myClassifier = ScrappyKNN()

// fit the model with training data
myClassifier.fit(trainingData: trainingData, trainingTargets: trainingTargets)


// Let the classifier make predictions on the testing set
let predictions = myClassifier.predict(testingData: testingData)

//: Now lets compare the actual target labels with the labels the model predicted, for the 50 items that we kept aside for testing.
// Right now, all the output values are in numbers, first lets convert them to human readable strings so that its a little easier for us to read.
let correctLabels = IrisDataset.outputLabels(for: testingTargets)
let predictedLabels = IrisDataset.outputLabels(for: predictions)

/*: Lets inspect! */
consoleLineBreak(withHeading: "Correct Labels")
print(correctLabels)

consoleLineBreak(withHeading: "Predicted Labels")
print(predictedLabels)

// Rather than manually inspecting, it is useful to see how correct the model was as a percentage.
consoleLineBreak(withHeading: "Accuracy: \(accuracyScore(predictions: predictions, answers: testingTargets)*100)%")

//: ### Thats it! Thank you for checking out my Xcode Playground. - Md Sahil Ak

// End
PlaygroundPage.current.finishExecution()
