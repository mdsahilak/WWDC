import Foundation

public struct IrisDataset {
    
    // metadata
    public static let featureNames: [String] = ["sepal_length (cm)", "sepal_width (cm)", "petal_length (cm)", "petal_width (cm)"]
    public static let targetNames: [String] = ["setosa", "versicolor", "virginica"]
    
    // data values
    public static let data: [[Double]] = [ [5.1, 3.5, 1.4, 0.2 ], [4.9, 3.0, 1.4, 0.2 ], [4.7, 3.2, 1.3, 0.2 ], [4.6, 3.1, 1.5, 0.2 ], [5.0, 3.6, 1.4, 0.2 ], [5.4, 3.9, 1.7, 0.4 ], [4.6, 3.4, 1.4, 0.3 ], [5.0, 3.4, 1.5, 0.2 ], [4.4, 2.9, 1.4, 0.2 ], [4.9, 3.1, 1.5, 0.1 ], [5.4, 3.7, 1.5, 0.2 ], [4.8, 3.4, 1.6, 0.2 ], [4.8, 3.0, 1.4, 0.1 ], [4.3, 3.0, 1.1, 0.1 ], [5.8, 4.0, 1.2, 0.2 ], [5.7, 4.4, 1.5, 0.4 ], [5.4, 3.9, 1.3, 0.4 ], [5.1, 3.5, 1.4, 0.3 ], [5.7, 3.8, 1.7, 0.3 ], [5.1, 3.8, 1.5, 0.3 ], [5.4, 3.4, 1.7, 0.2 ], [5.1, 3.7, 1.5, 0.4 ], [4.6, 3.6, 1.0, 0.2 ], [5.1, 3.3, 1.7, 0.5 ], [4.8, 3.4, 1.9, 0.2 ], [5.0, 3.0, 1.6, 0.2 ], [5.0, 3.4, 1.6, 0.4 ], [5.2, 3.5, 1.5, 0.2 ], [5.2, 3.4, 1.4, 0.2 ], [4.7, 3.2, 1.6, 0.2 ], [4.8, 3.1, 1.6, 0.2 ], [5.4, 3.4, 1.5, 0.4 ], [5.2, 4.1, 1.5, 0.1 ], [5.5, 4.2, 1.4, 0.2 ], [4.9, 3.1, 1.5, 0.2 ], [5.0, 3.2, 1.2, 0.2 ], [5.5, 3.5, 1.3, 0.2 ], [4.9, 3.6, 1.4, 0.1 ], [4.4, 3.0, 1.3, 0.2 ], [5.1, 3.4, 1.5, 0.2 ], [5.0, 3.5, 1.3, 0.3 ], [4.5, 2.3, 1.3, 0.3 ], [4.4, 3.2, 1.3, 0.2 ], [5.0, 3.5, 1.6, 0.6 ], [5.1, 3.8, 1.9, 0.4 ], [4.8, 3.0, 1.4, 0.3 ], [5.1, 3.8, 1.6, 0.2 ], [4.6, 3.2, 1.4, 0.2 ], [5.3, 3.7, 1.5, 0.2 ], [5.0, 3.3, 1.4, 0.2 ], [7.0, 3.2, 4.7, 1.4 ], [6.4, 3.2, 4.5, 1.5 ], [6.9, 3.1, 4.9, 1.5 ], [5.5, 2.3, 4.0, 1.3 ], [6.5, 2.8, 4.6, 1.5 ], [5.7, 2.8, 4.5, 1.3 ], [6.3, 3.3, 4.7, 1.6 ], [4.9, 2.4, 3.3, 1.0 ], [6.6, 2.9, 4.6, 1.3 ], [5.2, 2.7, 3.9, 1.4 ], [5.0, 2.0, 3.5, 1.0 ], [5.9, 3.0, 4.2, 1.5 ], [6.0, 2.2, 4.0, 1.0 ], [6.1, 2.9, 4.7, 1.4 ], [5.6, 2.9, 3.6, 1.3 ], [6.7, 3.1, 4.4, 1.4 ], [5.6, 3.0, 4.5, 1.5 ], [5.8, 2.7, 4.1, 1.0 ], [6.2, 2.2, 4.5, 1.5 ], [5.6, 2.5, 3.9, 1.1 ], [5.9, 3.2, 4.8, 1.8 ], [6.1, 2.8, 4.0, 1.3 ], [6.3, 2.5, 4.9, 1.5 ], [6.1, 2.8, 4.7, 1.2 ], [6.4, 2.9, 4.3, 1.3 ], [6.6, 3.0, 4.4, 1.4 ], [6.8, 2.8, 4.8, 1.4 ], [6.7, 3.0, 5.0, 1.7 ], [6.0, 2.9, 4.5, 1.5 ], [5.7, 2.6, 3.5, 1.0 ], [5.5, 2.4, 3.8, 1.1 ], [5.5, 2.4, 3.7, 1.0 ], [5.8, 2.7, 3.9, 1.2 ], [6.0, 2.7, 5.1, 1.6 ], [5.4, 3.0, 4.5, 1.5 ], [6.0, 3.4, 4.5, 1.6 ], [6.7, 3.1, 4.7, 1.5 ], [6.3, 2.3, 4.4, 1.3 ], [5.6, 3.0, 4.1, 1.3 ], [5.5, 2.5, 4.0, 1.3 ], [5.5, 2.6, 4.4, 1.2 ], [6.1, 3.0, 4.6, 1.4 ], [5.8, 2.6, 4.0, 1.2 ], [5.0, 2.3, 3.3, 1.0 ], [5.6, 2.7, 4.2, 1.3 ], [5.7, 3.0, 4.2, 1.2 ], [5.7, 2.9, 4.2, 1.3 ], [6.2, 2.9, 4.3, 1.3 ], [5.1, 2.5, 3.0, 1.1 ], [5.7, 2.8, 4.1, 1.3 ], [6.3, 3.3, 6.0, 2.5 ], [5.8, 2.7, 5.1, 1.9 ], [7.1, 3.0, 5.9, 2.1 ], [6.3, 2.9, 5.6, 1.8 ], [6.5, 3.0, 5.8, 2.2 ], [7.6, 3.0, 6.6, 2.1 ], [4.9, 2.5, 4.5, 1.7 ], [7.3, 2.9, 6.3, 1.8 ], [6.7, 2.5, 5.8, 1.8 ], [7.2, 3.6, 6.1, 2.5 ], [6.5, 3.2, 5.1, 2.0 ], [6.4, 2.7, 5.3, 1.9 ], [6.8, 3.0, 5.5, 2.1 ], [5.7, 2.5, 5.0, 2.0 ], [5.8, 2.8, 5.1, 2.4 ], [6.4, 3.2, 5.3, 2.3 ], [6.5, 3.0, 5.5, 1.8 ], [7.7, 3.8, 6.7, 2.2 ], [7.7, 2.6, 6.9, 2.3 ], [6.0, 2.2, 5.0, 1.5 ], [6.9, 3.2, 5.7, 2.3 ], [5.6, 2.8, 4.9, 2.0 ], [7.7, 2.8, 6.7, 2.0 ], [6.3, 2.7, 4.9, 1.8 ], [6.7, 3.3, 5.7, 2.1 ], [7.2, 3.2, 6.0, 1.8 ], [6.2, 2.8, 4.8, 1.8 ], [6.1, 3.0, 4.9, 1.8 ], [6.4, 2.8, 5.6, 2.1 ], [7.2, 3.0, 5.8, 1.6 ], [7.4, 2.8, 6.1, 1.9 ], [7.9, 3.8, 6.4, 2.0 ], [6.4, 2.8, 5.6, 2.2 ], [6.3, 2.8, 5.1, 1.5 ], [6.1, 2.6, 5.6, 1.4 ], [7.7, 3.0, 6.1, 2.3 ], [6.3, 3.4, 5.6, 2.4 ], [6.4, 3.1, 5.5, 1.8 ], [6.0, 3.0, 4.8, 1.8 ], [6.9, 3.1, 5.4, 2.1 ], [6.7, 3.1, 5.6, 2.4 ], [6.9, 3.1, 5.1, 2.3 ], [5.8, 2.7, 5.1, 1.9 ], [6.8, 3.2, 5.9, 2.3 ], [6.7, 3.3, 5.7, 2.5 ], [6.7, 3.0, 5.2, 2.3 ], [6.3, 2.5, 5.0, 1.9 ], [6.5, 3.0, 5.2, 2.0 ], [6.2, 3.4, 5.4, 2.3 ], [5.9, 3.0, 5.1, 1.8 ] ]
    
    // target values
    public static let targets: [Double] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    
    public static func trainTestSplit(numOfTestItems: Int) -> ([[Double]], [Double], [[Double]], [Double]) {
        var trainData = self.data
        var trainTargets = self.targets

        // testing data and testing targets ids
        var testIDs: [Int] = []

        // fill testIDs array
        // Using an array of 50 random indices (For testing set) with guarenteed non-repetition.
        for _ in 0..<numOfTestItems {
            var randomInt = Int(arc4random_uniform(149))
            
            // Check to see if that random int already exists in the array and if it does, lets try until we get one that isn't.
            var success = false
            while success == false {
                if testIDs.contains(randomInt) {
                    randomInt = Int(arc4random_uniform(149))
                } else {
                    success = true
                }
            }
            
            // Finally append the value to the array
            testIDs.append(randomInt)
        }

        testIDs.sort{ $0 > $1 }  // order the indices for test set in descending order as it is easier to remove the last element from an array as arrays in swift are ordered lists.

        var testData: [[Double]] = []
        var testTargets: [Double] = []

        // Splitting the data into training and testing data
        for id in testIDs {
            //print(id)
            let testValData = trainData.remove(at: id)
            let testValTarget = trainTargets.remove(at: id)
            testData.append(testValData)
            testTargets.append(testValTarget)
        }
        
        return (trainData, trainTargets, testData, testTargets)
    }
    
    public static func outputLabels(for targetVals: [Double]) -> [String] {
        var outputLabels: [String] = []
        for val in targetVals {
            switch val {
            case 0:
                outputLabels.append(Self.targetNames[0])
            case 1:
                outputLabels.append(Self.targetNames[1])
            case 2:
                outputLabels.append(Self.targetNames[2])
            default:
                outputLabels.append("INVALID INPUT: \(val)")
            }
        }
        
        return outputLabels
    }
    
}
