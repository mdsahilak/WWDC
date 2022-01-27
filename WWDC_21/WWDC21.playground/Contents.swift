import PlaygroundSupport
import SwiftUI
/*:
# Generic Decision Tree Classifier
## In this Playground, we'll explore an Algorithm called a Decision Tree and test it on a dataset of fruits.
### We will explore how Algorithms appear to 'Learn' or to make 'Decisions' by building a model for decision making using examples.
- - -
 */
//: ## Dataset Exploration
//: First, lets explore the Dataset. The Fruit Dataset contains information that describes the characteristics of a few different types of fruits. In this playground, we will explore how we can train a Decision Tree Classifier to learn the uniques characteristics that differentiate the different kinds of fruits from each other and then classify fruits with their correct name if given only the characteristics.
//: - Note: We are using a very simple dataset here with a few items each having only 2 characteristics for ease of understanding for this demo. The same algorithm can be applied to more complex datasets like for example the Iris Dataset, a classic Machine Learning Dataset.
// Data Points for Training - the color and diameter of each fruit
let trainingData: [[String]] = FruitDataset.trainingData
// The correct names of the fruit corresponding to each row in the training data
let trainingTargets: [String] = FruitDataset.trainingTargets

//: ---
//: ## Utility Functions
//:  A few methods to help us parse the data and build the decision tree.
// MARK: Find the unique values for a column in a dataset.
func uniqueValues<T: Hashable>(rows: [[T]], Column: Int) -> Set<T> {
    var ans: Set<T> = []
    for row in rows {
        let val = row[Column]
        ans.insert(val)
    }
    return ans
}

uniqueValues(rows: trainingData, Column: 0)

// MARK: Counts the number of each type of example in a dataset.
func classCounts<T>(rows: [[T]], targets: [String]) -> [String: Int] {
    var counts: [String: Int] = [:]
    
    for i in 0..<rows.count {
        let label = targets[i] //"\(rawLabel)"
        if counts[label] == nil {
            // if the label isnt there, then add it with count value as 0
            counts[label] = 0
        }
        // otherwise just append one to the count value corresponding to the label
        counts[label]! += 1
    }
    return counts
}

classCounts(rows: trainingData, targets: trainingTargets)

// MARK: Test if a value is numeric.
func isNumeric<T>(_ value: T) -> Bool {
    if value is Int || value is Float || value is Double {
        return true
    } else {
        return false
    }
}

isNumeric(50.0)

//: ---
//: ## Question
//: This struct just records a 'column number' (e.g., 0 for Color) and a 'column value' (e.g., Green). The 'match' method is used to compare the feature value in an example to the feature value stored in the question.
// MARK: A Question is used to partition a dataset.
struct Question<T: Comparable>: CustomStringConvertible {
    var column: Int
    var value: T
    
    // Custom string convertible-protocol required implementation
    var description: String {
        var condition: String {
            if isNumeric(value) {
                return ">="
            } else {
                return "=="
            }
        }
        return "Is the \(FruitDataset.headers[self.column]) \(condition) \(self.value)?"
    }
    
    // Compare the feature value in an example to the feature value in this question.
    func match(_ exampleRow: [T]) -> Bool {
        let val = exampleRow[column]
        if isNumeric(value) {
            return val >= self.value
        } else {
            return val == self.value
        }
    }
    
}

let q = Question(column: 0, value: "Green")
q.match(trainingData[0])
q

//: ---
//: ## Partition
//: For each row in the dataset, check if it matches the question. If so, add it to 'true rows', otherwise, add it to 'false rows'.
func partition<T: Comparable>(rows: [[T]], targets: [String], question: Question<T>) -> ([[T]], [String], [[T]], [String]) {
    var trueRows: [[T]] = []
    var trueTargets: [String] = []
    var falseRows: [[T]] = []
    var falseTargets: [String] = []
    
    for (i, row) in rows.enumerated() {
        if question.match(row)  {
            trueRows.append(row)
            trueTargets.append(targets[i])
        } else {
            falseRows.append(row)
            falseTargets.append(targets[i])
        }
    }
    return (trueRows, trueTargets, falseRows, falseTargets)
}

let (trueRowsxx, trueTargetsxx, falseRowsxx, falseTargetsxx) = partition(rows: trainingData, targets: trainingTargets, question: q)
trueRowsxx
falseRowsxx

//: ---
//: ## Gini Impurity
//: Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset.
// MARK: Calculate the Gini Impurity for a list of Rows
func gini<T>(rows: [[T]], targets: [String]) -> Double {
    let counts: [String: Int] = classCounts(rows: rows, targets: targets)
    var impurity: Double = 1
    for label in counts {
        let probOfLabel = Double(counts[label.key]!) / Double(rows.count)
        impurity -= pow(probOfLabel, 2)
    }
    return impurity
}

gini(rows: trainingData, targets: trainingTargets)
// For More Information, Visit: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity

//: ---
//: ## Information Gain
//:  The uncertainty of the starting node, minus the weighted impurity of two child nodes.
// MARK: Calculate the Information Gain at a Node
func infoGain<T>(leftRows: [[T]], leftTargets: [String], rightRows: [[T]], rightTargets: [String], currentUncertainity: Double) -> Double {
    let leftWeight = Double(leftRows.count) / Double(leftRows.count + rightRows.count)
    let rightWeight = 1 - leftWeight // or Double(right.count) / Double(left.count + right.count)
    
    let weightAvgImpurity = leftWeight * gini(rows: leftRows, targets: leftTargets) + rightWeight * gini(rows: rightRows, targets: rightTargets)
    let informationGain = currentUncertainity - weightAvgImpurity
    
    return informationGain
}

infoGain(leftRows: trueRowsxx, leftTargets: trueTargetsxx, rightRows: falseRowsxx, rightTargets: falseTargetsxx, currentUncertainity: gini(rows: trainingData, targets: trainingTargets))
// Fore More Information, Visit: https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain

//: ---
//: ## Best Question Finder
//: Find the best question to ask by iterating over every feature / value and calculating the information gain.
// MARK: Best Question to Ask
func findBestQuestion<T: Hashable & Comparable>(rows: [[T]], targets: [String]) -> (Double, Question<T>) {
    var bestGain: Double = 0
    var bestQuestion: Question<T> = Question(column: 0, value: rows.first![0]) // random starter question
    let currentUncertainity: Double = gini(rows: rows, targets: targets)
    let numOfFeatures: Int = rows.first!.count
    
    for column in 0..<numOfFeatures {
        let values = uniqueValues(rows: rows, Column: column)
        
        for value in values {
            let question = Question(column: column, value: value)
            let (trueRows, trueTargets, falseRows, falseTargets) = partition(rows: rows, targets: targets, question: question)
            if trueRows.count == 0 || falseRows.count == 0 {
                // do nothing - Skip the split if it doesn't divide the dataset
                continue
            }
            let gain = infoGain(leftRows: trueRows, leftTargets: trueTargets, rightRows: falseRows, rightTargets: falseTargets, currentUncertainity: currentUncertainity)
            if gain >= bestGain {
                bestGain = gain
                bestQuestion = question
            }
            
        }
        
    }
    
    return (bestGain, bestQuestion)
}

let (bestGain, bestQuestion) = findBestQuestion(rows: trainingData, targets: trainingTargets)
bestGain
bestQuestion

//: ---
//: ## Leaf
//: This holds a dictionary of class (e.g., "Apple") -> number of times it appears in the rows from the training data that reach this leaf.
// MARK: A Leaf Node Classifies Data
struct Leaf {
    var predictions: [String: Int]
    
    init<T>(rows: [[T]], targets: [String]) {
        self.predictions = classCounts(rows: rows, targets: targets)
    }
}

//: ---
//: ## Decision Node
//: This holds a reference to the question, and to the two child nodes.
// MARK: A Decision Node Asks a Question
struct DecisionNode<T: Comparable> {
    var question: Question<T>
    var trueBranch: Any
    var falseBranch: Any
    
}

//: ---
//: ## Decision Tree
//:  Decision Tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.  It is a way to display an algorithm that only contains conditional control statements.
// MARK: Recursive Function to Build the Tree
func BuildTree<T: Hashable & Comparable>(rows: [[T]], targets: [String]) -> Any {
    // Try partitioing the dataset on each of the unique attribute, calculate the information gain and return the question that produces the highest gain.
    let (gain, question) = findBestQuestion(rows: rows, targets: targets)
    
    // Base Case - No further information gain. Since we can ask no further questions, we'll return a Leaf.
    if gain == 0.0 {
        let predictions = Leaf(rows: rows, targets: targets)
        return predictions
    }
    
    // If we reach here, we have found a useful feature / value to partition on.
    let (trueRows, trueTargets, falseRows, falseTargets) = partition(rows: rows, targets: targets, question: question)
    
    // Recursively build the true branch
    let trueBranch = BuildTree(rows: trueRows, targets: trueTargets)
    // Recursively build the false branch
    let falseBranch = BuildTree(rows: falseRows, targets: falseTargets)
    
    // Return a Question node. This records the best feature / value to ask at this point, as well as the branches to follow dependingo on the answer.
    return DecisionNode(question: question, trueBranch: trueBranch, falseBranch: falseBranch)
}

//: ---
//: ## Tree Visualization
//: Printing a  visual representation of the decision tree to the Console
// MARK: Printing function for the tree
func printTree<T: Comparable>(node: Any, spacing: String = "", dTypeInstance: T) {
    // Base case - we've reached a leaf.
    if let leaf = node as? Leaf {
        print(spacing + "Predict", leaf.predictions)
        return
    }
    
    // Print the question at this node.
    let dNode = node as! DecisionNode<T>
    print(spacing + dNode.question.description)
    
    // Call this function recursively on the true branch
    print(spacing + "--> True:")
    printTree(node: dNode.trueBranch, spacing: spacing + "  ", dTypeInstance: dTypeInstance)
    
    // Call this function recursively on the false branch
    print(spacing + "--> False:")
    printTree(node: dNode.falseBranch, spacing: spacing + "  ", dTypeInstance: dTypeInstance)
}

// MARK: Output for a a classified leaf with % Confidence
func vizPredictionsAtLeaf(predictions: [String: Int]) -> [String: String] {
    // A nicer way to print the predictions at a leaf.
    let total = Double(predictions.values.reduce(0, +))
    var probabilities: [String: String] = [:]
    
    // Turn the output values into a % representation
    for label in predictions.keys {
        probabilities[label] = String((Double(predictions[label]!) / total * 100)) + "%"
    }
    
    // The Beautified Data
    return probabilities
}

// MARK: Classifies the Object based on the Decision Tree's Questions
func classify<T: Comparable>(sample: [T], node: Any) -> [String: String] {
    // Base case - we've reached a leaf.
    if let leaf = node as? Leaf {
        return vizPredictionsAtLeaf(predictions: leaf.predictions)
    }
    
    // Decide whether to follow the true-branch or the false-branch. Compare the feature / value stored in the node, to the example we're considering.
    let dNode = node as! DecisionNode<T>
    if dNode.question.match(sample) {
        return classify(sample: sample, node: dNode.trueBranch)
    } else {
        return classify(sample: sample, node: dNode.falseBranch)
    }
}

//: ---
//: # Decision Tree Classifier
//: Lets Build, Visualize and Test the Decision Tree

// MARK: Call the 'BuildTree(...)' Recursive Function on the training dataset
let myTree = BuildTree(rows: trainingData, targets: trainingTargets)


// MARK: Let us Visualize what the tree looks like internally in the Console
consoleLineBreak("Decision Tree Visualisation")
printTree(node: myTree, dTypeInstance: trainingData.first!.first!)


// MARK: Testing the Decision Tree on Data it has not seen before.
let testingData: [[String]] = FruitDataset.testingData
let testingTargets: [String] = FruitDataset.testingTargets

consoleLineBreak("Classifier Evaluation")
for (i, row) in testingData.enumerated() {
    let actualVal = testingTargets[i]
    let prediction = classify(sample: row, node: myTree)
    
    print("Prediction: \(prediction) | AcutalValue: \(actualVal)")
}

//: ### The output in the Console shows what our Decision Tree predicted a fruit would be and with how much confidence, along with the correct answer from our dataset. When we evalute this output, we can see that our algorithm has learned to differentiate between fruits by continously dividing them using more and more precise questions. And it Works!
//: - Note: Since this is Generic Implementation of a Decision Tree, This algorithm also works on other data types, not just strings. We can use datasets containg Integers or Booleans or even our own Custom Data Structures(provided they conform to the relevant protocols).
// MARK: -
// - Thank you for checking out my playground. Hope you have great day!
// - MD Sahil AK

consoleLineBreak("Playground Execution Completed")
PlaygroundPage.current.finishExecution()

// End //
