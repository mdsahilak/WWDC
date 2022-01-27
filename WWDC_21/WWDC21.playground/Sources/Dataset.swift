import Foundation

public struct FruitDataset {
    // MARK: - Column Names
    // Labels for what each of the items in data represents.
    public static let headers: [String] = ["color", "diameter", "label"]
    
    // MARK: - Training
    // Data for Training the Classifier
    public static let trainingData: [[String]] = [["Green", "3"], ["Yellow", "3"], ["Red", "1"], ["Red", "1"], ["Yellow", "3"]]
    
    // Labels for Training
    public static let trainingTargets: [String] = ["Apple", "Apple", "Grape", "Grape", "Lemon"]
    
    // MARK: - Testing
    // Data for Testing the Classifier
    public static let testingData: [[String]] = [["Green", "3"], ["Yellow", "4"], ["Red", "2"], ["Red", "1"], ["Yellow", "3"]]
    
    // Labels for Testing
    public static let testingTargets: [String] = ["Apple", "Apple", "Grape", "Grape", "Lemon"]
}
