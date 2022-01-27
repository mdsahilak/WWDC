import Foundation

// Euclidean distance calculation function
// Euclidean distance refers to the distance between any 2 points in n dimensional space
public func calculateEucDist(_ a: [Double], _ b: [Double]) -> Double {
    guard a.count == b.count else { fatalError("a and b do not have the same number of elements.") }
    let numberOfItems = a.count
    
    var initialValues: [Double] = []
    var newValues: [Double] = []
    
    for i in 0..<numberOfItems {
        initialValues.append(a[i])
        newValues.append(b[i])
    }
    
    var diffsSq: [Double] = []
    for i in 0..<numberOfItems {
        let diff = newValues[i] - initialValues[i]
        let sq = diff * diff
        diffsSq.append(sq)
    }
    
    let diffSqSum = diffsSq.reduce(0, +)
    let answer = sqrt(diffSqSum)
    
    return answer
}

