//
//  RunTTest.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 4/9/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

struct RunTTest {
	static func run(experiments: [String]) throws {
		// Find results files.
		let resultsFiles: [File] = try experiments.map { experimentFolderPath in
			let folder = try Folder(path: experimentFolderPath)
			guard folder.containsFile(named: ResultsManager.FileName.resultsCSV.rawValue) else {
				LogHelper.shared.printWithTimestamp(type: .error, message: "Missing results file in: \(experimentFolderPath)")
				return nil
			}
			return try folder.file(named: ResultsManager.FileName.resultsCSV.rawValue)
			
		}.compactMap { $0 }
		guard resultsFiles.count >= 2 else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Not enough experiments with results. At least two required.")
			return
		}
		
		// Read results files.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Reading files...")
		let results = try resultsFiles.map { try ResultsManager.readCSV($0) }
		let numberOfRows = results.first!.1.count
		guard results.allSatisfy({ $0.0 == results.first!.0 }) else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Results headers do not match.")
			return
		}
		guard results.allSatisfy({ $0.1.count == numberOfRows }) else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Results must have equal numbers of rows for Student's t-Test (not of importance for Welch's t-Test).")
			return
		}
		
		// Calculate statistics for columns.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Calculating column statistics...")
		let desiredColumnIndex = results.first!.0.firstIndex(where: { $0 == "Validation Accuracy" || $0 == "Validation Mse" })!
		let desiredResultsColumns: [[Double]] = results.map { $0.1.transposed()[desiredColumnIndex].map { x in Double(x)! } }
		let sums = desiredResultsColumns.map { values in values.reduce(0, { $0 + $1 }) }
		let means = sums.map { $0 / Double(numberOfRows) }
		let stddevsBiasCorrected: [Double] = desiredResultsColumns.enumerated().map { (arg) in // = sqrt(mean(abs(x - x.mean())**2))
			let (idx, values) = arg
			let mean = means[idx]
			let variance = values.map { pow(fabs($0 - mean), 2) }.sum / Double(numberOfRows - 1)
			return sqrt(variance)
		}
		
		// Get column pairs.
		var columnPairs = [(Int, Int)]()
		for firstIdx in (0..<desiredResultsColumns.count) {
			for secondIdx in (0..<desiredResultsColumns.count) {
				guard firstIdx < secondIdx else { continue } // Avoid duplicates and testing against self.
				columnPairs.append((firstIdx, secondIdx))
			}
		}
		print(columnPairs)
		
		// Perform t-Tests.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Performing t-Tests...")
		for pair in columnPairs {
			let unbiasedVarianceEstimators = (
				pow(stddevsBiasCorrected[pair.0], 2),
				pow(stddevsBiasCorrected[pair.1], 2)
			)
			// Student's t-Test.
			let pooledStddev = sqrt((unbiasedVarianceEstimators.0 + unbiasedVarianceEstimators.1) / 2.0)
			let studentTStatistic = (means[pair.0] - means[pair.1]) / (pooledStddev * sqrt(2.0 / Double(numberOfRows)) )
			let studentDOF = 2*numberOfRows - 2
			let studentPValue2Tail = tDistributionCDF(nu: Double(studentDOF), t: studentTStatistic)
			// Welch's t-Test.
			let numSamples = (desiredResultsColumns[pair.0].count, desiredResultsColumns[pair.1].count)
			let welchTDenominatorInner = (unbiasedVarianceEstimators.0 / Double(numSamples.0)) + (unbiasedVarianceEstimators.1 / Double(numSamples.1))
			let welchTStatistic = (means[pair.0] - means[pair.1]) / sqrt(welchTDenominatorInner)
			let welchDOFDenominatorTerm1 = pow(stddevsBiasCorrected[pair.0], 4) / (pow(Double(numSamples.0), 2) * Double(numSamples.0 - 1))
			let welchDOFDenominatorTerm2 = pow(stddevsBiasCorrected[pair.1], 4) / (pow(Double(numSamples.1), 2) * Double(numSamples.1 - 1))
			let welchDOF = pow(welchTDenominatorInner, 2) / (welchDOFDenominatorTerm1 + welchDOFDenominatorTerm2)
			let welchPValue2Tail = tDistributionCDF(nu: welchDOF, t: welchTStatistic)
			
			print(sums)
			print(means)
			print(stddevsBiasCorrected.map { $0 * $0})
			
			// Print info.
			print("--------------------------------")
			print("-- Set 1: \(resultsFiles[pair.0].parent!.name)")
			print("-- Set 2: \(resultsFiles[pair.1].parent!.name)")
			print("")
			print("Student's t-Test (equal variances):")
			print("* t statistic: \(studentTStatistic)")
			print("* DOF: \(studentDOF)")
			print("* 2-tailed p-value: \(studentPValue2Tail * 2.0)")
			print("* 1-tailed p-value: \(studentPValue2Tail)")
			print("")
			print("Welch's t-Test:")
			print("* t statistic: \(welchTStatistic)")
			print("* DOF: \(welchDOF)")
			print("* 2-tailed p-value: \(welchPValue2Tail * 2.0)")
			print("* 1-tailed p-value: \(welchPValue2Tail)")
			print("")
			print("")
		}
		
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished.")
	}
}
