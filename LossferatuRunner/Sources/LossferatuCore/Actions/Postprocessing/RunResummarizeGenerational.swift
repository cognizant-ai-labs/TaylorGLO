//
//  RunResummarizeGenerational.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 3/4/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

struct RunResummarizeGenerational {
	static func run(experimentsDir: String) throws {
		let experimentsFolder = try Folder(path: experimentsDir)
		let children = Array(experimentsFolder.subfolders).filter { !$0.name.hasPrefix("____") }
		
		let summaryColumns = ["Name", "Generations", "Children", "Best Generation", "Best Val Accuracy/MSE", "Best Test Accuracy/MSE", "Best Pretrained Test Accuracy/MSE", "Baseline Accuracy/MSE"]
		var summaryRows = [[String]]()
		let summaryRowsMutex = DispatchSemaphore(value: 1)
		
		let dispatchGroup = DispatchGroup()
		for experimentDir in children {
			dispatchGroup.enter()
			DispatchQueue.global().async {
				defer {
					dispatchGroup.leave() // We have completed the work for this job.
				}
				do {
			
					LogHelper.shared.printWithTimestamp(type: .checking, message: "READING EXPERIMENT \(experimentDir.name)...")

					// Read experiment.
					let experiment = try ResultsManager(experimentDirectory: experimentDir.path)
					guard experiment.config.kind != .oneShot else {
						LogHelper.shared.printWithTimestamp(type: .info, message: "Skipping one-shot experiment.")
						return
					}
					
					// Validate name.
					let trainingConfig = experiment.config.evaluation.baseTrainingConfig
					guard experimentDir.name.lowercased().hasPrefix("\(trainingConfig.target.rawValue)_") else {
						LogHelper.shared.printWithTimestamp(type: .error, message: "\(experimentDir.name) name / config mismatch! Config has target: \(trainingConfig.target.rawValue)")
						return
					}
					guard experimentDir.name.lowercased().contains("_\(trainingConfig.model.rawValue.replacingOccurrences(of: "_", with: ""))_") else {
						LogHelper.shared.printWithTimestamp(type: .error, message: "\(experimentDir.name) name / config mismatch! Config has model: \(trainingConfig.model.rawValue)")
						return
					}
					
					// Check generations.
					let completedGenerations = try experiment.allCompletedGenerationFolders()
					guard completedGenerations.count > 0 else {
						LogHelper.shared.printWithTimestamp(type: .info, message: "No completed generations. Skipping \(experimentDir.name)...")
						return
					}
					
					// Iterate over all generations.
					var bestGeneration = 0
					var bestAccuracy = 0.0
					for (i, folder) in completedGenerations.enumerated() {
						// Read CSV.
						let csv = try folder.file(named: ResultsManager.FileName.resultsCSV.rawValue)
						let tabularData = try ResultsManager.readCSV(csv)
						// Read candidates.
						let candidates = try experiment.getCandidates(forGeneration: folder)
						// Process remarks.
						let remarkSets = candidates.map { $0.remarks }
						let remarkHeaders = SavedCandidate.Remark.allCases.map { $0.rawValue }
						let remarkCounts = SavedCandidate.Remark.allCases.map { remark in
							return String(remarkSets.filter { $0.contains(remark) }.count)
						}
						// Collate data.
						let newHeaders: [String] = tabularData.0.suffix(from: 1).map { header in
							return tabularData.1.map { "\(header)_\($0[0])" }
						}.flatMap { $0 }
						let newValues: [String] = tabularData.1.map { $0.suffix(from: 1) }.transposed().flatMap { $0 }
						// Check if best.
						let accuracy = Double(newValues[newHeaders.firstIndex(where: { $0 == "Validation Accuracy_MAX" || $0 == "Validation Mse_MIN" })!])!
						if accuracy > bestAccuracy {
							bestAccuracy = accuracy
							bestGeneration = i
						}
					}
					
					// Get best generation's child's accuracy.
					var bestTestAccuracy: Double? = nil
					if experimentDir.containsSubfolder(named: ResultsManager.FileName.children.rawValue) {
						let childrenDir = try experimentDir.subfolder(named: ResultsManager.FileName.children.rawValue)
						if childrenDir.containsSubfolder(named: "Gen\(bestGeneration)") {
							let childFolder = try childrenDir.subfolder(named: "Gen\(bestGeneration)")
							if childFolder.containsFile(named: ResultsManager.FileName.statsCSV.rawValue) {
								let statsFile = try childFolder.file(named: ResultsManager.FileName.statsCSV.rawValue)
								let tabularStats = try ResultsManager.readCSV(statsFile)
								bestTestAccuracy = Double(tabularStats.1.first![tabularStats.0.firstIndex(where: { $0 == "Mean Validation Accuracy" || $0 == "Mean Validation Mse" })!])
							}
						}
					}
					
					// Get best generation's pretrained child's accuracy.
					var bestPretrainedTestAccuracy: Double? = nil
					if experimentDir.containsSubfolder(named: ResultsManager.FileName.children.rawValue) {
						let childrenDir = try experimentDir.subfolder(named: ResultsManager.FileName.children.rawValue)
						if childrenDir.containsSubfolder(named: "GenPretrained\(bestGeneration)") {
							let childFolder = try childrenDir.subfolder(named: "GenPretrained\(bestGeneration)")
							if childFolder.containsFile(named: ResultsManager.FileName.statsCSV.rawValue) {
								let statsFile = try childFolder.file(named: ResultsManager.FileName.statsCSV.rawValue)
								let tabularStats = try ResultsManager.readCSV(statsFile)
								bestPretrainedTestAccuracy = Double(tabularStats.1.first![tabularStats.0.firstIndex(where: { $0 == "Mean Validation Accuracy" || $0 == "Mean Validation Mse" })!])
							}
						}
					}
					
					// Try to get the baseline.
					var baselineAccuracy: Double? = nil
					let baselinesDir = try Folder(path: experimentsDir.components(separatedBy: "RUNS_").first!).subfolder(named: "RUNS_Baseline")
					let baselineName = experimentDir.name.components(separatedBy: "_").filter { !["k3","k4","tk3","tk4","ulk3","ulk4"].contains($0) }.joined(separator: "_")
					if baselinesDir.containsSubfolder(named: baselineName) {
						let baselineFolder = try baselinesDir.subfolder(named: baselineName)
						if baselineFolder.containsFile(named: ResultsManager.FileName.statsCSV.rawValue) {
							let statsFile = try baselineFolder.file(named: ResultsManager.FileName.statsCSV.rawValue)
							let tabularStats = try ResultsManager.readCSV(statsFile)
							baselineAccuracy = Double(tabularStats.1.first![tabularStats.0.firstIndex(where: { $0 == "Mean Validation Accuracy" || $0 == "Mean Validation Mse" })!])
						}
					}
					
					// Get children.
					var childrenCount = 0
					if experimentDir.containsSubfolder(named: ResultsManager.FileName.children.rawValue) {
						let childrenDir = try experimentDir.subfolder(named: ResultsManager.FileName.children.rawValue)
						try RunResummarize.run(experimentsDirsDir: childrenDir.path, lessVerbose: true)
						childrenCount = childrenDir.subfolders.count()
					}
					
					// Create row.
					summaryRowsMutex.wait()
					summaryRows.append([
						experimentDir.name,
						String(completedGenerations.count),
						String(childrenCount),
						String(bestGeneration),
						String(bestAccuracy),
						bestTestAccuracy.flatMap { String($0) } ?? "",
						bestPretrainedTestAccuracy.flatMap { String($0) } ?? "",
						baselineAccuracy.flatMap { String($0) } ?? ""
					])
					summaryRowsMutex.signal()
				} catch {
					LogHelper.shared.printWithTimestamp(type: .error, message: "Error in block for experiment \(experimentDir.name): \(error)")
				}
			}
		}
		dispatchGroup.wait()
		
		// Sort summary rows.
		summaryRows = summaryRows.sorted(by: { $1[0] > $0[0] })
		
		// Write summary.
		let resultsFile = try experimentsFolder.createFileIfNeeded(withName: ResultsManager.FileName.resultsCSV.rawValue)
		try resultsFile.write("") // Clear the file if it exists.
		for row in summaryRows {
			try ResultsManager.writeCSV(row, forColumns: summaryColumns, to: resultsFile)
		}
		
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished.")
	}
}
