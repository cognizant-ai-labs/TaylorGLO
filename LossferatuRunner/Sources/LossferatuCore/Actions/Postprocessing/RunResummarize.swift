//
//  RunResummarize.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 3/3/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

struct RunResummarize {
	static func run(experimentsDirsDir: String, lessVerbose: Bool = false) throws {
		let experimentsFolder = try Folder(path: experimentsDirsDir)
		var children = Array(experimentsFolder.subfolders).filter { !$0.name.hasPrefix("____") }
		
		// Sort directories.
		children = children.sorted(by: { left, right in
			if left.name.hasPrefix("Gen") && right.name.hasPrefix("Gen") {
				if left.name.contains("Val") && !right.name.contains("Val") {
					return false
				} else if !left.name.contains("Val") && right.name.contains("Val") {
					return true
				} else {
					if !left.name.contains("Pretrained") && !right.name.contains("Pretrained") {
						return Int(left.name.components(separatedBy: "Gen").last!.components(separatedBy: "Val").last!)! < Int(right.name.components(separatedBy: "Gen").last!.components(separatedBy: "Val").last!)!
					} else if left.name.contains("Pretrained") && !right.name.contains("Pretrained") {
						return false
					} else if !left.name.contains("Pretrained") && right.name.contains("Pretrained") {
						return true
					} else {
						return Int(left.name.components(separatedBy: "GenPretrained").last!.components(separatedBy: "Val").last!)! < Int(right.name.components(separatedBy: "GenPretrained").last!.components(separatedBy: "Val").last!)!
					}
				}
			} else {
				return left.name < right.name
			}
		})
		
		// Iterate over children.
		var csvNamePairs = [(File, String)]()
		for experimentsDir in children {
			if !lessVerbose {
				LogHelper.shared.printWithTimestamp(type: .checking, message: "Reading experiments in \(experimentsDir.name)...")
			}
			
			// Run collateoneshots.
			do {
				try RunCollateOneShots.run(experimentsDir: experimentsDir.path, lessVerbose: true)
			} catch {
				LogHelper.shared.printWithTimestamp(type: .error, message: "Unable to collate \(experimentsDir.name)!")
			}
			
			// Get the stats.
			if experimentsDir.containsFile(named: ResultsManager.FileName.statsCSV.rawValue) {
				let csv = try experimentsDir.file(named: ResultsManager.FileName.statsCSV.rawValue)
				csvNamePairs.append((csv, experimentsDir.name))
			}
			
			// Check one child.
			guard !experimentsDir.name.hasPrefix("Gen") else {
				continue
			}
			guard experimentsDir.subfolders.count() > 0 else {
				LogHelper.shared.printWithTimestamp(type: .error, message: "\(experimentsDir.name) has no subdirectories!")
				continue
			}
			for experimentDir in [experimentsDir.subfolders.first!] {
				
				// Read experiment.
				let experiment = try ResultsManager(experimentDirectory: experimentDir.path)
				
				// Validate name.
				let nameComponents = experimentsDir.name.lowercased().components(separatedBy: "_").filter { !$0.isEmpty }
				let comparisonName = nameComponents.joined(separator: "_")
				let trainingConfig = experiment.config.evaluation.baseTrainingConfig
				if !comparisonName.hasPrefix("\(trainingConfig.target.rawValue)_") {
					LogHelper.shared.printWithTimestamp(type: .error, message: "\(experimentsDir.name) name / config mismatch! Config has target: \(trainingConfig.target.rawValue)")
				}
				if !(comparisonName.contains("_\(trainingConfig.model.rawValue.replacingOccurrences(of: "_", with: ""))_") || comparisonName.hasSuffix("_\(trainingConfig.model.rawValue.replacingOccurrences(of: "_", with: ""))")) {
					LogHelper.shared.printWithTimestamp(type: .error, message: "\(experimentsDir.name) name / config mismatch! Config has model: \(trainingConfig.model.rawValue)")
				}
				
			}
		}
		
		// Collate stats CSVs.
		if !lessVerbose {
			LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Collating stats CSVs...")
		}
		let statsFile = try experimentsFolder.createFileIfNeeded(withName: ResultsManager.FileName.statsCSV.rawValue)
		try statsFile.write("") // Clear the file if it exists.
		var headers = [String]()
		var allValues = [[Double]]()
		for (i, (csv, name)) in csvNamePairs.enumerated() {
			// Read CSV.
			let tabularData = try ResultsManager.readCSV(csv)
			// Collate data.
			let newHeaders: [String] = tabularData.0
			let newValues: [String] = tabularData.1.first!
			headers = newHeaders
			allValues.append(newValues.map { Double($0)! })
			try ResultsManager.writeCSV([name] + newValues, forColumns: ["EXPERIMENT_DIR"] + newHeaders, to: statsFile)
		}
		if !lessVerbose {
			LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished.")
		}
	}
}
