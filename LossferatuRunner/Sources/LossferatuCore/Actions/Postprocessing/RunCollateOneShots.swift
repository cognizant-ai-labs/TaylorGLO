//
//  RunCollateOneShots.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 1/21/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

struct RunCollateOneShots {
	static func run(experimentsDir: String, lessVerbose: Bool = false) throws {
		let experimentsFolder = try Folder(path: experimentsDir)
		let children = Array(experimentsFolder.subfolders)
		var csvs = [File]()
		for experimentDir in children {
			if !lessVerbose {
				LogHelper.shared.printWithTimestamp(type: .checking, message: "Reading experiment \(experimentDir.name)...")
			}
			
			// Get one-shot.
			let folder = try experimentDir.subfolder(named: ResultsManager.FileName.oneShot.rawValue)
			let csv = try folder.file(named: ResultsManager.FileName.resultsCSV.rawValue)
			csvs.append(csv)
		}
		
		// Collate results CSVs.
		if !lessVerbose {
			LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Collating results CSVs...")
		}
		let resultsFile = try experimentsFolder.createFileIfNeeded(withName: ResultsManager.FileName.resultsCSV.rawValue)
		let statsFile = try experimentsFolder.createFileIfNeeded(withName: ResultsManager.FileName.statsCSV.rawValue)
		try resultsFile.write("") // Clear the file if it exists.
		try statsFile.write("") // Clear the file if it exists.
		var headers = [String]()
		var allValues = [[Double]]()
		for (i, csv) in csvs.enumerated() {
			// Read CSV.
			let tabularData = try ResultsManager.readCSV(csv)
			// Collate data.
			let newHeaders: [String] = tabularData.0
			let newValues: [String] = tabularData.1.first!
			headers = newHeaders
			allValues.append(newValues.map { Double($0)! })
			try ResultsManager.writeCSV(["\(children[i].name)"] + newValues, forColumns: ["Experiment"] + newHeaders, to: resultsFile)
		}
		
		// Stats file.
		let sums = allValues.transposed().map { values in values.reduce(0, { $0 + $1 }) }
		let maxes = allValues.transposed().map { values in values.reduce(0, { max($0, $1) }) }
		let means = sums.map { $0 / Double(csvs.count) }
		let stddevs: [Double] = allValues.transposed().enumerated().map { (arg) in // = sqrt(mean(abs(x - x.mean())**2))
			let (idx, values) = arg
			let mean = means[idx]
			let variance = values.map { pow(fabs($0 - mean), 2) }.sum / Double(csvs.count)
			return sqrt(variance)
		}
		try ResultsManager.writeCSV(means.map { String($0) } + stddevs.map { String($0) }, forColumns: headers.map { "Mean \($0)" } + headers.map { "Stddev \($0)" }, to: statsFile)
		
		// Check for outliers
		if headers.contains(where: { $0 == "Validation Accuracy" || $0 == "Validation Mse" }) {
			let idx = headers.firstIndex(where: { $0 == "Validation Accuracy" || $0 == "Validation Mse" })!
			if means[idx] > 0.5 || maxes[idx] > 0.8 {
				let outliers = allValues.map { $0[idx] }.filter { $0 < 0.2 }.count
				if outliers > 0 {
					LogHelper.shared.printWithTimestamp(type: .info, message: "Found outliers: \(outliers)")
				}
			}
		}
		
		if !lessVerbose {
			LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished.")
		}
	}
}
