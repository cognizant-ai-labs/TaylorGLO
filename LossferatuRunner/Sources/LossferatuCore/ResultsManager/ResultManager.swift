//
//  ResultsManager.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

class ResultsManager {
	
	/// The experiment root.
	let root: Folder
	
	/// The experiment's configuration.
	let config: ExperimentConfig
	
	/// Initializes a new experiment directory with a given config file.
	static func createExperimentDirectory(_ experimentDir: String, configFile: String) throws {
		// Create root.
		guard !FileManager.default.fileExists(atPath: experimentDir) else {
			throw Error.directoryAlreadyExists
		}
		try FileManager.default.createDirectory(atPath: experimentDir, withIntermediateDirectories: true, attributes: nil)
		let root = try Folder(path: experimentDir)
		
		// Copy config.
		let configFile = try File(path: configFile)
		let config = try JSONDecoder().decode(ExperimentConfig.self, from: try configFile.read())
		try configFile.copy(to: root)
		try root.files.first!.rename(to: FileName.config.rawValue)
		
		// Create generations directory if necessary.
		switch config.kind {
		case .generational:
			try root.createSubfolder(named: FileName.generations.rawValue)
		case .oneShot:
			try root.createSubfolder(named: FileName.oneShot.rawValue)
		}
		
		// Add helper shell scripts.
		
		let runFile = try root.createFile(named: FileName.shRunGeneration.rawValue)
		try runFile.write(HelperScripts.runGeneration)
		"chmod u+x \(runFile.path)".runAsZshCommandLine()
		
		switch config.kind {
		case .generational:
			let analyzeFile = try root.createFile(named: FileName.shRunAnalysis.rawValue)
			try analyzeFile.write(HelperScripts.runAnalysis)
			"chmod u+x \(analyzeFile.path)".runAsZshCommandLine()
		case .oneShot:
			break
		}
	}
	
	/// Creates the results manager for an existing experiment directory.
	init(experimentDirectory: String) throws {
		guard FileManager.default.fileExists(atPath: experimentDirectory) else {
			throw Error.directoryDoesNotExist
		}
		root = try Folder(path: experimentDirectory)
		let configData = try root.file(named: FileName.config.rawValue).read()
		config = try JSONDecoder().decode(ExperimentConfig.self, from: configData)
	}
	
	// MARK: Generations
	
	/// Returns the number of generations.
	func countGenerations() throws -> Int {
		return try root.subfolder(named: FileName.generations.rawValue).subfolders.count()
	}
	
	/// Returns the folder for the current generation.
	func currentGenerationFolder() throws -> Folder {
		switch config.kind {
		case .generational:
			let generationsFolder = try root.subfolder(named: FileName.generations.rawValue)
			let subfolders = generationsFolder.subfolders.sorted { Int($1.name)! > Int($0.name)! }
			guard let generation = subfolders.last, generation.containsFile(named: FileName.inProgressMarker.rawValue) else {
				let folder = try generationsFolder.createSubfolder(named: subfolders.last.flatMap { String(Int($0.name)! + 1) } ?? "0")
				try folder.createFile(named: FileName.inProgressMarker.rawValue)
				return folder
			}
			return generation
		case .oneShot:
			let folder = try root.subfolder(named: FileName.oneShot.rawValue)
			try folder.createFile(named: FileName.inProgressMarker.rawValue)
			return folder
		}
	}
	
	/// Closes the current generation, ensuring it is in a complete state, as previous generations.
	func closeCurrentGeneration() throws {
		let generation = try currentGenerationFolder()
		try generation.file(named: FileName.inProgressMarker.rawValue).delete()
	}
	
	/// Returns the folder for the previous generation.
	func previousGenerationFolder() throws -> Folder {
		guard config.kind != .oneShot else { throw Error.methodCalledInOneShot }
		let current = try currentGenerationFolder()
		let prevName = String(Int(current.name)! - 1)
		return try current.parent!.subfolder(named: prevName)
	}
	
	/// If there isn't a checkpoint file in the current generation directory, this function
	/// copies the previous generation's checkpoint file.
	func copyCheckpointFromPreviousGenerationIfNeeded() throws {
		guard config.kind != .oneShot else { throw Error.methodCalledInOneShot }
		let current = try currentGenerationFolder()
		if !FileManager.default.fileExists(atPath: current.path + "/" + FileName.generationCheckpoint.rawValue) {
			let previous = try previousGenerationFolder()
			try previous.file(named: ResultsManager.FileName.generationCheckpoint.rawValue).copy(to: current)
		}
	}
	
	/// Returns a list of folders for all completed generations in ascending numerical order.
	func allCompletedGenerationFolders() throws -> [Folder] {
		switch config.kind {
		case .generational:
			let generationsFolder = try root.subfolder(named: FileName.generations.rawValue)
			return generationsFolder.subfolders.filter { folder in
				// Filter out folders that have an in-progress marker.
				return !folder.containsFile(named: FileName.inProgressMarker.rawValue)
			}.sorted(by: { Int($1.name)! > Int($0.name)! })
			
		case .oneShot:
			let oneShot = try root.subfolder(named: FileName.oneShot.rawValue)
			return [oneShot].filter { folder in
				// Filter out folders that have an in-progress marker.
				return !folder.containsFile(named: FileName.inProgressMarker.rawValue)
			}
		}
	}
	
	// MARK: Jobs
	
	private let jobNamesMutex = DispatchSemaphore(value: 1)
	
	/// Returns the job names for the current generation.
	func currentGenerationJobNames(includeCompleted: Bool = false) throws -> [String] {
		jobNamesMutex.wait()
		defer { jobNamesMutex.signal() }
		
		let generation = try currentGenerationFolder()
		let jobsString = try generation.file(named: FileName.jobNames.rawValue).readAsString()
		return jobsString.components(separatedBy: "\n").filter { !$0.isEmpty && (includeCompleted || !$0.hasPrefix("#")) }
	}
	
	/// Returns the one-shot's job name.
	func oneShotJobName() throws -> String {
		guard config.kind == .oneShot else { throw Error.methodCalledInOneShot }
		
		jobNamesMutex.wait()
		defer { jobNamesMutex.signal() }
		
		let oneShot = try root.subfolder(named: FileName.oneShot.rawValue)
		let jobsString = try oneShot.file(named: FileName.jobNames.rawValue).readAsString()
		return jobsString.components(separatedBy: "\n").first!.components(separatedBy: "# ").filter { !$0.isEmpty }.first!
	}
	
	/// Adds a new job name to the current generation.
	func add(jobName: String) throws {
		jobNamesMutex.wait()
		defer { jobNamesMutex.signal() }
		
		let folder = try currentGenerationFolder()
		try folder.createFileIfNeeded(withName: FileName.jobNames.rawValue)
		try folder.file(named: FileName.jobNames.rawValue).append("\(jobName)\n")
	}
	
	/// Marks the provided job name as finished.
	func markFinished(jobName: String) throws {
		jobNamesMutex.wait()
		defer { jobNamesMutex.signal() }
		
		let folder = try currentGenerationFolder()
		let jobsFile = try folder.file(named: FileName.jobNames.rawValue)
		let contents = try jobsFile.readAsString()
		let updatedContents = contents.components(separatedBy: "\n").map { line in
			if line == jobName {
				return "# \(jobName)"
			} else {
				return line
			}
		}.joined(separator: "\n")
		try jobsFile.write(updatedContents)
	}
	
	// MARK: Results
	
	/// Writes a set of values as a new row in a CSV file.
	static func writeCSV(_ values: [String], forColumns columns: [NamedColumnable], to file: File, sortColumns: Bool = false) throws {
		precondition(values.count == columns.count)
		var zipped = Array(zip(values, columns))
		zipped = sortColumns ? zipped.sorted(by: { $1.1.columnName > $0.1.columnName }) : zipped
		let finalValues = zipped.map { $0.0 }
		let finalColumns = zipped.map { $0.1 }
		// Add the header row if this is a new CSV.
		if try file.readAsString().isEmpty {
			try file.write(finalColumns.map { $0.columnName }.joined(separator: ",") + "\n")
		}
		// Add the new row.
		try file.append(finalValues.joined(separator: ",") + "\n")
	}
	
	/// Reads a CSV file and returns a tuple of column headers and rows.
	static func readCSV(_ file: File) throws -> ([String], [[String]]) {
		let lines = try file.readAsString().components(separatedBy: "\n").filter { !$0.isEmpty }
		let headers = lines[0].components(separatedBy: ",")
		let rows = lines.suffix(from: 1).map { $0.components(separatedBy: ",") }
		return (headers, rows)
	}
	
	// MARK: Candidates
	
	private func candidatesFile() throws -> File? {
		guard try currentGenerationFolder().containsFile(named: FileName.candidates.rawValue) else { return nil }
		return try currentGenerationFolder().file(named: FileName.candidates.rawValue)
	}
	
	private let savedCanidatesMutex = DispatchSemaphore(value: 1)
	
	func lockGenerationCandidates() {
		savedCanidatesMutex.wait()
	}
	
	func unlockGenerationCandidates() {
		savedCanidatesMutex.signal()
	}
	
	func getCandidates(forGeneration folder: Folder) throws -> [SavedCandidate] {
		guard folder.containsFile(named: FileName.candidates.rawValue) else { return [] }
		let file = try folder.file(named: FileName.candidates.rawValue)
		let data = try file.read()
		return try JSONDecoder().decode([SavedCandidate].self, from: data)
	}
	
	func getBestCandidate(forGeneration folder: Folder, bestMetric: EvaluationMetric) throws -> SavedCandidate {
		try getCandidates(forGeneration: folder).filter { $0.results != nil }.max(by: { rhs, lhs in
			if bestMetric.metric.maximize {
				return rhs.fitness! < lhs.fitness!
			} else {
				return rhs.fitness! > lhs.fitness!
			}
		})! // yikes, that's a lot of force unwrapping...
	}
	
	func getGenerationCandidates() throws -> [SavedCandidate] {
		return try getCandidates(forGeneration: currentGenerationFolder())
	}
	
	func saveGenerationCandidates(_ candidates: [SavedCandidate]) throws {
		let file = try currentGenerationFolder().createFileIfNeeded(withName: FileName.candidates.rawValue)
		let data = try JSONEncoder().encode(candidates)
		try file.write(data)
	}
	
	// MARK: Logs
	
	func generationErrorLog(named name: String) throws -> File {
		let errors = try currentGenerationFolder().createSubfolderIfNeeded(withName: FileName.errors.rawValue)
		return try errors.createFileIfNeeded(withName: name)
	}
	
	func generationLog(named name: String) throws -> File {
		let errors = try currentGenerationFolder().createSubfolderIfNeeded(withName: FileName.logs.rawValue)
		return try errors.createFileIfNeeded(withName: name)
	}
}

extension ResultsManager {
	enum FileName: String {
		// Directories.
		case generations = "generations"
		case oneShot = "one_shot"
		case errors = "errors"
		case logs = "logs"
		case analyses = "analyses"
		case children = "children"
		
		// Scripts.
		case shRunGeneration = "run_generation"
		case shRunAnalysis = "analyze"
		case shRunOneShot = "experiment_one_shot"
		case shRunBestCandidateOneShot = "experiment_best_candidate_one_shot"
		
		// Files.
		case config = "config.json"
		case jobNames = "job_names.txt"
		case studioLog = "studiolog.txt"
		case candidates = "candidates.json"
		case generationCheckpoint = "generation_checkpoint.json"
		case inProgressMarker = "IN_PROGRESS.marker"
		case resultsCSV = "results.csv"
		case statsCSV = "stats.csv"
		case analysesWolfram = "results.wl"
		case resultsNotebook = "results.nb"
		case bestCandidateOneShotConfig = "ExperimentConfig_best_candidate.json"
		
		case candidateOneShotConfigPrefix = "ExperimentConfig_"
		case candidateOneShotConfigSuffix = "_candidate.json"
	}
	
	enum Error: Swift.Error {
		case directoryAlreadyExists
		case directoryDoesNotExist
		case methodCalledInOneShot
		case methodCalledInGenerational
	}
}
