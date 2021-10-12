//
//  RunCheckGeneration.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation
import SwiftGenetics

struct RunCheckGeneration {
	static func run(experimentDir: String) throws {
		// Ensure reachability.
		TrainingInterface().assertAvailable()
		
		let experiment = try ResultsManager(experimentDirectory: experimentDir)
		guard experiment.config.kind != .oneShot else {
			// Check results of one-shot training.
			try RunCheckGeneration.runOneShot(experiment)
			return
		}
		let folder = try experiment.currentGenerationFolder()
		let generation = Int(folder.name)!
		LogHelper.shared.printWithTimestamp(type: .checking, message: "Checking jobs for generation \(generation)...")
		
		// Check results for generation.
		let jobs = try experiment.currentGenerationJobNames()
		if jobs.isEmpty {
			LogHelper.shared.printWithTimestamp(type: .error, message: "No jobs found!")
		}
		var incompleteJobs = [String]()
		let incompleteJobsMutex = DispatchSemaphore(value: 1)
		
		// Iterate over jobs.
		var jobResults = [String: (TrainingResults, Set<SavedCandidate.Remark>)]()
		let jobResultsMutex = DispatchSemaphore(value: 1)
		let dispatchGroup = DispatchGroup()
		for job in jobs {
			dispatchGroup.enter()
			DispatchQueue.global().async {
				defer {
					dispatchGroup.leave() // We have completed the work for this job.
				}
				do {
					// Check job results.
					let results = try checkJobResults(jobName: job, experiment: experiment, incompleteJobHandler: { incompleteJob in
						incompleteJobsMutex.wait()
						incompleteJobs.append(job)
						incompleteJobsMutex.signal()
					})
					jobResultsMutex.wait()
					jobResults[job] = results
					jobResultsMutex.signal()
				} catch {
					LogHelper.shared.printWithTimestamp(type: .error, message: "Error in block for job \(job): \(error)")
				}
			}
		}
		dispatchGroup.wait()
		
		// Load candidates.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Reading candidates...")
		experiment.lockGenerationCandidates() // Make sure nobody changes the candidates underneath our feet.
		defer { experiment.unlockGenerationCandidates() }
		var intragenerationalCandidates = try experiment.getGenerationCandidates()
		
		// Update and save candidates.
		if !jobResults.isEmpty {
			for job in jobResults.keys {
				let idx = intragenerationalCandidates.firstIndex(where: { $0.jobName == job })!
				let results = jobResults[job]!.0
				intragenerationalCandidates[idx].results = results
				intragenerationalCandidates[idx].fitness = results[experiment.config.evolution!.fitnessMetric]
				intragenerationalCandidates[idx].remarks.formUnion(jobResults[job]!.1)
			}
			try experiment.saveGenerationCandidates(intragenerationalCandidates)
			LogHelper.shared.printWithTimestamp(type: .completion, message: "Updated candidates.")
		}
		
		// Check whether all jobs are complete.
		guard !incompleteJobs.contains(where: { $0 != "???" }) && incompleteJobs.count <= experiment.config.evolution!.maximumMissingEvaluations else {
			exit(CommandLineTool.ExitCode.incompleteJob.rawValue) // Abort with non-zero exit code.
		}
		
		// Experiment-specific finish.
		guard let experimentConfig = experiment.config.experiment else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Expected experiment config in config file.")
			fatalError()
		}
		switch experimentConfig.kind {
		case .gloTaylorClassification:
			try TaylorGLOGenerationFinisher.run(experiment: experiment, intragenerationalCandidates: intragenerationalCandidates)
		case .gloTreeClassification:
			fatalError("NOT IMPLEMENTED!")
		}
		
		// Calculate statistics
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Calculating statistics...")
		let concreteResults = intragenerationalCandidates.compactMap { $0.results }
		let resultsFile = try experiment.currentGenerationFolder().createFileIfNeeded(withName: ResultsManager.FileName.resultsCSV.rawValue)
		let keys = Array(concreteResults.first!.keys)
		let meanValues: [String] = keys.map { key in
			let reasonableValues = concreteResults.filter { $0[key]! != Double.greatestFiniteMagnitude && $0[key]! != Double.infinity && $0[key]! != Double.nan }
			let mean = reasonableValues.reduce(0.0, { $0 + $1[key]! }) / Double(concreteResults.count)
			return String(format: "%.4f", mean)
		}
		let maxValues: [String] = keys.map { key in
			let reasonableValues = concreteResults.filter { $0[key]! != Double.greatestFiniteMagnitude && $0[key]! != Double.infinity && $0[key]! != Double.nan }
			let max = reasonableValues.compactMap { $0[key] }.max() ?? Double.nan
			return String(format: "%.4f", max)
		}
		let minValues: [String] = keys.map { key in
			let reasonableValues = concreteResults.filter { $0[key]! != Double.greatestFiniteMagnitude && $0[key]! != Double.infinity && $0[key]! != Double.nan }
			let min = reasonableValues.compactMap { $0[key] }.min() ?? Double.nan
			return String(format: "%.4f", min)
		}
		try ResultsManager.writeCSV(["MAX"] + maxValues, forColumns: ["STATISTIC"] + keys, to: resultsFile, sortColumns: true)
		try ResultsManager.writeCSV(["MEAN"] + meanValues, forColumns: ["STATISTIC"] + keys, to: resultsFile, sortColumns: true)
		try ResultsManager.writeCSV(["MIN"] + minValues, forColumns: ["STATISTIC"] + keys, to: resultsFile, sortColumns: true)
		
		// Cleanup.
		try LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished generation \(experiment.currentGenerationFolder().name).")
		try experiment.closeCurrentGeneration()
	}
	
	private static func runOneShot(_ experiment: ResultsManager) throws {
		// Get the job name.
		guard let job = try experiment.currentGenerationJobNames().first else {
			if try experiment.currentGenerationJobNames(includeCompleted: true).count > 0 {
				LogHelper.shared.printWithTimestamp(type: .info, message: "Already fetched all jobs. Nothing to do.")
				return
			} else {
				LogHelper.shared.printWithTimestamp(type: .error, message: "No jobs found!")
				return
			}
		}
		LogHelper.shared.printWithTimestamp(type: .checking, message: "Checking job: \(job)")
		
		// Check job results.
		let results = try checkJobResults(jobName: job, experiment: experiment, incompleteJobHandler: { incompleteJob in
			exit(CommandLineTool.ExitCode.incompleteJob.rawValue) // Abort with non-zero exit code.
		})
		
		// Handle results.
		if let successfulResults = results {
			let keys = Array(successfulResults.0.keys)
			let values = keys.map { String(successfulResults.0[$0]!) }
			let resultsFile = try experiment.currentGenerationFolder().createFileIfNeeded(withName: ResultsManager.FileName.resultsCSV.rawValue)
			try ResultsManager.writeCSV(values, forColumns: keys, to: resultsFile, sortColumns: true)
		}
		
		// Save studio log.
		let log = TrainingInterface().fetchJobLog(jobFile: job)
		let logFile = try experiment.currentGenerationFolder().createFileIfNeeded(withName: ResultsManager.FileName.studioLog.rawValue)
		try logFile.write(log)
		
		// Close experiment.
		try experiment.closeCurrentGeneration()
	}
	
	private static func checkJobResults(jobName job: String, experiment: ResultsManager, incompleteJobHandler: (String) -> Void) throws -> (TrainingResults, Set<SavedCandidate.Remark>)? {
		let evalPartition = experiment.config.evaluation.evalPartition
		let (jobState, possibleResults) = TrainingInterface().checkJobResults(jobFile: job, config: experiment.config.evaluation.baseTrainingConfig, evaluationPartition: evalPartition)
		guard jobState == .finished else {
			LogHelper.shared.printWithTimestamp(type: .info, message: "Job incomplete\(jobState == .running ? " (running)" : ""): \(job)")
			incompleteJobHandler(job)
			return nil
		}
		
		// Handle completed job
		try experiment.markFinished(jobName: job)
		guard var results = possibleResults else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Job failed successfully: \(job)")
			let results = [
				EvaluationMetric(metric: .loss, dataset: .validation): Double.greatestFiniteMagnitude,
				EvaluationMetric(metric: .accuracy, dataset: .validation): 0.0,
				EvaluationMetric(metric: .mse, dataset: .validation): Double.greatestFiniteMagnitude
			]
			return (results, [.jobFailed])
		}
		// Normalize NaN values into real values.
		for metric in results.keys {
			if results[metric]!.isNaN {
				results[metric] = metric.metric.maximize ? -Double.greatestFiniteMagnitude : Double.greatestFiniteMagnitude
			}
 		}
		// Done!
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Job finished: \(job)")
		return (results, [])
	}
}
