//
//  RunStartGeneration.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct RunStartGeneration {
	static func run(experimentDir: String) throws {
		// Ensure reachability.
		TrainingInterface().assertAvailable()
		
		let experiment = try ResultsManager(experimentDirectory: experimentDir)
		guard experiment.config.kind != .oneShot else {
			// Kick off one-shot training.
			try RunStartGeneration.runOneShot(experiment)
			return
		}
		
		// Copy checkpoint from previous generation if we haven't yet
		try experiment.copyCheckpointFromPreviousGenerationIfNeeded()
		
		// Kick off another generation.
		guard let experimentConfig = experiment.config.experiment else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Expected experiment config in config file.")
			exit(1)
		}
		switch experimentConfig.kind {
		case .gloTaylorClassification:
			try TaylorGLOGenerationStarter.run(experiment: experiment)
		case .gloTreeClassification:
			fatalError("NOT IMPLEMENTED!")
		}
	}
	
	private static func runOneShot(_ experiment: ResultsManager) throws {
		var jobName: String? = nil
		var attemptsRemaining = experiment.config.evaluation.maximumJobSubmissionRetries ?? 3
		while jobName == nil && attemptsRemaining > 0 {
			attemptsRemaining -= 1
			
			// Submit the job.
			LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Submitting job...")
			let evalPartition = experiment.config.evaluation.evalPartition
			jobName = TrainingInterface().submitTrainingJob(config: experiment.config.evaluation.baseTrainingConfig, experimentName: experiment.config.name, evaluationPartition: evalPartition, failureOutput: { failureOutput in
				try! experiment.generationErrorLog(named: "studio_run").write(failureOutput)
			})
			
			// Print an error message for failed submissions.
			if jobName == nil {
				LogHelper.shared.printWithTimestamp(type: .error, message: "Job submission failed (remaining attempts: \(attemptsRemaining))")
			}
		}
		
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Submitted one-shot job: \(jobName ?? "???")")
		try experiment.add(jobName: jobName ?? "???")
	}
}
