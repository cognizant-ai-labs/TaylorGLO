//
//  RunAnalyzeExperiment.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 1/2/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

struct RunAnalyzeExperiment {
	static func run(experimentDir: String) throws {
		let experiment = try ResultsManager(experimentDirectory: experimentDir)
		guard experiment.config.kind != .oneShot else {
			try analyzeOneShot(experiment: experiment)
			return
		}
		
		// Get generations.
		LogHelper.shared.printWithTimestamp(type: .checking, message: "Running experiment analysis...")
		let completedGenerations = try experiment.allCompletedGenerationFolders()
		guard completedGenerations.count > 0 else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Analysis requires completed generations in experiment.")
			fatalError()
		}
		LogHelper.shared.printWithTimestamp(type: .checking, message: "Found \(completedGenerations.count) completed generation\(completedGenerations.count == 1 ? "" : "s").")
		
		// Create empty analyses directory.
		if experiment.root.containsSubfolder(named: ResultsManager.FileName.analyses.rawValue) {
			try experiment.root.subfolder(named: ResultsManager.FileName.analyses.rawValue).delete()
		}
		let analyses = try experiment.root.createSubfolderIfNeeded(withName: ResultsManager.FileName.analyses.rawValue)
		
		// Collate results CSVs.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Collating results CSVs...")
		var bestGeneration = 0
		var bestAccuracy = 0.0
		var bestMSE = Double.greatestFiniteMagnitude
		let resultsFile = try analyses.createFileIfNeeded(withName: ResultsManager.FileName.resultsCSV.rawValue)
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
			try ResultsManager.writeCSV(["\(i)"] + newValues + remarkCounts, forColumns: ["Generation"] + newHeaders + remarkHeaders, to: resultsFile)
			// Check if best.
			let accuracyOrMse = Double(newValues[newHeaders.firstIndex(where: { $0 == "Validation Accuracy_MAX" || $0 == "Validation Mse_MIN" })!])!
			if newHeaders.contains("Validation Accuracy_MAX") {
				if accuracyOrMse > bestAccuracy {
					bestAccuracy = accuracyOrMse
					bestGeneration = i
				}
			} else {
				if accuracyOrMse < bestMSE {
					bestMSE = accuracyOrMse
					bestGeneration = i
				}
			}
		}
		LogHelper.shared.printWithTimestamp(type: .info, message: "BestGeneration \(bestGeneration), BestAccuracy \(bestAccuracy), BestMSE: \(bestMSE)")
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished.")
		
		// Get best candidates from each generation.
		let bestMetric = experiment.config.evolution!.fitnessMetric // ?? EvaluationMetric(metric: .accuracy, dataset: .validation)
		let bestCandidates = try experiment.allCompletedGenerationFolders().map { folder in
			return try experiment.getBestCandidate(forGeneration: folder, bestMetric: bestMetric)
		}
		
		// Build Mathematica notebook.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Building Wolfram code...")
		let bestCandidate = bestCandidates[bestGeneration]
		let candidatesCSV: String = try {
			var rows = [String]()
			for generation in try experiment.allCompletedGenerationFolders() {
				let evaluations = try experiment.getCandidates(forGeneration: generation).filter { $0.results != nil }
				let maxFitness = evaluations.max(by: { rhs, lhs in
					if bestMetric.metric.maximize {
						return rhs.fitness! > lhs.fitness!
					} else {
						return rhs.fitness! < lhs.fitness!
					}
				})?.fitness // yikes, that's a lot of force unwrapping...
				if bestGeneration == Int(generation.name) {
					let testFitness = maxFitness ?? Double.nan
					guard testFitness < bestMSE + 0.001 || testFitness < bestAccuracy + 0.001 || testFitness > bestMSE - 0.001 || testFitness > bestAccuracy - 0.001 else {
						print("Internal inconsistency. Best fitness does not match previous.")
						print("- Generation: \(generation)")
						print(bestMetric)
						print(maxFitness)
						print(bestMSE)
						print(bestAccuracy)
						print(evaluations.map { $0.results![bestMetric]! })
						print("\n")
						print(evaluations.map { $0.results! })
						exit(1)
					}
				}
				let bestIndex = evaluations.firstIndex(where: { $0.fitness == maxFitness })
				for (idx, evaluation) in evaluations.enumerated() {
					let isBestInGeneration = idx == bestIndex
					// generation, fitness, isBestInGeneration, params...
					let values = [generation.name, String(evaluation.fitness!), isBestInGeneration ? "True" : "False"] + (evaluation.encodedCandidate ?? evaluation.lossFunctionTaylor!).map { String($0) }
					rows.append(values.joined(separator: ","))
				}
			}
			return rows.joined(separator: "\n")
		}()
		let wolframCode = try MathematicaCode.analysisNotebook(resultsCSV: resultsFile.readAsString(), candidatesCSV: candidatesCSV, best: bestCandidate, experiment: experiment)
		let notebookFile = try analyses.createFileIfNeeded(withName: ResultsManager.FileName.analysesWolfram.rawValue)
		try notebookFile.write(wolframCode)
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Invoking wolframscript to create Mathematica notebook...")
		let runCommand = """
			echo "$(cat \(notebookFile.path))\\nUsingFrontEnd[NotebookSave[notebook, \\"\(experiment.root.path)/\(ResultsManager.FileName.resultsNotebook.rawValue)\\"]]" | wolframscript
			"""
		LogHelper.shared.printWithTimestamp(type: .info, message: "COMMAND: \(runCommand)")
		runCommand.runAsZshCommandLine()
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished.")
		
		// Create one-shot training config for best candidate.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Creating one-shot experiment configuration for best candidate...")
		let bestCandidateOneShotConfigFile = try analyses.createFileIfNeeded(withName: ResultsManager.FileName.bestCandidateOneShotConfig.rawValue)
		try writeOneShotConfig(forCandidate: bestCandidate, to: bestCandidateOneShotConfigFile, baseConfig: experiment.config)
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Config available at: \(bestCandidateOneShotConfigFile.path(relativeTo: Folder.root))")
		
		// Create training configs for each generation's best candidate.
		for candidate in bestCandidates { // testing, Full Training
			let configFilename = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "Gen\(candidate.generation)_best" + ResultsManager.FileName.candidateOneShotConfigSuffix.rawValue
			let oneShotFile = try analyses.createFileIfNeeded(withName: configFilename)
			try writeOneShotConfig(forCandidate: candidate, to: oneShotFile, baseConfig: experiment.config)
		}
		for candidate in bestCandidates { // validation, Full Training
			let configFilename = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "GenVal\(candidate.generation)_best" + ResultsManager.FileName.candidateOneShotConfigSuffix.rawValue
			let oneShotFile = try analyses.createFileIfNeeded(withName: configFilename)
			try writeOneShotConfig(forCandidate: candidate, to: oneShotFile, baseConfig: experiment.config, dataset: .validation)
		}
		let pretrainedGroup = DispatchGroup()
		for candidate in bestCandidates {
			pretrainedGroup.enter()
			DispatchQueue.global().async {
				defer {
					pretrainedGroup.leave()
				}
				do {
					guard TrainingInterface().checkJobModelTarExists(jobFile: candidate.jobName!) else {
						LogHelper.shared.printWithTimestamp(type: .error, message: "No model tar for Gen \(candidate.generation) best!")
						return
					}
					// testing, Using Pretrained Base
					let configFilenameTest = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "GenPretrained\(candidate.generation)_best" + ResultsManager.FileName.candidateOneShotConfigSuffix.rawValue
					let oneShotFileTest = try analyses.createFileIfNeeded(withName: configFilenameTest)
					try writeOneShotConfig(forCandidate: candidate, to: oneShotFileTest, baseConfig: experiment.config, fromPretrainedJob: candidate.jobName)
					// validation, Using Pretrained Base
					let configFilenameVal = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "GenPretrainedVal\(candidate.generation)_best" + ResultsManager.FileName.candidateOneShotConfigSuffix.rawValue
					let oneShotFileVal = try analyses.createFileIfNeeded(withName: configFilenameVal)
					try writeOneShotConfig(forCandidate: candidate, to: oneShotFileVal, baseConfig: experiment.config, dataset: .validation, fromPretrainedJob: candidate.jobName)
				} catch {
					LogHelper.shared.printWithTimestamp(type: .error, message: "Error in candidate block: \(error)")
			    }
			}
		}
		pretrainedGroup.wait()
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Created configs for each generation's best candidate.")
		
		// Create invocation shell script.
		let bestCandidateOneShotInvocation = try analyses.createFile(named: ResultsManager.FileName.shRunBestCandidateOneShot.rawValue)
		try bestCandidateOneShotInvocation.write(HelperScripts.runOneShotChildExperiment(at: bestCandidateOneShotConfigFile.path(relativeTo: experiment.root), generation: bestGeneration))
		"chmod u+x \(bestCandidateOneShotInvocation.path)".runAsZshCommandLine()
		LogHelper.shared.printWithTimestamp(type: .completion, message: "One-shot experiment available at: \(bestCandidateOneShotConfigFile.path(relativeTo: Folder.root))")
	}
	
	// MARK: Configurations
	
	/// Creates a one-shot training config for the specified candidate in the specified file.
	static func writeOneShotConfig(forCandidate candidate: SavedCandidate?, to file: File, baseConfig: ExperimentConfig, dataset: DatasetPartition = .testing, fromPretrainedJob: String? = nil, configChanges: ((ExperimentConfig) -> ExperimentConfig)? = nil) throws {
		var oneShotConfig = baseConfig
		oneShotConfig.name = oneShotConfig.name + "_OneShot"
		oneShotConfig.kind = .oneShot
		oneShotConfig.evaluation.maximumJobSubmissionRetries = 10 // Don't want to have failed submissions.
		oneShotConfig.evaluation.baseTrainingConfig.trainingAttempts = 10 // Let's retry things up to 10 times.
		if let candidate = candidate {
			// If this is a config for a specific candidate, modify the configuration appropriately.
			oneShotConfig.evaluation.baseTrainingConfig.lossFunction = candidate.lossFunctionTensorFlow!
			if let activationTf = candidate.activationFunctionTensorFlow { // Set the activation function if it exists.
				oneShotConfig.evaluation.baseTrainingConfig.overrideActivationFunction = activationTf
			}
			if let taylorGloConfig = baseConfig.experiment!.taylorGloConfig {
				if let lrDecayEpochs = candidate.encodedCandidate.flatMap({ TaylorGLOEncoding.learningRateDecayEpochs($0, config: taylorGloConfig, trainingConfig: baseConfig.evaluation.baseTrainingConfig) }) {
					oneShotConfig.evaluation.baseTrainingConfig.learningRateSchedule.decayEpochs = lrDecayEpochs
				}
				if let lr = candidate.encodedCandidate.flatMap({ TaylorGLOEncoding.learningRate($0, config: taylorGloConfig) }) {
					oneShotConfig.evaluation.baseTrainingConfig.learningRateSchedule.initial = lr
				}
			}
		}
		if let jobName = fromPretrainedJob {
			oneShotConfig.name = oneShotConfig.name + "_FromPretrained"
			oneShotConfig.evaluation.baseTrainingConfig.epochs = oneShotConfig.evaluation.baseTrainingConfig.fullTrainingEpochs // Train up to the full number of epochs.
			oneShotConfig.evaluation.baseTrainingConfig.startEpoch = baseConfig.evaluation.baseTrainingConfig.epochs
			oneShotConfig.evaluation.baseTrainingConfig.baseModelS3Tar = "s3://YOURBUCKET-datasets/experiments/\(jobName)/modeldir.tar|checkpoint.pth.tar"
			// Simulate learning rate decay.
			var simulatedInitialLearningRate = baseConfig.evaluation.baseTrainingConfig.learningRateSchedule.initial
			for epoch in (0..<oneShotConfig.evaluation.baseTrainingConfig.startEpoch!) {
				simulatedInitialLearningRate *= oneShotConfig.evaluation.baseTrainingConfig.learningRateSchedule.everyEpochGamma
				if oneShotConfig.evaluation.baseTrainingConfig.learningRateSchedule.decayEpochs.contains(epoch) {
					simulatedInitialLearningRate *= oneShotConfig.evaluation.baseTrainingConfig.learningRateSchedule.decayEpochGamma
				}
			}
			oneShotConfig.evaluation.baseTrainingConfig.learningRateSchedule.initial = simulatedInitialLearningRate
		} else {
			oneShotConfig.evaluation.baseTrainingConfig.epochs = oneShotConfig.evaluation.baseTrainingConfig.fullTrainingEpochs
		}
		oneShotConfig.evaluation.evaluatedMetrics = oneShotConfig.evaluation.evaluatedMetrics.map { evalMetric in
			guard candidate == nil || (evalMetric.dataset != .testing || oneShotConfig.name.contains("TESTINGEVAL")) else {
				LogHelper.shared.printWithTimestamp(type: .error, message: "Found testing dataset metric in experiment config. Possible testing data contamination. Aborting!")
				fatalError()
			}
			var newEvalMetric = evalMetric
			newEvalMetric.dataset = dataset
			return newEvalMetric
		}
		oneShotConfig.evaluation.baseTrainingConfig.fgsm = nil // Avoid evaluating with FGSM attacks.
		oneShotConfig.evolution = nil
		oneShotConfig.experiment = nil
		
		// Make final changes with closure.
		oneShotConfig = configChanges?(oneShotConfig) ?? oneShotConfig
		
		let oneShotConfigData = try JSONEncoder().encode(oneShotConfig)
		try file.write(oneShotConfigData)
	}
	
	// MARK: - One Shot Analysis
	
	private static func analyzeOneShot(experiment: ResultsManager) throws {
		// Create empty analyses directory.
		if experiment.root.containsSubfolder(named: ResultsManager.FileName.analyses.rawValue) {
			try experiment.root.subfolder(named: ResultsManager.FileName.analyses.rawValue).delete()
		}
		let analyses = try experiment.root.createSubfolderIfNeeded(withName: ResultsManager.FileName.analyses.rawValue)
		
		// Get the candidate.
		guard let oneShot = try? experiment.allCompletedGenerationFolders().first else {
			LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Incomplete one-shot experiment at: \(experiment.root.path(relativeTo: Folder.root))")
			return
		}
		let candidateJobName = try experiment.oneShotJobName()
		
		// FGSM attack experiments.
		for (epsilon, title) in [(0.05, "005"), (0.1, "01"), (0.15, "015"), (0.2, "02"), (0.25, "025"), (0.3, "03")] {
			let configFilenameTest = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "PretrainedFGSM\(title).json"
			let oneShotFileTest = try analyses.createFileIfNeeded(withName: configFilenameTest)
			try writeOneShotConfig(forCandidate: nil, to: oneShotFileTest, baseConfig: experiment.config, fromPretrainedJob: candidateJobName) { config in
				// Modify config.
				var updatedConfig = config
				updatedConfig.evaluation.baseTrainingConfig.fgsm = FGSMAttackConfig(epsilon: epsilon)
				return updatedConfig
			}
		}
		
		// Grid storage experiment.
		for (grid, title) in [("'-1:1:51::-1:1:51'", "Unit51x51")] {
			let configFilenameTest = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "PretrainedGrid.json"
			let oneShotFileTest = try analyses.createFileIfNeeded(withName: configFilenameTest)
			try writeOneShotConfig(forCandidate: nil, to: oneShotFileTest, baseConfig: experiment.config, fromPretrainedJob: candidateJobName) { config in
				// Modify config.
				var updatedConfig = config
				updatedConfig.evaluation.baseTrainingConfig.evaluationGridConfig = EvaluationGridConfig(grid: grid)
				return updatedConfig
			}
		}
		
		// Activation storage experiment.
		do {
			let configFilenameTest = ResultsManager.FileName.candidateOneShotConfigPrefix.rawValue + "LossActivationsTest.json"
			let oneShotFileTest = try analyses.createFileIfNeeded(withName: configFilenameTest)
			try writeOneShotConfig(forCandidate: nil, to: oneShotFileTest, baseConfig: experiment.config) { config in
				// Modify config.
				var updatedConfig = config
				updatedConfig.evaluation.baseTrainingConfig.evaluationStoreLossActivations = true
				return updatedConfig
			}
		}
		
		
		// Create invocation shell script.
		let oneShotInvocation = try analyses.createFile(named: ResultsManager.FileName.shRunOneShot.rawValue)
		try oneShotInvocation.write(HelperScripts.runOneShotOneShotChildExperiment())
		"chmod u+x \(oneShotInvocation.path)".runAsZshCommandLine()
		LogHelper.shared.printWithTimestamp(type: .completion, message: "One-shot experiments available at: \(oneShotInvocation.path(relativeTo: Folder.root))")
	}
}
