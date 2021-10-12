//
//  TaylorGLOGenerationFinisher.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/26/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation
import SwiftCMA

struct TaylorGLOGenerationFinisher {
	static func run(experiment: ResultsManager, intragenerationalCandidates: [SavedCandidate]) throws {
		// Load checkpoint.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Reading checkpoint...")
		let folder = try experiment.currentGenerationFolder()
		let cmaes = try CMAES.from(checkpoint: folder.url.appendingPathComponent(ResultsManager.FileName.generationCheckpoint.rawValue))
		let taylorGloConfig = experiment.config.experiment!.taylorGloConfig!
		
		// Restructure results and determine whether we should minimize or maximize fitness.
		let allResults: [(Vector, Double)] = intragenerationalCandidates.filter { $0.lossFunctionTaylor != nil && $0.fitness != nil}.map { candidate in
			let fitness = candidate.fitness!
			let minimizingObjective: Double = {
				let fitnessMetric = experiment.config.evolution!.fitnessMetric
				switch fitnessMetric.metric.maximize {
				case true:
					if fitnessMetric.metric == .accuracy {
						return 1.0 - fitness
					} else {
						return -fitness
					}
				case false:
					return fitness
				}
			}()
			return (candidate.encodedCandidate ?? candidate.lossFunctionTaylor!, minimizingObjective)
		} // [(candidate, fitness)]
		
		// Ensure we have enough results.
		let expectedCount = taylorGloConfig.populationSize
		let maxMissing = experiment.config.evolution!.maximumMissingEvaluations
		LogHelper.shared.printWithTimestamp(type: .info, message: "Found \(allResults.count) candidates, missing \(expectedCount - allResults.count) (maximum allowed \(maxMissing)), target population \(expectedCount).")
		guard allResults.count + maxMissing >= expectedCount else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Not enough results to move forward.")
			exit(CommandLineTool.ExitCode.missingJobs.rawValue) // Abort with non-zero exit code.
		}
		
		// Finish CMA-ES generation.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Finishing CMA-ES generation...")
		cmaes.finishEpoch(candidateFitnesses: allResults)
		
		// Save updated CMA-ES.
		try cmaes.save(checkpoint: folder.url.appendingPathComponent(ResultsManager.FileName.generationCheckpoint.rawValue))
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Saved checkpoint.")
	}
}
