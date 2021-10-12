//
//  TaylorGLOInitializer.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/25/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation
import SwiftCMA

struct TaylorGLOInitializer {
	static func run(experiment: ResultsManager) throws {
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Initializing TaylorGLO...")
		
		let taylorGloConfig = experiment.config.experiment!.taylorGloConfig!
		
		// Hyperparameters.
		let order = taylorGloConfig.order
		let inputVariables = taylorGloConfig.inputVariables ?? TaylorGloConfig.defaultInputVariables
		let variables = inputVariables.split(separator: .lossFunctionDelimiter).map { $0.count }
		let populationSize = taylorGloConfig.populationSize
		let stepSigma = taylorGloConfig.sigma
		
		// Solution variables.
		let paramSetTypes = taylorGloConfig.taylorParameterSetTypes
		let taylorSizes = zip(paramSetTypes, variables).map { TaylorExpansion.parameterCount(forVariables: $0.1, order: order, parameterSetType: $0.0) }
		
		let startTaylors = zip(taylorSizes, variables).map { Array(repeating: 0.0, count: $0.0 + $0.1) }
		
		// Activation function variables.
		let startActivationTaylor: Vector? = taylorGloConfig.evolvedActivationFunction.flatMap { activationConfig in
			let taylorSize = TaylorExpansion.parameterCount(forVariables: 1, order: activationConfig.order)
			let variables = 1
			return Array(repeating: 0.0, count: taylorSize + variables)
		}
		
		// Learning rate variable.
		var evolvableLearningRateVariable: Double? = nil
		if taylorGloConfig.evolveLearningRate {
			fatalError("NOT IMPLEMENTED YET")
//			evolvableLearningRateVariable = experiment.config.evaluation.baseTrainingConfig.learningRateSchedule.initial
		}
		
		// Learning rate decay point variables.
		var evolvableDecayPoints = [Double]()
		if let decayPoints = taylorGloConfig.evolvedLearningRateDecayPoints {
			let baseSchedule = experiment.config.evaluation.baseTrainingConfig.learningRateSchedule
			evolvableDecayPoints.append(contentsOf: baseSchedule.decayEpochs.prefix(min(decayPoints, baseSchedule.decayEpochs.count)).map { Double($0) / Double(experiment.config.evaluation.baseTrainingConfig.fullTrainingEpochs) })
			if decayPoints > baseSchedule.decayEpochs.count {
				evolvableDecayPoints.append(contentsOf: Array(repeating: 0.0, count: decayPoints - baseSchedule.decayEpochs.count))
			}
		}
		
		// Create the encoded, evolvable vector.
		let (startSolution, searchSpace) = TaylorGLOEncoding.encode(
			taylorParameterLists: startTaylors,
			activationFunctionParameterList: startActivationTaylor,
			learningRate: evolvableLearningRateVariable,
			learningRateDecayPoints: evolvableDecayPoints.isEmpty ? nil : evolvableDecayPoints,
			config: taylorGloConfig
		)
		
		// Print some useful stats.
		LogHelper.shared.printWithTimestamp(type: .info, message: "Parameterization Input Variables: \(variables)")
		LogHelper.shared.printWithTimestamp(type: .info, message: "Parameterization Order: \(order)")
		LogHelper.shared.printWithTimestamp(type: .info, message: "CMA-ES Vector Size: \(startSolution.count)")
		if let decayEpochs = TaylorGLOEncoding.learningRateDecayEpochs(startSolution, config: taylorGloConfig, trainingConfig: experiment.config.evaluation.baseTrainingConfig) {
			LogHelper.shared.printWithTimestamp(type: .info, message: "Initial LR Decay Epochs: [\(decayEpochs.map { String($0) }.joined(separator: ", "))]")
		}
		
		// Create the initial CMA-ES object.
		let cmaes = CMAES(
			startSolution: startSolution,
			populationSize: populationSize,
			stepSigma: stepSigma,
			searchSpaceConfiguration: searchSpace
		)
		
		// Persist the CMA-ES object.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Saving checkpoint...")
		let folder = try experiment.currentGenerationFolder()
		try cmaes.save(checkpoint: folder.url.appendingPathComponent(ResultsManager.FileName.generationCheckpoint.rawValue))
		
		LogHelper.shared.printWithTimestamp(type: .completion, message: "TaylorGLO experiment ready.")
	}
}
