//
//  TaylorGLOEncoding.swift
//  LossferatuRunner
//
//  Created by Santiago Gonzalez on 4/21/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import SwiftCMA

/// Tools to manipulate the evolvable TaylorGLO representation.
///
/// Format: `[TaylorParamsLists,LossFnTransitionPoints,TaylorActivation,LR,LRDecayPoints]`
///
struct TaylorGLOEncoding {
	
	static func encode(taylorParameterLists: [Vector], activationFunctionParameterList: Vector?, learningRate: Double?, learningRateDecayPoints: [Double]?, config: TaylorGloConfig) -> (Vector, CMAES.SearchSpaceConfiguration?) {
		
		// Encode.
		var encoded = [Double]()
		for params in taylorParameterLists {
			encoded.append(contentsOf: params)
		}
		if let activation = activationFunctionParameterList {
			encoded.append(contentsOf: activation)
		}
		if let lr = learningRate {
			encoded.append(lr) // TODO: properly encode LR
		}
		if var lrDecayPoints = learningRateDecayPoints {
			if let encoding = config.evolvedLearningRateDecayPointRepresentationEncoding {
				switch encoding {
				case .normalCDF: lrDecayPoints = lrDecayPoints.map { normalCDFInverse(p: $0) }
				case .logistic: lrDecayPoints = lrDecayPoints.map { logitFunction(p: $0) }
				}
			}
			encoded.append(contentsOf: lrDecayPoints)
		}
		
		// Create search space config if needed.
		let searchSpace: CMAES.SearchSpaceConfiguration? = learningRateDecayPoints.flatMap { lrDecayPoints in
			guard let lrDecayRepresentationScaling = config.evolvedLearningRateDecayPointRepresentationScaling else {
				print("evolvedLearningRateDecayPointRepresentationScaling must be set in TaylorGLO configuration.")
				exit(1)
			}
			let bounds = Array(repeating: nil, count: encoded.count - lrDecayPoints.count) +
				Array(repeating: config.evolvedLearningRateDecayPointRepresentationEncoding == nil ? 0.0...1.0 : nil, count: lrDecayPoints.count)
			var scaling = Array(repeating: 1.0, count: encoded.count - lrDecayPoints.count)
			scaling.append(contentsOf: Array(repeating: lrDecayRepresentationScaling, count: lrDecayPoints.count))
			return CMAES.SearchSpaceConfiguration(
				bounds: bounds,
				scalingFactors: scaling,
				bchm: .darwinianReflection
			)
		}
		return (encoded, searchSpace)
	}
	
	static func taylorParameterLists(_ from: Vector, config: TaylorGloConfig) -> [Vector] {
		let inputVariables = config.inputVariables ?? TaylorGloConfig.defaultInputVariables
		let variables = inputVariables.split(separator: .lossFunctionDelimiter).map { $0.count }
		let paramSetTypes = config.taylorParameterSetTypes
		let taylorSizes = zip(paramSetTypes, variables).map { TaylorExpansion.parameterCount(forVariables: $0.1, order: config.order, parameterSetType: $0.0) }
		
		let totalTaylors = config.evolvedLossFunctions
		return (0..<totalTaylors).map { idx in
			from.prefix(taylorSizes[0...idx].enumerated().reduce(0, { $0 + $1.1 + variables[$1.0] })).suffix(taylorSizes[idx] + variables[idx])
		}
	}
	
	static func taylorTransitionPoints(_ from: Vector, config: TaylorGloConfig) -> [Double] {
		guard config.evolvedLossFunctions > 0 else { return [] }
		guard config.evolvedLossFunctions > 1 else { return [0.0] }
		
		let inputVariables = config.inputVariables ?? TaylorGloConfig.defaultInputVariables
		let variables = inputVariables.count
		let paramSetTypes = config.taylorParameterSetTypes
		let taylorSizes = paramSetTypes.map { TaylorExpansion.parameterCount(forVariables: variables, order: config.order, parameterSetType: $0) }
		let totalTaylorSize = taylorSizes.reduce(0, { $0 + $1 + variables })
		
		let totalTaylors = config.evolvedLossFunctions
		let transitionPoints = (config.evolvedLossFunctionTransitions ?? false) ? totalTaylors - 1 : 0
		return [0.0] + from.prefix(totalTaylorSize + transitionPoints).suffix(transitionPoints)
	}
	
	static func taylorTransitionEpochs(_ from: Vector, config: TaylorGloConfig, trainingConfig: TrainingConfig) -> [Int] {
		return TaylorGLOEncoding.taylorTransitionPoints(from, config: config).map { Int(floor($0 * Double(trainingConfig.fullTrainingEpochs))) }
	}
	
	static func taylorActivationParameterList(_ from: Vector, config: TaylorGloConfig) -> Vector? {
		guard let activationConfig = config.evolvedActivationFunction else { return nil }
		let taylorSize = TaylorExpansion.parameterCount(forVariables: 1, order: activationConfig.order)
		let variables = 1
		let encodedLRAndActivation = from.suffix(config.evolvedLearningRateDecayPoints ?? 0 + 1 + taylorSize + variables)
		return Array(encodedLRAndActivation.prefix(taylorSize + variables))
	}
	
	static func learningRate(_ from: Vector, config: TaylorGloConfig) -> Double? {
		guard config.evolveLearningRate else { return nil }
		let encodedLR = from.suffix(config.evolvedLearningRateDecayPoints ?? 0 + 1).first!
		return encodedLR // TODO: properly decode LR
	}
	
	static func learningRateDecayPoints(_ from: Vector, config: TaylorGloConfig) -> [Double]? {
		guard let decayPoints = config.evolvedLearningRateDecayPoints else { return nil }
		let points = from.suffix(decayPoints)
		if let encoding = config.evolvedLearningRateDecayPointRepresentationEncoding {
			switch encoding {
			case .normalCDF: return points.map { normalCDF(x: $0) }
			case .logistic: return points.map { logisticFunction(x: $0) }
			}
		}
		return Array(points)
	}
	
	static func learningRateDecayEpochs(_ from: Vector, config: TaylorGloConfig, trainingConfig: TrainingConfig) -> [Int]? {
		return TaylorGLOEncoding.learningRateDecayPoints(from, config: config).flatMap { points in
			return points.map { Int(floor($0 * Double(trainingConfig.fullTrainingEpochs))) }
		}
	}
	
}
