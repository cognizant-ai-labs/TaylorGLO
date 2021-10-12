//
//  ExperimentConfig.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct ExperimentConfig: Codable {
	enum Kind: String, Codable {
		/// One single training session.
		case oneShot
		/// A generational training modality.
		case generational
	}
	
	/// The experiment's name.
	var name: String
	/// The general type of the experiment.
	var kind: Kind
	
	var evaluation: EvaluationConfig
    var evolution: EvolutionConfig?
    var experiment: ExperimentConfig?
	
	struct EvaluationConfig: Codable {
	    enum Evaluator: String, Codable {
			/// Train using *studio.ml*.
			case studio
		}
		
		/// The base config from which training parameters are drawn.
		var baseTrainingConfig: TrainingConfig
		/// What will be evaluating each candidate.
		let evaluator: Evaluator
		/// What metrics are recorded by the evaluator.
		var evaluatedMetrics: [EvaluationMetric]
		/// The maximum number of times that a job resubmission can be attempted, or nil for none.
		var maximumJobSubmissionRetries: Int?
		
		var evalPartition: DatasetPartition {
			assert(Set(evaluatedMetrics.map { $0.dataset }).count == 1) // TODO: handle multi-dataset metrics better.
			return evaluatedMetrics.first!.dataset
		}
	}
	
	struct EvolutionConfig: Codable {
		/// The maximum number of generations that can run.
		let maxGenerations: Int
		/// The fitness metric that evolution runs against.
		let fitnessMetric: EvaluationMetric
		/// The maximum number of evaluations that can be missing while allowing the next generation to proceed.
		let maximumMissingEvaluations: Int
	}
	
	struct ExperimentConfig: Codable {
		enum Kind: String, Codable {
			case gloTreeClassification
			case gloTaylorClassification
		}
		
		let kind: Kind
		
		let taylorGloConfig: TaylorGloConfig?
		let treeGloConfig: TreeGloConfig?
	}
}

// MARK: - Experiment Configs

protocol GenerationalExperimentConfig: Codable {
	/// Population size.
	var populationSize: Int { get }
}

struct ActivationFunctionConfig: Codable {
	/// The order of the Taylor expansion activation function.
	let order: Int
}

struct TaylorGloConfig: GenerationalExperimentConfig {
	
	/// Encoding strategies for real numbers.
	enum RealEncodingStrategy: String, Codable {
		/// Pass the genotype real through the normal distribution's CDF to get the phenotype.
		case normalCDF
		/// Pass the genotype real through the logistic function to get the phenotype.
		case logistic
	}
	
	/// Population size.
	let populationSize: Int
	/// The order of the Taylor expansion loss function.
	let order: Int
	/// CMA-ES sigma step-size.
	let sigma: Double
	/// Should the learning rate also be evolved?
	let evolveLearningRate: Bool
	/// Should the loss function invariant not be checked?
	let ignoreInvariantCheck: Bool?
	/// The inputs to the loss function.
	let inputVariables: [LossFunctionInput]?
	/// The number of evolved learning rate decay points, or `nil` if using the default schedule.
	let evolvedLearningRateDecayPoints: Int?
	/// The CMA-ES scaling factor for evolved learning rate decay points. Since the values are
	/// in [0,1], we want some factor less than one since `sigma` will likely be too large.
	let evolvedLearningRateDecayPointRepresentationScaling: Double?
	let evolvedLearningRateDecayPointRepresentationEncoding: RealEncodingStrategy?
	
	let evolvedLossFunctionTransitions: Bool?
	
	let evolvedActivationFunction: ActivationFunctionConfig?
	
	static let defaultInputVariables: [LossFunctionInput] = [.trueLabel, .scaledLogits]
	
	/// How many loss functions are simultaneously evolved.
	var evolvedLossFunctions: Int {
		return (inputVariables ?? TaylorGloConfig.defaultInputVariables).split(separator: .lossFunctionDelimiter, omittingEmptySubsequences: true).count
	}
	
	var taylorParameterSetTypes: [TaylorExpansion.ParameterSetType] {
		let concreteInputVariables = inputVariables ?? TaylorGloConfig.defaultInputVariables
		let variableChunks = concreteInputVariables.split(separator: .lossFunctionDelimiter, omittingEmptySubsequences: true)
		let parameterSetTypes: [TaylorExpansion.ParameterSetType] = variableChunks.map { lossFunctionInputVariables in
			let lossFunctionInputVariablesArray = Array(lossFunctionInputVariables)
			return TaylorExpansion.ParameterSetType.missingZeroPartialTerms(forVarIndices: [
				lossFunctionInputVariablesArray.firstIndex(of: .scaledLogits),
				lossFunctionInputVariablesArray.firstIndex(of: .unscaledLogits),
				lossFunctionInputVariablesArray.firstIndex(of: .residuals)
			].compactMap { $0 }) // we don't care about terms where d/dy = 0
		}
		return parameterSetTypes
	}
}

struct TreeGloConfig: GenerationalExperimentConfig {
	/// Population size.
	let populationSize: Int
}

// MARK: - Loss Function Parameterization

enum LossFunctionInput: String, Codable {
	case trueLabel
	case scaledLogits
	case unscaledLogits
	case residuals
	case trainingCompleted
	
	case lossFunctionDelimiter
	
	var tensorflowName: String {
		switch self {
		case .trueLabel: return "loss_x"
		case .scaledLogits: return "loss_y"
		case .unscaledLogits: return "loss_logits"
		case .residuals: return "(loss_y-loss_x)"
		case .trainingCompleted: return "loss_t"
		case .lossFunctionDelimiter: fatalError("Non-input token.")
		}
	}
	
	var mathematicaName: String {
		switch self {
		case .trueLabel: return "x"
		case .scaledLogits: return "y"
		case .unscaledLogits: return "logits"
		case .residuals: return "(y-x)"
		case .trainingCompleted: return "t"
		case .lossFunctionDelimiter: fatalError("Non-input token.")
		}
	}
}

// MARK: - Metrics and Results

enum Metric: String, Codable {
	case accuracy
	case mse
	case error
	case loss
	/// F1 score.
	case f1
	/// Matthews correlation coefficient.
	case mcc
	/// MNLI matched accuracy / mismatched accuracy.
	case mnli_mm_acc
	/// Spearman's rank correlation coefficient.
	case spearman
	/// Pearson correlation.
	case pearson
	/// Correlation.
	case correlation
	/// A proportion of exact matches; e.g., in a question answering environment.
	case exactMatch
	
	var maximize: Bool {
		switch self {
		case .accuracy: return true
		case .mse: return false
		case .error: return false
		case .loss: return false
		case .f1: return true
		case .mcc: return true
		case .mnli_mm_acc: return true
		case .spearman: return true
		case .pearson: return true
		case .correlation: return true
		case .exactMatch: return true
		}
	}
}

enum DatasetPartition: String, Codable {
	case training
	case validation
	case testing
}

struct EvaluationMetric: Codable, Hashable {
	let metric: Metric
	var dataset: DatasetPartition
}

struct GenerationResult: Codable {
    let timestamp: Date
    let generation: Int
    let populationAverageMetrics: [EvaluationMetric: Double]
}

typealias TrainingResults = [EvaluationMetric: Double]


protocol NamedColumnable {
	var columnName: String { get }
}

extension EvaluationMetric: NamedColumnable {
	var columnName: String { return "\(dataset.rawValue.capitalized) \(metric.rawValue.capitalized)" }
}

extension String: NamedColumnable {
	var columnName: String { return self }
}
