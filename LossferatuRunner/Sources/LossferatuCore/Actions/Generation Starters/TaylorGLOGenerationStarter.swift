//
//  TaylorGLOGenerationStarter.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/26/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation
import SwiftGenetics
import SwiftCMA

let VIOLATED_INVARIANT_FITNESS_VALUE = 0.0

struct TaylorGLOGenerationStarter {
	static func run(experiment: ResultsManager) throws {
		// Load checkpoint.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Reading checkpoint...")
		let folder = try experiment.currentGenerationFolder()
		let cmaes = try CMAES.from(checkpoint: folder.url.appendingPathComponent(ResultsManager.FileName.generationCheckpoint.rawValue))
		let taylorGloConfig = experiment.config.experiment!.taylorGloConfig!
		let generation = Int(folder.name)!
		
		// Hyperparameters.
		let order = taylorGloConfig.order
		let inputVariables = taylorGloConfig.inputVariables ?? TaylorGloConfig.defaultInputVariables
		let variables = inputVariables.split(separator: .lossFunctionDelimiter).map { $0.count }
		
		// Solution variables.
		let paramSetTypes = taylorGloConfig.taylorParameterSetTypes
		let taylorSizes = zip(paramSetTypes, variables).map { TaylorExpansion.parameterCount(forVariables: $0.1, order: order, parameterSetType: $0.0) }
		let totalTaylorSize = taylorSizes.reduce(0, { $0 + $1 })
		
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Starting generation \(generation).")
	
		// Get candidates.
		let candidates = cmaes.startEpoch()
		var intragenerationalCandidates: [SavedCandidate] = candidates.map { candidate in
			let taylors = TaylorGLOEncoding.taylorParameterLists(candidate, config: taylorGloConfig)
			let activationTaylor = TaylorGLOEncoding.taylorActivationParameterList(candidate, config: taylorGloConfig)
			return SavedCandidate(
				generation: generation,
				encodedCandidate: candidate,
				lossFunctionTaylor: taylors.joined(separator: [SavedCandidate.lossFunctionTaylorDelimiter]).compactMap { $0 },
				activationFunctionTaylor: activationTaylor
			)
		}
		let intragenerationalCandidatesMutex = DispatchSemaphore(value: 1)
		
		// Begin evaluation.
		let experimentMutex = DispatchSemaphore(value: 1)
		let dispatchGroup = DispatchGroup()
		for (i, candidate) in candidates.enumerated() {
			dispatchGroup.enter()
			DispatchQueue.global().sync {
				defer {
					dispatchGroup.leave() // We have completed the work for this candidate.
				}
				do {
					var jobName: String? = nil
					var attemptsRemaining = experiment.config.evaluation.maximumJobSubmissionRetries ?? 3
					while jobName == nil && attemptsRemaining > 0 {
						attemptsRemaining -= 1
						
						// Create organism from candidate.
						let parameterLists = TaylorGLOEncoding.taylorParameterLists(candidate, config: taylorGloConfig)
						let organism: Organism<TaylorLossGenome> = {
							let taylors = paramSetTypes.enumerated().map { idx, paramSetType in
								TaylorExpansion(
									variables: variables[idx],
									order: order,
									center: parameterLists[idx].suffix(variables[idx]),
									parameters: Array(parameterLists[idx].prefix(parameterLists[idx].count - variables[idx])), parametersIncludeFactorial: true, parameterSetType: paramSetType)
							}
							let activationTaylor: TaylorExpansion? = TaylorGLOEncoding.taylorActivationParameterList(candidate, config: taylorGloConfig).flatMap {
								let variables = 1
								return TaylorExpansion(
									variables: variables,
									order: taylorGloConfig.evolvedActivationFunction!.order,
									center: $0.suffix(variables),
									parameters: Array($0.prefix($0.count - variables)),
									parametersIncludeFactorial: true
								)
							}
							let lr = 0.0 // TODO: handle LR correctly
							return Organism(fitness: nil, genotype: TaylorLossGenome(taylors: taylors, activation: activationTaylor, lr: lr, lrDecayPoints: taylorGloConfig.evolvedLearningRateDecayPoints.flatMap { candidate.suffix($0) }), birthGeneration: generation)
						}()
						
						// Create strings.
						let tfString = organism.genotype.taylors.enumerated().map { $0.1.string(style: .tensorflow, varNames: inputVariables.split(separator: .lossFunctionDelimiter)[$0.0].map { $0.tensorflowName }) }.joined(separator: "===")
						let mathematicaString = organism.genotype.taylors.enumerated().map { $0.1.string(style: .mathematica, varNames: inputVariables.split(separator: .lossFunctionDelimiter)[$0.0].map { $0.mathematicaName }) }.joined(separator: "===")
						let argString = "[" + candidate.map { String(format: "%.4f", $0) }.joined(separator: ", ") + "]"
						
						intragenerationalCandidatesMutex.wait()
						intragenerationalCandidates[i].lossFunctionTensorFlow = tfString
						intragenerationalCandidates[i].lossFunctionMathematica = mathematicaString
						intragenerationalCandidatesMutex.signal()
						
						// Create activation strings.
						if let activationTaylor = organism.genotype.activation {
							let activationTfString = activationTaylor.string(style: .tensorflow, varNames: ["x"])
							let activationMathematicaString = activationTaylor.string(style: .mathematica, varNames: ["x"])
							intragenerationalCandidatesMutex.wait()
							intragenerationalCandidates[i].activationFunctionTensorFlow = activationTfString
							intragenerationalCandidates[i].activationFunctionMathematica = activationMathematicaString
							intragenerationalCandidatesMutex.signal()
						}
						
						// Verify loss-function invariant.
						if let n = experiment.config.evaluation.baseTrainingConfig.target.numClasses, variables == [2] && organism.genotype.taylors.first!.order == 3 {
							let p = parameterLists.first!
							let (p0, p1, p2, p3, p4, p5, p6, p7) = (p[6], p[7], p[0], p[1], p[2], p[3], p[4], p[5])
							
							let c1: Double = -2.0 * p1 * p3 + 2.0 * p0 * p1 * p6 + p2 - p5 * p0 + p7 * p0 * p0 + 3.0 * p4 * p1 * p1
							let ch: Double = 2.0 * p3 - 2.0 * p6 * p0 - 6.0 * p4 * p1
							let chh: Double = 3.0 * p4
							let chy: Double = 2.0 * p6
							let cy: Double = -2.0 * p1 * p6 + p5 - 2.0 * p7 * p0
							let cyy: Double = p7
							
							let leftSide: Double = cy + cyy + chy / Double(n)
							let rightSide: Double = Double(Double(n) - 2.0) * Double(c1 + ch/Double(n) + chh/Double(n*n))
							
							
							if leftSide < rightSide {
								// We have violated the invariant.
								intragenerationalCandidatesMutex.wait()
								intragenerationalCandidates[i].remarks.insert(.violatedInvariant)
								if taylorGloConfig.ignoreInvariantCheck ?? false {
									// Continue.
									intragenerationalCandidatesMutex.signal()
								} else {
									// Abort.
									intragenerationalCandidates[i].fitness = VIOLATED_INVARIANT_FITNESS_VALUE
									intragenerationalCandidatesMutex.signal()
									LogHelper.shared.printWithTimestamp(type: .info, message: "Violated invariant: \(argString)")
									return
								}
							}
						}
						/*if variables == [2] {
							let monotonicity = organism.genotype.taylors.first!.monotonicityEstimate(inRange: 0.0...1.0, variableIndex: 1, atPoint: [1.0, 666.666], evalPoints: 50) // NOTE: this is only checked on the first loss function.
							if monotonicity == .decreasing {
								// We have violated the invariant.
//								LogHelper.shared.printWithTimestamp(type: .completion, message: "Violated invariant: \(argString)")
								intragenerationalCandidatesMutex.wait()
								intragenerationalCandidates[i].remarks.insert(.violatedInvariant)
								if taylorGloConfig.ignoreInvariantCheck ?? false {
									// Continue.
									intragenerationalCandidatesMutex.signal()
								} else {
									// Abort.
									intragenerationalCandidates[i].fitness = VIOLATED_INVARIANT_FITNESS_VALUE
									intragenerationalCandidatesMutex.signal()
									return
								}
							}
						}*/
						
						// Train the model.
						LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Submitting job for: \(argString)")
						var modelConfig = experiment.config.evaluation.baseTrainingConfig // Copy the base training config.
						modelConfig.lossFunction = tfString // Set the loss function.
						if let activationTaylor = organism.genotype.activation { // Set the activation function if it exists.
							let activationTfString = activationTaylor.string(style: .tensorflow, varNames: ["x"])
							modelConfig.overrideActivationFunction = activationTfString
						}
						if let lr = TaylorGLOEncoding.learningRate(candidate, config: taylorGloConfig) { // Set the initial learning rate if it exists.
							modelConfig.learningRateSchedule.initial = lr
						}
						if let epochs = TaylorGLOEncoding.learningRateDecayEpochs(candidate, config: taylorGloConfig, trainingConfig: experiment.config.evaluation.baseTrainingConfig) { // Set the learning rate decay 		schedule if it exists.
							modelConfig.learningRateSchedule.decayEpochs = epochs
						}
						assert(Set(experiment.config.evaluation.evaluatedMetrics.map { $0.dataset }).count == 1) // TODO: handle multi-dataset metrics better.
						let evalPartition = experiment.config.evaluation.evalPartition
						jobName = TrainingInterface(/*experimentForOptionalLogging: experiment*/).submitTrainingJob(config: modelConfig, experimentName: experiment.config.name, evaluationPartition: evalPartition, failureOutput: { failureOutput in
							experimentMutex.wait()
							try! experiment.generationErrorLog(named: "studio_run_candidate_\(i)").write(failureOutput)
							experimentMutex.signal()
						})
						
						// Print an error message for failed submissions.
						if jobName == nil {
							LogHelper.shared.printWithTimestamp(type: .error, message: "Job submission failed for: \(argString) (remaining attempts: \(attemptsRemaining))")
							// Sleep a random amount to prevent overloading studio if there are too many failures.
							sleep(UInt32.random(in: 1...5))
						}
					}

					intragenerationalCandidatesMutex.wait()
					intragenerationalCandidates[i].jobName = jobName ?? "???"
					intragenerationalCandidatesMutex.signal()
					
					LogHelper.shared.printWithTimestamp(type: .completion, message: "Submitted job: \(jobName ?? "???")")
					experimentMutex.wait()
					try experiment.add(jobName: jobName ?? "???")
					experimentMutex.signal()
				
				} catch {
					LogHelper.shared.printWithTimestamp(type: .error, message: "Error in candidate block: \(error)")
				}
			}
		}
		dispatchGroup.wait()
		
		// Save candidates.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Saving candidates...")
		try experiment.saveGenerationCandidates(intragenerationalCandidates)
		
		// Save updated CMA-ES.
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Saving checkpoint...")
		try cmaes.save(checkpoint: folder.url.appendingPathComponent(ResultsManager.FileName.generationCheckpoint.rawValue))
		
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Finished starting generation \(generation).")
	}
}
