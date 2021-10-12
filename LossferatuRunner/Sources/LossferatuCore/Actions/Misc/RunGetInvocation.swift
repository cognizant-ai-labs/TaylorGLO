//
//  RunGetInvocation.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 3/5/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import Files

struct RunGetInvocation {
	/// Prints the training invocation for a job with the given config.
	static func run(configFile: String) throws {
		// Read config.
		let configFile = try File(path: configFile)
		let config = try JSONDecoder().decode(ExperimentConfig.self, from: try configFile.read())
		
		// Get invocation.
		let evalPartition = config.evaluation.evalPartition
		let invocation = TrainingInterface().trainingJobInvocation(config: config.evaluation.baseTrainingConfig, evaluationPartition: evalPartition)
		print(invocation)
	}
}
