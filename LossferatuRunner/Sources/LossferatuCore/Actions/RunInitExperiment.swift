//
//  RunInitExperiment.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct RunInitExperiment {
	static func run(experimentDir: String, configFile: String) throws {
		// Create experiment directory.
		try ResultsManager.createExperimentDirectory(experimentDir, configFile: configFile)
		LogHelper.shared.printWithTimestamp(type: .completion, message: "Successfully initialized new experiment directory: \(experimentDir)")
		
		let experiment = try ResultsManager(experimentDirectory: experimentDir)
		guard experiment.config.kind != .oneShot else {
			return // Don't need any initial setup for one-shot.
		}
		
		// Run experiment-specific initialization.
		guard let experimentConfig = experiment.config.experiment else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Expected experiment config in config file.")
			exit(1)
		}
		switch experimentConfig.kind {
		case .gloTaylorClassification:
			try TaylorGLOInitializer.run(experiment: experiment)
		case .gloTreeClassification:
			fatalError("NOT IMPLEMENTED!")
		}
	}
}
