//
//  RunReadStudioLog.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/24/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct RunReadStudioLog {
	/// Prints the studio log for the job with the given name.
	static func run(jobName: String, parse: Bool) throws {
		// Ensure reachability.
		TrainingInterface().assertAvailable()
		
		// Print the log.
		let log = TrainingInterface().fetchJobLog(jobFile: jobName)
		print(log)
		
		// Parse the log if needed.
		guard parse else { return }
		print("\n\n")
		LogHelper.shared.printWithTimestamp(type: .checking, message: "Parsing log...")
		/*let output = log.components(separatedBy: "FINISHED_RUNNING_SYSTEM_COMMAND")[0]
		let parseResults = InterfaceHelpers.processOutput(output, config: XXXX, evaluationPartition: .validation) // validation just for shits and gigles.
		debugPrint(parseResults) */
		fatalError("Parsing needs to be looked at further at a future data. Since the process output function now is dependent on a specific evaluator.")
		
	}
}
