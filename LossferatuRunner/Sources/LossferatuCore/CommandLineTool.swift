//
//  CommandLineTool.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

public final class CommandLineTool {
    private let arguments: [String]

    public init(arguments: [String] = CommandLine.arguments) {
        self.arguments = arguments
    }

	/// The CLI's entrypoint.
    public func run() throws {
		guard arguments.count > 1 else {
			return printHelp()
		}
		
		guard let actionFlag = ActionFlag(rawValue: arguments[1]) else {
			throw Error.unexpectedActionFlag
		}
		
		switch actionFlag {
		case .help:
			printHelp()
		
		case .initExperiment:
			let configFile = try getConfigFile()
			let experimentDir = try getExperimentDir()
			try RunInitExperiment.run(experimentDir: experimentDir, configFile: configFile)
		
		case .startGeneration:
			let experimentDir = try getExperimentDir()
			try RunStartGeneration.run(experimentDir: experimentDir)
			
		case .checkGeneration:
			let experimentDir = try getExperimentDir()
			try RunCheckGeneration.run(experimentDir: experimentDir)
			
		case .analyzeExperiment:
			let experimentDir = try getExperimentDir()
			try RunAnalyzeExperiment.run(experimentDir: experimentDir)
			
		case .collateOneShots:
			let experimentsDir = try getExperimentDir()
			try RunCollateOneShots.run(experimentsDir: experimentsDir)
			
		case .resummarize:
			let experimentsDirsDir = try getExperimentDir()
			try RunResummarize.run(experimentsDirsDir: experimentsDirsDir)
			
		case .resummarizeGenerational:
			let experimentsDir = try getExperimentDir()
			try RunResummarizeGenerational.run(experimentsDir: experimentsDir)
		
		case .tTest:
			let experimentsDir1 = try getExperimentDir(argPosition: 0)
			let experimentsDir2 = try getExperimentDir(argPosition: 1)
			try RunTTest.run(experiments: [experimentsDir1, experimentsDir2])
			
		case .getInvocation:
			let configFile = try getConfigFile(argPosition: 0)
			try RunGetInvocation.run(configFile: configFile)
		
		case .readStudioLog:
			let jobName = try getJobName()
			try RunReadStudioLog.run(jobName: jobName, parse: shouldParse())
			
		case .test:
			try RunTest.run()
			
		}
	}
	
	/// Returns the config file from the CLI.
	func getConfigFile(argPosition: Int = 1) throws -> String {
		guard arguments.count > 2 + argPosition else {
			throw Error.incorrectNumberOfArguments
		}
		let filename = arguments[2 + argPosition]
		guard filename.hasSuffix(".json") else {
			throw Error.expectedJsonFile
		}
		return filename
	}
	
	/// Returns the experiment directory from the CLI.
	func getExperimentDir(argPosition: Int = 0) throws -> String {
		guard arguments.count > 2 + argPosition else {
			throw Error.incorrectNumberOfArguments
		}
		return arguments[2 + argPosition]
	}
	
	/// Returns the job name from the CLI.
	func getJobName() throws -> String {
		guard arguments.count > 2 else {
			throw Error.incorrectNumberOfArguments
		}
		return arguments[2]
	}
	
	/// Whether the studio log should be parsed.
	func shouldParse() -> Bool {
		return arguments.contains("parse")
	}
	
	/// Prints a standard help message.
	func printHelp() {
		print(
"""
USAGE:
$ LossferatuRunner help
- Running experiments:
$ LossferatuRunner init EXPERIMENT_DIR CONFIG.json
$ LossferatuRunner start EXPERIMENT_DIR
$ LossferatuRunner check EXPERIMENT_DIR
- Postprocessing results:
$ LossferatuRunner analyze EXPERIMENT_DIR
$ LossferatuRunner collateoneshots EXPERIMENTS_DIR
$ LossferatuRunner resummarize EXPERIMENTS_DIRS_DIR
$ LossferatuRunner resummarizegenerational EXPERIMENTS_DIR
$ LossferatuRunner ttest EXPERIMENTS_DIR_1 EXPERIMENTS_DIR_2
- Miscellaneous:
$ LossferatuRunner getinvocation CONFIG.json
$ LossferatuRunner studiolog JOB_NAME (parse)
$ LossferatuRunner test
"""
		)
	}
}

public extension CommandLineTool {
	enum ActionFlag: String {
		case initExperiment = "init"
		case startGeneration = "start"
		case checkGeneration = "check"
		case analyzeExperiment = "analyze"
		case collateOneShots = "collateoneshots"
		case resummarize = "resummarize"
		case resummarizeGenerational = "resummarizegenerational"
		case tTest = "ttest"
		case help = "help"
		case getInvocation = "getinvocation"
		case readStudioLog = "studiolog"
		case test = "test"
	}
	
	enum ExitCode: Int32 {
		case success = 0
		case incompleteJob = 1
		case interfaceUnavailable = 2
		case missingJobs = 10
	}
	
	enum Error: Swift.Error {
		case incorrectNumberOfArguments
		case missingActionFlag
		case unexpectedActionFlag
		case expectedJsonFile
	}
}
