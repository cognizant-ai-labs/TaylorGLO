//
//  RunTest.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 3/25/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct RunTest {
	static func run() throws {
		
		LogHelper.shared.printWithTimestamp(type: .indeterminate, message: "Printing Taylor expansions...")
		
		for order in 3...5 {
			let possibleInputs: [[LossFunctionInput]] = [
				[.residuals],
				[.trueLabel, .scaledLogits],
				[.trueLabel, .scaledLogits, .unscaledLogits],
				[.trueLabel, .scaledLogits, .residuals],
				[.trueLabel, .scaledLogits, .trainingCompleted]
			]
			for inputVariables in possibleInputs {
			
				// Print an informative message
				LogHelper.shared.printWithTimestamp(type: .info, message: "Taylor k=\(order), variables=[\(inputVariables.map { $0.tensorflowName }.joined(separator: ", "))]")
				
				// Solution variables.
				let variables = inputVariables.count
				let paramSetType = TaylorExpansion.ParameterSetType.missingZeroPartialTerms(forVarIndices: [inputVariables.firstIndex(of: .scaledLogits), inputVariables.firstIndex(of: .unscaledLogits), inputVariables.firstIndex(of: .residuals)].compactMap { $0 }) // we don't care about terms where d/dy = 0
				let taylorSize = TaylorExpansion.parameterCount(forVariables: variables, order: order, parameterSetType: paramSetType)
				LogHelper.shared.printWithTimestamp(type: .completion, message: "Parameters: \(taylorSize) + \(variables)")
				
				// Build the Taylor expansion.
				let taylor = TaylorExpansion(variables: variables, order: order, center: Array(repeating: 1, count: variables), parameters: Array(repeating: 1, count: taylorSize), parametersIncludeFactorial: true, parameterSetType: paramSetType)
				
				// Print it!
				LogHelper.shared.printWithTimestamp(type: .completion, message: taylor.string(style: .tensorflow, varNames: inputVariables.map { $0.tensorflowName }, decimals: 0))
				
			}
		}
	}
}
