//
//  Taylor+StringRepresentation.swift
//  LossferatuRunner
//
//  Created by Santiago Gonzalez on 8/16/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

extension TaylorExpansion {
	/// Returns a string representation for the Taylor expansion.
	public func string(style: ExpressionStringStyle, varNames: [String], decimals: Int = 4) -> String {
		precondition(varNames.count == variables, "Expected \(variables), got \(varNames.count)")
		let format: (Double) -> String = { num in String(format: "%.\(decimals)f", num) }
		
		let terms: [String?] = (0..<TaylorExpansion.parameterCount(forVariables: variables, order: order, parameterSetType: .all)).map { i in
			let multiIndex = TaylorExpansion.partialMultiIndex(forParameter: i, variables: variables, order: order)
			let multiIndexDegree = multiIndex.reduce(0, { $0 + $1 })
			let denominator = multiIndexDegree.factorial
			
			let coefficient = !parametersIncludeFactorial ? parameters[i] / Double(denominator) : parameters[i]
			
			guard coefficient != 0.0 else { return nil }
			
			let powChunkStrings: [String] = multiIndex.enumerated().map { i, subindex in
				let varChunk: String = {
					if center[i] == 0.0 {
						return "\(varNames[i])"
					} else {
						return "(\(varNames[i])-\(format(center[i])))"
					}
				}()
				if subindex == 0 {
					return nil
				} else if subindex == 1 {
					return varChunk
				} else {
					return style.pow(x: varChunk, y: "\(subindex)")
				}
			}.compactMap { $0 }
			let powChunk = powChunkStrings.joined(separator: "*")
			
			if powChunk.isEmpty {
				return format(coefficient)
			} else if coefficient == 1.0 {
				return powChunk
			} else {
				return style.scalarMul(x: powChunk, scalar: format(coefficient))
			}
		}
		
		// Join everything together.
		return terms.compactMap { $0 }.joined(separator: "+").replacingOccurrences(of: "--", with: "+")
	}
}
