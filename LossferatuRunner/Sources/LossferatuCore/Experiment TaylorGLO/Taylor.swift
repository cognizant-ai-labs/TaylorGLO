//
//  Taylor.swift
//  LossferatuRunner
//
//  Created by Santiago Gonzalez on 8/13/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

extension Int {
	/// Calculates and returns the integer's factorial.
	/// - Note: This implementation is not well suited for large numbers.
	var factorial: Int {
		guard self >= 1 else { return 1}
		return (1...self).reduce(1, { $0 * $1 })
	}
}

/// Defines the different styles that a string representation of an expression
/// can take.
public enum ExpressionStringStyle {
	/// A string suited for Wolfram Mathematica.
	case mathematica
	/// A string suited for TensorFlow.
	case tensorflow
	
	/// Returns the specific string for an exponetiation operation.
	func pow(x: String, y: String) -> String {
		switch self {
		case .mathematica: return "(\(x))^\(y)"
		case .tensorflow: return "tf.pow(\(x),\(y))"
		}
	}
	
	/// Returns the specific string for a scalar multiplication operation.
	func scalarMul(x: String, scalar: String) -> String {
		let scalarInside = scalar.first == "(" && scalar.last == ")" ? String(scalar.prefix(scalar.count - 1).suffix(scalar.count - 2)) : scalar
		switch self {
		case .mathematica: return "(\(scalarInside)*\(x))"
		case .tensorflow: return "(\(scalarInside)*\(x))" //"torch.mul(\(x),\(scalarInside))" //"tf.scalar_mul(\(scalarInside),\(x))"
		}
	}
}

/// A specific Taylor series expansion of a function.
public struct TaylorExpansion: Codable {
	
	/// Defines types of parameter sets.
	public enum ParameterSetType {
		/// A standard Taylor expansion with all terms.
		case all
		/// A Taylor expansion that excludes terms whose partial derivative with
		/// respect to the variables with the given indices is zero.
		case missingZeroPartialTerms(forVarIndices: [Int])
	}
	
	/// The number of variables that the approximated function takes.
	public let variables: Int
	/// The order of the Taylor expansion.
	public let order: Int
	/// The point around which the approximation is made (0 for Maclaurin series).
	public let center: [Double]
	/// A flat list of **every** parameter in the Taylor expansion.
	public let parameters: [Double]
	/// Whether the factorial dividends are already included in the parameters.
	public let parametersIncludeFactorial: Bool
	
	/// Creates a new Taylor expansion.
	public init(variables: Int, order: Int, center: [Double], parameters: [Double], parametersIncludeFactorial: Bool, parameterSetType: ParameterSetType = .all) {
		
		// Precondition checks.
		precondition(variables >= 1)
		precondition(order >= 0)
		precondition(center.count == variables)
		let expectedParams: Int = TaylorExpansion.parameterCount(forVariables: variables, order: order, parameterSetType: parameterSetType)
		precondition(parameters.count == expectedParams, "Expected \(expectedParams) params, got \(parameters.count)")
		
		self.variables = variables
		self.order = order
		self.center = center
		self.parametersIncludeFactorial = parametersIncludeFactorial
		
		// Populate the parameters.
		let expectedFullParams = TaylorExpansion.parameterCount(forVariables: variables, order: order)
		switch parameterSetType {
		case .all:
			self.parameters = parameters
		case .missingZeroPartialTerms(forVarIndices: let zeroPartialVarIndices):
			/*
			
			func indexCondition(paramSet: [Double]) -> Bool {
				return !(zeroPartialVarIndices.map { zeroPartialVarIndex in
					let partial = TaylorExpansion.partialMultiIndex(forParameter: paramSet.count, variables: variables, order: order)[zeroPartialVarIndex]
					return partial == 0 // Return if the partial for this variable is zero.
				}.reduce(true, { $0 && $1 } ))
			}
			var fullParameterSet = [Double]()
			for knownParam in parameters {
				while fullParameterSet.count < expectedFullParams && indexCondition(paramSet: fullParameterSet) {
					fullParameterSet.append(0.0)
				}
				fullParameterSet.append(knownParam)
			}
			while fullParameterSet.count < expectedFullParams && indexCondition(paramSet: fullParameterSet) {
				fullParameterSet.append(0.0)
			}
			self.parameters = fullParameterSet
			*/
			func indexUnnecessaryCondition(paramSet: [Double]) -> Bool {
				return zeroPartialVarIndices.map { zeroPartialVarIndex in
					return 0 == TaylorExpansion.partialMultiIndex(forParameter: paramSet.count, variables: variables, order: order)[zeroPartialVarIndex]
				}.allSatisfy { $0 == true } //.reduce(false, { $0 || $1 } )
			}
			var fullParameterSet = [Double]()
			for knownParam in parameters {
				while fullParameterSet.count < expectedFullParams && indexUnnecessaryCondition(paramSet: fullParameterSet) {
					fullParameterSet.append(0.0)
				}
				fullParameterSet.append(knownParam)
			}
			if fullParameterSet.count != expectedFullParams && indexUnnecessaryCondition(paramSet: fullParameterSet) {
				fullParameterSet.append(0.0)
			}
			self.parameters = fullParameterSet
		}
		
		assert(self.parameters.count == expectedFullParams, "Expected \(expectedFullParams) params, got \(self.parameters.count)")
	}
	
	/// Returns the type of partial for a given parameter as a multi-index.
	/// That is, an array of the orders of partials for each variable. In the
	/// structure that we have, you can think of this multi-index as being a set
	/// of digits for a base-`(order+1)` number. These multi-index numbers are
	/// all consecutive and monotonically increasing in our representation.
	static func partialMultiIndex(forParameter parameterIndex: Int, variables: Int, order: Int) -> [Int] {
		// TODO: deloopify this horror
		var i = 0
		var subsetIndex = 0
		repeat {
			let candidate = convertToBaseN(number: i, places: variables, base: order + 1)
			let sum = candidate.reduce(0, { $0 + $1 })
			
			if sum <= order {
				subsetIndex += 1
			}
			
			if subsetIndex - 1 == parameterIndex {
				return candidate
			}
			
			i += 1
		} while Double(i) < pow(Double(order + 1), Double(variables))
		return [] // We should never reach this.
	}
	
	private static func convertToBaseN(number: Int, places: Int, base: Int) -> [Int] {
		var digits = [Int]()
		var divisor = number
		while divisor != 0 {
			digits.append(divisor % base)
			divisor /= base
		}
		for _ in 0..<(places - digits.count) {
			digits.append(0)
		}
		return digits.reversed()
	}
	
	/// Returns the expected number of parameters for a Taylor expansion of a
	/// given order with a given number of variables.
	static func parameterCount(forVariables variables: Int, order: Int, parameterSetType: ParameterSetType = .all) -> Int {
		switch parameterSetType {
		case .all:
			return (variables + order).factorial / (variables.factorial * order.factorial)
			
		case .missingZeroPartialTerms(forVarIndices: let zeroPartialVarIndices):
			// TODO: deloopify this horror
			var i = 0
			var count = 0
			repeat {
				let candidate = convertToBaseN(number: i, places: variables, base: order + 1)
				let sum = candidate.reduce(0, { $0 + $1 })
				
				let indexNecessaryCondition = zeroPartialVarIndices.map { candidate[$0] != 0 }.contains(true)
				if indexNecessaryCondition && sum <= order {
					count += 1
				}
				
				i += 1
			} while Double(i) < pow(Double(order + 1), Double(variables))
			return count
		}
	}
}

extension TaylorExpansion {
	
	/// Returns the function's value at the specified point.
	func evaluated(at point: [Double]) -> Double {
		precondition(point.count == variables, "Expected \(variables), got \(point.count)")
		return (0..<TaylorExpansion.parameterCount(forVariables: variables, order: order, parameterSetType: .all)).map { i in
			let multiIndex = TaylorExpansion.partialMultiIndex(forParameter: i, variables: variables, order: order)
			let multiIndexDegree = multiIndex.reduce(0, { $0 + $1 })
			let denominator = multiIndexDegree.factorial
			
			let coefficient = !parametersIncludeFactorial ? parameters[i] / Double(denominator) : parameters[i]
			let powerInteriors = multiIndex.enumerated().map { pow(point[$0] - center[$0], Double($1)) }
			
			return coefficient * powerInteriors.reduce(1, { $0 * $1 })
		}.reduce(0, { $0 + $1 })
	}
	
}
