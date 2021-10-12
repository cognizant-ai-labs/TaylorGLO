//
//  Taylor+Monotonicity.swift
//  LossferatuRunner
//
//  Created by Santiago Gonzalez on 8/16/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

/// Represents the different kinds of monotonic behavior.
enum Monotonicity {
	/// Neither monotonically increasing nor decreasing. A curvy boi.
	case none
	/// Monotonically increasing.
	case increasing
	/// Monotonically decreasing.
	case decreasing
}

extension TaylorExpansion {
	
	/// Provides an estimate for the function's monotonicity within a given
	/// range, with respect to a given variable, at the specified granularity.
	/// Uses finite differences to estimate derivatives.
	func monotonicityEstimate(inRange range: ClosedRange<Double>, variableIndex: Int, atPoint point: [Double], evalPoints: Int) -> Monotonicity {
		precondition(variableIndex < variables && variableIndex >= 0)
		precondition(evalPoints >= 1, "Expected evalPoints to be >= 1, received \(evalPoints).")
		precondition(point.count == variables, "Expected \(variables), got \(point.count)")
		
		var monotonicitySoFar: Monotonicity?
		
		for i in 0..<evalPoints {
			// Figure the next evenly-distributed location in the range.
			let locationStart = range.lowerBound + Double(i) * (range.upperBound - range.lowerBound) / Double(evalPoints + 1)
			let locationEnd = range.lowerBound + Double(i+1) * (range.upperBound - range.lowerBound) / Double(evalPoints + 1)
			
			// TODO: Actually evaluate the value of the derivative instead of using FDM.
			
			var truePointStart = point
			var truePointEnd = point
			truePointStart[variableIndex] = locationStart
			truePointEnd[variableIndex] = locationEnd
			let fd = evaluated(at: truePointEnd) - evaluated(at: truePointStart)
			
			let localMonotonicity: Monotonicity = {
				if fd > 0 {
					return .increasing
				} else if fd < 0 {
					return .decreasing
				} else {
					return .none
				}
			}()
			
			if monotonicitySoFar == nil {
				monotonicitySoFar = localMonotonicity
			} else {
				monotonicitySoFar = localMonotonicity == monotonicitySoFar ? monotonicitySoFar : Monotonicity.none
			}
			guard monotonicitySoFar != Monotonicity.none else { return .none }
		}
		return monotonicitySoFar ?? .none
	}
	
}
