//
//  Functions.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 4/28/20.
//	Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation

func logisticFunction(x: Double, k: Double = 1.0) -> Double {
	return 1.0 / (1.0 + exp(-k * x))
}

/// The inverse of the logistic function for k = 1.0
func logitFunction(p: Double) -> Double {
	return -log((1.0 / p) - 1.0)
}
