//
//  ProbabilityDistributions.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 4/12/20.
//	Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation

func tDistributionPDF(nu: Double, t: Double) -> Double {
	let coeff = tgamma((nu + 1.0) / 2.0) / (sqrt(nu * Double.pi) * tgamma(nu / 2.0))
	let power = -(nu + 1.0) / 2.0
	return coeff * pow(1.0 + t*t/nu, power)
}

func tDistributionCDF(nu: Double, t: Double) -> Double {
	let command = "echo \"from scipy import stats; print(stats.t.sf(abs(\(t)), \(nu)))\" | python"
	let zshOutput = command.runAsZshCommandLine()
	guard let output = Double(zshOutput.components(separatedBy: "\n").first!) else {
		LogHelper.shared.printWithTimestamp(type: .error, message: "UNABLE TO RUN PYTHON WITH SCIPY")
		print(zshOutput)
		exit(1)
	}
	return output
	
//	let x = (t + sqrt(t * t + nu)) / (2.0 * sqrt(t * t + nu))
//    return incbeta(nu/2.0, nu/2.0, x)
//
//	let x = nu / (t * t + nu)
//	return 1.0 - 0.5 * incbeta(nu/2.0, 1.0/2.0, x)
}

/// The CDF of the normal distribution, also known as `Φ(x)`.
func normalCDF(x: Double) -> Double {
	return 0.5 * (1.0 + erf(x / sqrt(2.0)))
}

/// The quantile function of the normal distribution.
func normalCDFInverse(p: Double) -> Double {
	fatalError("Not implemented.")
}
