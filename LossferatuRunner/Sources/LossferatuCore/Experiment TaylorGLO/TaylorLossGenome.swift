//
//  TaylorLossGenome.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/26/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import SwiftGenetics

struct TaylorLossGenome: Genome, Codable {
	// BEGIN USELESS CRAP.
	func mutate(rate: Double, environment: TaylorLossGenome.Environment) { }
	func crossover(with partner: TaylorLossGenome, rate: Double, environment: TaylorLossGenome.Environment) -> (TaylorLossGenome, TaylorLossGenome) { return (self, partner) }
	typealias Environment = LivingStringEnvironment
	// END USELESS CRAP.
	
	let taylors: [TaylorExpansion]
	let activation: TaylorExpansion?
	let lr: Double
	let lrDecayPoints: [Double]?
}
