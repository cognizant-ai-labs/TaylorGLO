//
//  SavedCandidate.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 1/1/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation
import SwiftCMA

/// A generational candidate model.
struct SavedCandidate: Codable {
	/// Tags that can be added to a candidate.
	enum Remark: String, Codable, CaseIterable {
		case jobFailed
		case violatedInvariant
	}
	
	/// The job's identifier, if it has been dispatched.
	var jobName: String?
	/// The candidate's fitness, if it has been evaluated.
	var fitness: Double?
	
	/// The candidate's generation of origin.
	var generation: Int
	
	/// Any remarks that are tagged on.
	var remarks: Set<Remark>
	
	/// The full, encoded candidate.
	var encodedCandidate: Vector?
	
	static let lossFunctionTaylorDelimiter = -666.0 // NOTE: this is really gross, but we can't use NaN because Codable will throw a fit.
	var lossFunctionTaylor: Vector?
	var lossFunctionTensorFlow: String?
	var lossFunctionMathematica: String?
	
	var activationFunctionTaylor: Vector?
	var activationFunctionTensorFlow: String?
	var activationFunctionMathematica: String?
	
	var results: TrainingResults?
	
	init(
		jobName: String? = nil,
		jobSucceeded: Bool? = nil,
		fitness: Double? = nil,
		generation: Int,
		remarks: Set<Remark> = Set<Remark>(),
		encodedCandidate: Vector? = nil,
		lossFunctionTaylor: Vector? = nil,
		lossFunctionTensorFlow: String? = nil,
		lossFunctionMathematica: String? = nil,
		activationFunctionTaylor: Vector? = nil,
		activationFunctionTensorFlow: String? = nil,
		activationFunctionMathematica: String? = nil,
		results: TrainingResults? = nil
	) {
		self.jobName = jobName
		self.fitness = fitness
		self.generation = generation
		self.remarks = remarks
		self.encodedCandidate = encodedCandidate
		self.lossFunctionTaylor = lossFunctionTaylor
		self.lossFunctionTensorFlow = lossFunctionTensorFlow
		self.lossFunctionMathematica = lossFunctionMathematica
		self.activationFunctionTaylor = activationFunctionTaylor
		self.activationFunctionTensorFlow = activationFunctionTensorFlow
		self.activationFunctionMathematica = activationFunctionMathematica
		self.results = results
	}
	
}
