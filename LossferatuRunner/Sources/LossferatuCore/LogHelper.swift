//
//  LogHelper.swift
//  Lossferatucore
//
//  Created by Santiago Gonzalez on 12/23/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

public struct LogHelper {
	
	static public let shared = LogHelper()
	
	let logMutex = DispatchSemaphore(value: 1)
	
	public enum PrintMessageType: String {
		case info = "✴️ "
		case error = "🆘"
		case completion = "✅"
		case indeterminate = "🔄"
		case checking = "🛂"
	}
	
	public func printWithTimestamp(type: PrintMessageType, message: String) {
		logMutex.wait()
		let formatter = DateFormatter()
		formatter.dateStyle = .none
		formatter.timeStyle = .medium
		let date = formatter.string(from: Date())
		print("\(date): \(type.rawValue) \(message)")
		logMutex.signal()
	}
}
