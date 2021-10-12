//
//  CommandLineTool.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import LossferatuCore

let tool = CommandLineTool()

do {
    try tool.run()
} catch {
	LogHelper.shared.printWithTimestamp(type: .error, message: "Error: \(error)")
}
