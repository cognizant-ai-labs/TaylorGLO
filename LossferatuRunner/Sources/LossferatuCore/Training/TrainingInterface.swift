//
//  TrainingInterface.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 11/20/18.
//  Copyright © 2018 Santiago Gonzalez. All rights reserved.
//

import Foundation

#if os(Linux)
let locationPrefix = "/home/username_here/"
#else
let locationPrefix = "/Users/username_here/"
#endif


enum JobState {
	case dispatched
	case running
	case finished
}

struct TrainingInterface {
	
	let experiment: ResultsManager?
	init(experimentForOptionalLogging experiment: ResultsManager? = nil) {
		self.experiment = experiment
	}
	
	/// Wraps `isAvailable()`.
	func assertAvailable() {
		guard self.isAvailable() else {
			LogHelper.shared.printWithTimestamp(type: .error, message: "Unable to access training interface. Check VPN.")
			exit(CommandLineTool.ExitCode.interfaceUnavailable.rawValue)
		}
	}
	
	/// Check whether the interface is available.
	func isAvailable() -> Bool {
		let pingOutput = "ping xxxxxxx_REPLACE_ME_xxxxxxx -c 1".runAsZshCommandLine()
		return pingOutput.contains("1 packets received") // macOS ping
			|| pingOutput.contains("1 received") // Ubuntu ping
	}
	
	func trainingJobInvocation(config: TrainingConfig, evaluationPartition: DatasetPartition) -> String {
		let args = InterfaceHelpers.args(config: config, evaluationPartition: evaluationPartition)
		let prefetchCommand = config.target.prefetchCommand
		let modelDownloadCommand: String? = config.baseModelS3Tar.flatMap { s3Tar in
			let tar = s3Tar.components(separatedBy: "|").first!
			return "export AWS_ACCESS_KEY_ID=research; export AWS_SECRET_ACCESS_KEY=**************************; date -u; aws s3 cp --endpoint-url=http://minio.somewhere.com:9000 \(tar) . && echo DOWNLOADED MODEL FROM MINIO && ls -l && echo XXXXXXXXXXXXXXXXXXXX && tar -xvf modeldir.tar -C . && echo XXXXXXXXXXXXXXXXXXXX && ls -l && echo LOAD MODEL END && date -u"
		}
		return "\(prefetchCommand.flatMap { "\($0); " } ?? "")\(modelDownloadCommand.flatMap { "\($0); " } ?? "")python3.7 \(args)"
	}
	
	func submitTrainingJob(config: TrainingConfig, experimentName: String, evaluationPartition: DatasetPartition, failureOutput: (String) -> Void) -> String? {
		let invocation = trainingJobInvocation(config: config, evaluationPartition: evaluationPartition)
		let queueName: String = {
			switch config.evaluator {
			case .fumanchu: return "rmq_Lossferatu_TrainingFumanchu"
			case .transformer: return "rmq_Lossferatu_TrainingTransformer"
			}
		}() //"rmq_Lossferatu_\(experimentName)"
		let maxJobDurationMins = config.target.maxJobDurationMins
		let jobName = invocation.runAsStudioML(from: locationPrefix + config.target.scriptLocation, queueName: queueName, maxJobDurationMins: maxJobDurationMins, gpuSetting: config.gpuSetting, trainingAttempts: config.trainingAttempts ?? 1, failureOutput: failureOutput, experiment: experiment)
		return jobName
	}
	
	func fetchJobLog(jobFile: String) -> String {
		let command = "\(locationPrefix)torch/bin/aws s3 cp --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/experiments/\(jobFile)/output.tar -" // 2> /dev/null
		let fullOutput = command.runAsZshCommandLine()
		return fullOutput
	}
	
	func checkJobModelTarExists(jobFile: String) -> Bool {
		 let command = "\(locationPrefix)torch/bin/aws s3 ls --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/experiments/\(jobFile)/modeldir.tar > /dev/null; echo $?"
		 let fullOutput = command.runAsZshCommandLine()
		return fullOutput.contains("0")
	}
	
	func checkJobResults(jobFile: String, config: TrainingConfig, evaluationPartition: DatasetPartition) -> (JobState, TrainingResults?) {
		let fullOutput = fetchJobLog(jobFile: jobFile)
		guard !fullOutput.contains("failed when downloading user data") else {
			fatalError("*** 🆘 Job failed when downloading user data: \(jobFile). Consider increasing the job's lifetime. ***") // Studio failed.
		}
		guard !fullOutput.contains("download failed: s3://YOURBUCKET-datasets/experiments") else {
			return (.dispatched, nil) // Log doesn't exist yet.
		}
/*		guard fullOutput.contains("+ result") else {
			return (false, nil) // Still running.
		}
		guard fullOutput.contains("+ result=0\n") else {
			return (true, nil) // We have a non-zero exit code.
		}*/
		guard fullOutput.contains("result=$?") else {
			return (.running, nil) // Still running.
		}
		guard !fullOutput.contains("+ result=1\n") else {
			return (.finished, nil) // We have a non-zero exit code.
		}
//		guard fullOutput.contains("FINISHED_RUNNING_SYSTEM_COMMAND") else {
//			return (true, nil) // We have a failure of some sort.
//		}
		let output = fullOutput.components(separatedBy: "FINISHED_RUNNING_SYSTEM_COMMAND")[0]
		
		return InterfaceHelpers.processOutput(output, config: config, evaluationPartition: evaluationPartition)
		
	}
}

enum StudioGPUSetting: Int {
	case halfGPU = 1
	case fullGPU = 2
	
	var memoryGB: Int {
		switch self {
		case .halfGPU: return 4
		case .fullGPU: return 8
		}
	}
}

extension String {
	
	/// Runs the string as a zsh command line and returns stdout.
	@discardableResult
	func runAsZshCommandLine() -> String {
		let task = Process()
		#if os(Linux)
		task.executableURL = URL(string: "/bin/zsh")
		#else
		task.launchPath = "/bin/zsh"
		#endif
		task.arguments = ["-l"]
		let input = Pipe()
		task.standardInput = input
		input.fileHandleForWriting.write(self.data(using: .utf8)!)
		let pipe = Pipe()
		task.standardOutput = pipe
		task.standardError = pipe
		#if os(Linux)
		try! task.run()
		#else
		task.launch()
		#endif
		input.fileHandleForWriting.closeFile()
		let data = pipe.fileHandleForReading.readDataToEndOfFile()
		let output = NSString(data: data, encoding: String.Encoding.utf8.rawValue)
		return output! as String
	}
	
	/// Submits a Studio ML job using the string. Returns the job ID.
	func runAsStudioML(from dir: String, queueName: String, maxJobDurationMins: Int, gpuSetting: StudioGPUSetting, trainingAttempts: Int, failureOutput: (String) -> Void, experiment: ResultsManager? = nil) -> String? {
		let id = UUID().uuidString
		let goodTrainingMarker = "'reached_unacceptable_accuracy_threshold': False"
		let unescapedCommand = trainingAttempts > 1 ? "for attempt in {1..\(trainingAttempts)} ; do \(self) | tee /dev/stderr | grep \"\(goodTrainingMarker)\" > /dev/null && break; done" : self
		let command = "NEW_LOSSGA_COMMAND=\"\(unescapedCommand.replacingOccurrences(of: "'", with: "\\'").replacingOccurrences(of: "\"", with: "\\\\\\\\\\\\\\\""))\""
		let pipTorch = "pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html"
		let fileCreation = "echo \"import os; os.system('\(pipTorch) || \(pipTorch)'); os.system('/bin/bash -c \\\"$NEW_LOSSGA_COMMAND\\\"'); print('', flush=True); print('', flush=True); print('', flush=True); print('FINISHED_RUNNING_SYSTEM_COMMAND', flush=True); print('', flush=True); print('', flush=True); print('', flush=True); import time; time.sleep(3)\" > _runme_\(id).py"
		let copyDir = "/tmp/AutoLossferatu_JOB_\(id)"
		let str = "source \(locationPrefix)/torch/bin/activate && cp -r \(dir) \(copyDir) && cd \(copyDir) && rm -rf .git && \(command); \(fileCreation); export AWS_ACCESS_KEY_ID=research; export AWS_SECRET_ACCESS_KEY=********************************; studio run --lifetime=\(maxJobDurationMins * 10)m --max-duration=\(maxJobDurationMins)m --gpus \(gpuSetting.rawValue) --gpuMem \(gpuSetting.memoryGB)GB --queue=\(queueName) --force-git _runme_\(id).py 2>&1 && echo SUCCESS; rm -rf \(copyDir)"
//		print(str)
//		exit(1)
		let zshout = str.runAsZshCommandLine()
		guard zshout.hasSuffix("SUCCESS\n") else {
			failureOutput(zshout)
			return nil
		}
		let jobName = /*zshout.components(separatedBy: "S3Provider - Added experiment ").last!.components(separatedBy: "\n").first!*/ /*zshout.components(separatedBy: "studio-runner - sending message {\"experiment\": {").last!.components(separatedBy: "\"key\": \"").last!.components(separatedBy: "\"").first!*/ zshout.components(separatedBy: "studio run: submitted experiment ").last!.components(separatedBy: "\n").first!
		if let experiment = experiment {
			try! experiment.generationLog(named: jobName).write("\(str)\n\n\n\n\(zshout)")
		}
		return jobName
	}
	
}

struct InterfaceHelpers {
	
	static func args(config: TrainingConfig, evaluationPartition: DatasetPartition) -> String {
		let valTestTrain: String = {
			switch evaluationPartition {
			case .training: return "train"
			case .testing: return "test"
			case .validation: return "val"
			}
		}()
		let modelDir = "../modeldir"
		let file = config.target.scriptFile
		
		switch config.evaluator {
		case .fumanchu:
			// FUMANCHU:
			return file + " " + {
				var str = "--dataset \(config.target.rawValue) --arch \(config.fumanchuModelArgument)"
				str += " --loss '\(config.lossFunction)'"
				str += config.overrideActivationFunction.flatMap { activationFunction in
					return " --activation-fn '0.0+\(activationFunction)'" // NOTE: the "0.0+" thing is a workaround for a stupid issue with argparse that breaks if the activation function starts with a "-".
				} ?? ""
				str += " --eval-dataset \(valTestTrain)"
				str += " --epochs \(config.epochs)"
				str += " --full-run-epochs \(config.fullTrainingEpochs)"
				str += " --dataset-percentage \(config.trainingDatasetPercentage)"
				str += " --checkpoint \(modelDir)"
				str += config.baseModelS3Tar.flatMap { s3Tar in
					let ckpt = s3Tar.components(separatedBy: "|").last!
					return " --resume \(ckpt)"
				} ?? ""
				str += config.startEpoch.flatMap { startEpoch in
					return " --start-epoch \(startEpoch)"
				} ?? ""
	//			str += " --evaluated-metrics \(config.)" /// TODO: Pass the evaluated metrics in once they're supported.
				//--drop
				str += " --train-batch \(config.trainingBatchSize)"
				str += " --test-batch \(config.evalBatchSize)"
				str += " --optimizer \(config.optimizer.rawValue)"
				//--momentum
				if let auxiliaryClassifiers = config.auxiliaryClassifiers {
					str += " --auxiliary-classifiers \(auxiliaryClassifiers)"
				}
				if let seed = config.seed {
					str += " --manual-seed \(seed)"
				}
				if let cutout = config.cutout {
					str += " --cutout --cutout-n-holes \(cutout.numberHoles) --cutout-length \(cutout.length)"
				}
				if let cutmix = config.cutmix {
					str += " --cutmix --cutmix-alpha \(cutmix.alpha) --cutmix-prob \(cutmix.probability)"
				}
				if let fgsm = config.fgsm {
					str += " --fgsm --fgsm-epsilon \(fgsm.epsilon)"
				}
				if let gridConfig = config.evaluationGridConfig {
					str += " --evaluation-grid=\(gridConfig.grid) --evaluation-grid-directory \(modelDir)"
				}
				if (config.evaluationStoreLossActivations ?? false) == true {
					str += " --evaluation-store-loss-activations --evaluation-store-loss-activations-directory \(modelDir)"
				}
				if config.startEpoch == config.fullTrainingEpochs || config.epochs == 0 {
					str += " --evaluate"
				}
				str += " --weight-decay \(config.weightDecay)"
				str += " --learning-rate \(config.learningRateSchedule.initial)"
				if !config.learningRateSchedule.decayEpochs.isEmpty {
					str += " --schedule \(config.learningRateSchedule.decayEpochs.map { String($0) }.joined(separator: " ") )"
				}
				str += " --gamma \(config.learningRateSchedule.decayEpochGamma)"
				str += " --every-epoch-gamma \(config.learningRateSchedule.everyEpochGamma)"
				str += " --unacceptable-accuracy-threshold \((config.fgsm?.epsilon ?? 0.0 > 0.0) ? 0.0 : 0.15)" // Only threshold accuracy if not running FGSM.
				str += " --unacceptable-accuracy-threshold-epoch 10"
				return str
			}()
			
		case .transformer:
			// TRANSFORMER:
			let taskName: String? = {
				switch config.target {
				case .glue_mrpc: return "MRPC"
				case .glue_cola: return "CoLA"
				case .glue_mnli: return "MNLI"
				case .glue_sst2: return "SST-2"
				case .glue_rte: return "RTE"
				case .glue_wnli: return "WNLI"
				case .glue_qqp: return "QQP"
				case .glue_stsb: return "STS-B"
				case .glue_qnli: return "QNLI"
				case .glue_ax: return "diagnostic"
				case .squad2: return nil
				default: return "UNKNOWN_TASK"
				}
			}()
			let dataDir = config.target.rawValue.contains("glue") ? "glue" : (config.target == .squad2 ? "data" : "UNKNOWN")
			guard taskName != "UNKNOWN_TASK" && dataDir != "UNKNOWN" else { fatalError() }
			return file + " " + {
				var str = " --model_name_or_path \(config.fumanchuModelArgument)"
				if let taskName = taskName {
					str += " --task_name \(taskName)"
				}
				str += " --loss_function '\(config.lossFunction)'"
//				str += " --eval-dataset \(valTestTrain)"
//				str += " --dataset-percentage \(config.trainingDatasetPercentage)"
				str += " --num_train_epochs \(config.epochs)"
				str += " --learning_rate \(config.learningRateSchedule.initial)"
				
				switch config.target {
				case .squad2:
					str += " --model_type \(config.fumanchuModelArgument.components(separatedBy: "-").first!)"
					str += " --max_seq_length 384 --doc_stride 128"
					str += " --version_2_with_negative --train_file \(dataDir)/train-v2.0.json --predict_file \(dataDir)/dev-v2.0.json"
					str += " --per_gpu_train_batch_size \(config.trainingBatchSize)"
					str += " --per_gpu_eval_batch_size \(config.evalBatchSize)"
				default:
					str += " --data_dir \(dataDir)/\(taskName ?? "")"
					str += " --max_seq_length 128"
					str += " --per_device_train_batch_size \(config.trainingBatchSize)"
					str += " --per_device_eval_batch_size \(config.evalBatchSize)"
				}
				
				if evaluationPartition == .testing {
					str += " --do_predict"
				} else {
					str += " --do_eval"
				}
				str += " --do_train --overwrite_output_dir"
				str += " --output_dir ../modeldir/\(taskName.flatMap { "\($0)/" } ?? "")"
				str += " --save_steps 20000" // checkpoint interval
				return str
			}()
		}
	}
	
	static func processOutput(_ output: String, config: TrainingConfig, evaluationPartition: DatasetPartition) -> (JobState, TrainingResults?) {
		let output = output.replacingOccurrences(of: "\0", with: "")
		guard !output.contains("NaN loss during training.") && !output.contains("Found Inf or NaN global norm.") && !output.contains("Validation perplexity is too high. Breaking out early...") && !output.contains("Nan in summary histogram") else {
			let results = [
				EvaluationMetric(metric: .loss, dataset: evaluationPartition): Double.greatestFiniteMagnitude,
				EvaluationMetric(metric: .accuracy, dataset: evaluationPartition): 0.0
			]
			return (.finished, results)
		}
		
		//		guard let finalLoss = output.components(separatedBy: "Loss for final step: ").last.flatMap({ str in
		//			return str.components(separatedBy: ".\n").first.flatMap { Double($0) }
		//		}) else {
		//			return nil
		//		}
		
		
		
		// General stat checking.
		
		let everythingSeparatedByStatsEnd = output.components(separatedBy: "}")
		let everythingBeforeStatsEnd = everythingSeparatedByStatsEnd[everythingSeparatedByStatsEnd.count == 1 ? 0 : everythingSeparatedByStatsEnd.count - 2]
		
		guard let statsLine = everythingBeforeStatsEnd.components(separatedBy: "\n").suffix(3).last else {
			return (.finished, nil)
		}
		
		guard !statsLine.isEmpty else {
			return (.finished, nil)
		}
		let statsMap = statsLine[statsLine.index(statsLine.firstIndex(of: "{")!, offsetBy: 1)..<String.Index(encodedOffset: statsLine.count)]
		let stats = statsMap.components(separatedBy: ", ").map { $0.components(separatedBy: ": ") }
		
		// Make sure we have a loss and at least one other metric.
		guard stats.count > 2 && (stats.first { $0[0] == (config.evaluator == .transformer ? "'eval_loss'" : "'loss'") } != nil) else {
			return (.finished, nil)
		}
		
		// Extract universal metrics.
		let statLoss = stats.first { $0[0] == (config.evaluator == .transformer ? "'eval_loss'" : "'loss'") }![1]
		
		var results = [EvaluationMetric(metric: .loss, dataset: .validation): Double(statLoss)!]
		
		// Perform evaluator specific result processing.
		switch config.evaluator {
		case .fumanchu:
			// FUMANCHU
			
			let statAccuracy = stats.first { $0[0] == "'accuracy'" }.flatMap { $0[1] }
			let statMSE = stats.first { $0[0] == "'mse'" }.flatMap { $0[1] }
			
			if let stat = statAccuracy { results[EvaluationMetric(metric: .accuracy, dataset: .validation)] = Double(stat)! }
			if let stat = statMSE { results[EvaluationMetric(metric: .mse, dataset: .validation)] = Double(stat)! }
			
		case .transformer:
			// TRANSFORMER
			
			let statAccuracy = stats.first { $0[0] == "'eval_acc'" }.flatMap { $0[1] }
			let statF1 = stats.first { $0[0] == "'eval_f1'" }.flatMap { $0[1] }
			let statMCC = stats.first { $0[0] == "'eval_mcc'" }.flatMap { $0[1] }
			let statMNLImmAcc = stats.first { $0[0] == "'eval_mnli-mm/acc'" }.flatMap { $0[1] }
			let statSpearman = stats.first { $0[0] == "'eval_spearmanr'" }.flatMap { $0[1] }
			let statPearson = stats.first { $0[0] == "'eval_pearson'" }.flatMap { $0[1] }
			let statCorrelation = stats.first { $0[0] == "'eval_corr'" }.flatMap { $0[1] }
			let statExactMatch = stats.first { $0[0] == "'exact'" }.flatMap { $0[1] }
			
			if let stat = statAccuracy { results[EvaluationMetric(metric: .accuracy, dataset: .validation)] = Double(stat)! }
			if let stat = statF1 { results[EvaluationMetric(metric: .f1, dataset: .validation)] = Double(stat)! }
			if let stat = statMCC { results[EvaluationMetric(metric: .mcc, dataset: .validation)] = Double(stat)! }
			if let stat = statMNLImmAcc { results[EvaluationMetric(metric: .mnli_mm_acc, dataset: .validation)] = Double(stat)! }
			if let stat = statSpearman { results[EvaluationMetric(metric: .spearman, dataset: .validation)] = Double(stat)! }
			if let stat = statPearson { results[EvaluationMetric(metric: .pearson, dataset: .validation)] = Double(stat)! }
			if let stat = statCorrelation { results[EvaluationMetric(metric: .correlation, dataset: .validation)] = Double(stat)! }
			if let stat = statExactMatch { results[EvaluationMetric(metric: .exactMatch, dataset: .validation)] = Double(stat)! }
			
		}
		
		// Return results.
		return (.finished, results)
	}
}
