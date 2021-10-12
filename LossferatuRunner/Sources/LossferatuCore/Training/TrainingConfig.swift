//
//  TrainingConfig.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/19/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct LearningRateSchedule: Codable {
	/// The initial learning rate for training.
	var initial: Double
	/// At which epochs is a decay scheduled to happen?
	var decayEpochs: [Int]
	/// How much LR is multiplied by at scheduled epochs.
	let decayEpochGamma: Double
	/// How much LR gets multiplied by at each epoch. You typically don't use this if you also have a decay schedule.
	let everyEpochGamma: Double
}

struct CutoutConfig: Codable {
	let numberHoles: Int
	let length: Int
}

struct CutMixConfig: Codable {
	let alpha: Double
	let probability: Double
}

struct FGSMAttackConfig: Codable {
	let epsilon: Double
}

struct EvaluationGridConfig: Codable {
	let grid: String
}

struct TrainingConfig: Codable {
	
	enum Model: String, Codable {
		case noskip_resnet20, resnet20, fixup_resnet20, pre_resnet20
		case noskip_resnet32, resnet32, fixup_resnet32, pre_resnet32
		case noskip_resnet56, resnet56, fixup_resnet56, pre_resnet56
		case pyramidnet110a48, pyramidnetb110a48
		case wrn168
		case wrn1016
		case wrn226
		case wrn285
		case wrn2810
		case densenetbc100k12
		case densenetbc250k24
		case alexnet
		case allcnnc
		case deepbind
		case transformer_bert_base_cased
		case transformer_distilbert_base_cased
	}
	
	enum Target: String, Codable {
		case mnist
		case cifar10
		case cifar100
		case svhn
		case aircraft
		case crispr
		case glue_mrpc
		case glue_cola
		case glue_mnli
		case glue_sst2
		case glue_rte
		case glue_wnli
		case glue_qqp
		case glue_stsb
		case glue_qnli
		case glue_ax
		case squad2
	}
	
	enum Evaluator: String, Codable {
		case fumanchu
		case transformer
	}
	
	enum Optimizer: String, Codable {
		case sgd = "SGD"
		case adam = "Adam"
	}
	
	/// The dataset that's used.
	let target: Target
	/// The model architecture that is used.
	var model: Model
	
	/// Loss function as a Python string.
	var lossFunction: String
	/// Number of training epochs.
	var epochs: Int
	/// Learning rate schedule.
	var learningRateSchedule: LearningRateSchedule
	/// What proportion of the training dataset to use, in the range (0,1].
	var trainingDatasetPercentage: Double
	/// An optional seed for the evaluators PRNGs.
	let seed: Int?
	/// The S3 location for where a base model's tar file lives; this allows fine-tuning or reevaluation.
	var baseModelS3Tar: String?
	/// A "start epoch" from which training continues.
	var startEpoch: Int?
	/// An optional weight decay value that overrides the default (as specified in `weightDecay`).
	let overrideWeightDecay: Double?
	/// An optional optimizer setting that overrides the default (as specified in `optimizer`).
	let overrideOptimizer: Optimizer?
	/// An optional number of auxiliary classifiers to use in the model.
	let auxiliaryClassifiers: Int?
	/// An optional overriden activation function as a Python string.
	var overrideActivationFunction: String?
	/// An optional configuration to perform cutout data augmentation.
	let cutout: CutoutConfig?
	/// An optional configuration to perform cutmix data augmentation.
	let cutmix: CutMixConfig?
	
	/// An optional configuration to perform an FGSM attack during evaluation.
	var fgsm: FGSMAttackConfig?
	
	/// An optional evaluation grid configuration.
	var evaluationGridConfig: EvaluationGridConfig?
	/// Whether loss function input activations are logged at each evaluation.
	var evaluationStoreLossActivations: Bool?
	
	/// The maximum number of times that training is attempted. Defaults to 1.
	var trainingAttempts: Int?
	
	/// An evaluator to use.
	var evaluator: Evaluator {
		switch target {
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax, .squad2: return .transformer
		default: return .fumanchu
		}
	}
	
	var weightDecay: Double {
		if let wd = overrideWeightDecay { return wd }
		switch target {
		case .mnist: return 0
		case .cifar10, .cifar100, .svhn, .aircraft:
			switch model {
			case .alexnet: return 0.0005
			case .noskip_resnet20, .resnet20, .fixup_resnet20, .pre_resnet20,
				 .noskip_resnet32, .resnet32, .fixup_resnet32, .pre_resnet32,
				 .noskip_resnet56, .resnet56, .fixup_resnet56, .pre_resnet56: return 0.0001
			case .pyramidnet110a48, .pyramidnetb110a48: return 0.0001
			case .wrn168, .wrn1016, .wrn226, .wrn285, .wrn2810: return 0.0005
			case .densenetbc100k12, .densenetbc250k24: return 0.0001
			case .allcnnc: return 0.001
			case .deepbind, .transformer_bert_base_cased, .transformer_distilbert_base_cased: fatalError("Cannot use DeepBind on this domain.")
			}
		case .crispr: return 0
		default: fatalError("N/A")
		}
	}
	
	var fullTrainingEpochs: Int {
		switch target {
		case .mnist: return 40
		case .cifar10, .cifar100, .svhn, .aircraft:
			switch model {
			case .alexnet: return 164
			case .noskip_resnet20, .resnet20, .fixup_resnet20, .pre_resnet20,
				 .noskip_resnet32, .resnet32, .fixup_resnet32, .pre_resnet32,
				 .noskip_resnet56, .resnet56, .fixup_resnet56, .pre_resnet56: return 200
			case .wrn168, .wrn1016, .wrn226, .wrn285, .wrn2810: return target == .svhn ? 160 : 200
			case .pyramidnet110a48, .pyramidnetb110a48: return 300
			case .densenetbc100k12, .densenetbc250k24: return 300
			case .allcnnc: return 350
			case .deepbind, .transformer_bert_base_cased, .transformer_distilbert_base_cased: fatalError("Cannot use DeepBind on this domain.")
			}
		case .crispr: return 16
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax: return 3
		case .squad2: return 4
		}
	}
	
	var trainingBatchSize: Int {
		switch target {
		case .mnist, .cifar10, .cifar100, .svhn, .aircraft: return 128
		case .crispr: return 256
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax: return 32
		case .squad2: return 2
		}
	}
	
	var evalBatchSize: Int {
		switch target {
		case .mnist, .cifar10, .cifar100, .svhn, .aircraft: return 100
		case .crispr: return 256
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax: return 8
		case .squad2: return 2
		}
	}
	
	var optimizer: Optimizer {
		if let opt = overrideOptimizer { return opt }
		switch target {
		case .mnist, .cifar10, .cifar100, .svhn, .aircraft: return .sgd
		case .crispr: return .adam
		default: fatalError("N/A")
		}
	}
	
	var gpuSetting: StudioGPUSetting {
		switch model {
		case .densenetbc100k12, .densenetbc250k24: return .fullGPU
		case .noskip_resnet20, .resnet20, .fixup_resnet20, .pre_resnet20, .noskip_resnet32, .resnet32, .fixup_resnet32, .pre_resnet32, .noskip_resnet56, .resnet56, .fixup_resnet56, .pre_resnet56, .wrn168, .wrn1016, .wrn226, .wrn285, .wrn2810, .alexnet, .allcnnc, .deepbind, .pyramidnet110a48, .pyramidnetb110a48: return .halfGPU
		case .transformer_distilbert_base_cased: return .halfGPU
		case .transformer_bert_base_cased: return .fullGPU
		}
	}
	
	var fumanchuModelArgument: String {
		switch model {
		case .alexnet, .allcnnc, .deepbind: return model.rawValue
		case .wrn168: return "wrn --depth 16 --widen-factor 8 --drop \(target == .svhn ? "0.4" : "0.3")"
		case .wrn1016: return "wrn --depth 10 --widen-factor 16 --drop \(target == .svhn ? "0.4" : "0.3")"
		case .wrn226: return "wrn --depth 22 --widen-factor 6 --drop \(target == .svhn ? "0.4" : "0.3")"
		case .wrn285: return "wrn --depth 28 --widen-factor 5 --drop \(target == .svhn ? "0.4" : "0.3")"
		case .wrn2810: return "wrn --depth 28 --widen-factor 10 --drop \(target == .svhn ? "0.4" : "0.3")"
		case .pyramidnet110a48: return "pyramidnet --depth 110 --pyramidnet-alpha 48"
		case .pyramidnetb110a48: return "pyramidnet --depth 110 --pyramidnet-alpha 48 --pyramidnet-bottleneck"
		case .densenetbc100k12: return "densenet --depth 100 --growthRate 12"
		case .densenetbc250k24: return "densenet --depth 250 --growthRate 24"
		case .noskip_resnet20, .resnet20, .fixup_resnet20, .pre_resnet20,
			 .noskip_resnet32, .resnet32, .fixup_resnet32, .pre_resnet32,
			 .noskip_resnet56, .resnet56, .fixup_resnet56, .pre_resnet56:
			let prefix = model.rawValue.components(separatedBy: "_").first!
			let coreName = model.rawValue.components(separatedBy: "_").last!
			let depth = coreName.suffix(from: String.Index(utf16Offset: 6, in: coreName))
			if prefix == "pre" {
				return "preresnet --depth \(depth)"
			} else {
				return "\(prefix != coreName ? "\(prefix)_" : "")resnet --depth \(depth)"
			}
		case .transformer_distilbert_base_cased: return "distilbert-base-cased"
		case .transformer_bert_base_cased: return "bert-base-cased"
		}
	}
}


extension TrainingConfig.Target {
	
	var numClasses: Int? {
		switch self {
		case .mnist, .cifar10, .svhn: return 10
		case .cifar100, .aircraft: return 2000
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax, .squad2, .crispr: return nil
		}
	}
	
	var maxJobDurationMins: Int {
		switch self {
		case .mnist: return 30
		case .cifar10, .cifar100, .svhn, .aircraft, .crispr: return 2000
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax: return 2000
		case .squad2: return 2000
		}
	}
	
	var scriptLocation: String {
		switch self {
		case .mnist, .cifar10, .cifar100, .svhn, .aircraft, .crispr: return "torch/fumanchu/"
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax, .squad2: return "torch/transformers/"
		}
	}
	
	var scriptFile: String {
		switch self {
		case .mnist, .cifar10, .cifar100, .svhn, .aircraft, .crispr: return "fumanchu.py"
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax: return "examples/text-classification/run_glue.py"
		case .squad2: return "examples/question-answering/run_squad.py"
		}
	}
	
	var prefetchCommand: String? {
		switch self {
		case .mnist: fatalError()//"export AWS_ACCESS_KEY_ID=research; export AWS_SECRET_ACCESS_KEY=************************; date -u; aws s3 sync --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/MNIST-data MNIST-data && echo DOWNLOADED DATASET FROM MINIO && pwd && ls && echo MINIO END && date -u"
		case .cifar10, .cifar100, .svhn, .aircraft, .crispr: return "export AWS_ACCESS_KEY_ID=research; export AWS_SECRET_ACCESS_KEY=************************; date -u; aws s3 sync --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/\(self.rawValue)_data data && echo DOWNLOADED DATASET FROM MINIO && pwd && ls && echo MINIO END && date -u"
		case .glue_mrpc, .glue_cola, .glue_mnli, .glue_sst2, .glue_rte, .glue_wnli, .glue_qqp, .glue_stsb, .glue_qnli, .glue_ax: return "pip install .; pip install --upgrade tensorflow==2.2.0; pip install -r ./examples/requirements.txt; python utils/download_glue_data.py --data_dir glue --tasks all"
		case .squad2: return "export AWS_ACCESS_KEY_ID=research; export AWS_SECRET_ACCESS_KEY=************************; date -u; aws s3 sync --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/\(self.rawValue)_data data && aws s3 cp --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/cached_train_distilbert-base-cased_384 . && echo DOWNLOADED DATASET FROM MINIO && pwd && ls && echo MINIO END && date -u;          pip install .; pip install --upgrade tensorflow==2.2.0; pip install -r ./examples/requirements.txt"
		}
	}
}

extension TrainingConfig: CustomStringConvertible {
	var description: String {
		return "\(target.rawValue) for \(epochs) epochs, LR=\(learningRateSchedule.initial), train%=\(trainingDatasetPercentage), loss=\(lossFunction)"
	}
}
