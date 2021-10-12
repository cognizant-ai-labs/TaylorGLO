//
//  HelperScripts.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 12/27/19.
//  Copyright © 2019 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct HelperScripts {
	static var runGeneration = """
		\(shabang)
		\(HelperScripts.scriptDir)
		LossferatuRunner start $DIRECTORY
		\(commandFinishGeneration)
		"""
	
	static var runAnalysis = """
		\(shabang)
		\(HelperScripts.scriptDir)
		LossferatuRunner analyze $DIRECTORY
		"""
	
	static var finishGeneration = """
		\(shabang)
		\(HelperScripts.scriptDir)
		\(commandFinishGeneration)
		"""
	
	
	static func runOneShotOneShotChildExperiment() -> String {
		return """
		\(shabang)
		\(scriptDir)
		cd $DIRECTORY
		cd ..
		
		# How many children to create.
		CHILDREN=$1
		echo "Children: $CHILDREN"

		# Which generation config to use.
		GENERATION=$2
		
		# Whether to download the modeldir.
		DOWNLOAD_AFTERWARDS=${3:-false}

		# Run child.
		mkdir children
		mkdir children/${GENERATION}
		PARENT_DIRECTORY="$(pwd)/children/${GENERATION}"

		# Init and start.
		for run in {1..${CHILDREN}}
		do
		  DIRECTORY="$PARENT_DIRECTORY/childOneShot_$(date +%s)"
		  echo "- Starting new experiment in directory:"
		  echo $DIRECTORY
		  LossferatuRunner init $DIRECTORY analyses/ExperimentConfig_${GENERATION}.json
		  LossferatuRunner start $DIRECTORY
		  sleep 2
		done
		
		# Check and finish.
		for DIRECTORY in $PARENT_DIRECTORY/childOneShot_*
		do
			JOBID=$(cat ${DIRECTORY}/one_shot/job_names.txt)
		
			\(commandFinishGeneration)
		
			# Download the modeldir if required.
			if [ "$DOWNLOAD_AFTERWARDS" = true ] ; then

				export AWS_ACCESS_KEY_ID=research
				export AWS_SECRET_ACCESS_KEY=****************************
				aws s3 cp --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/experiments/${JOBID}/modeldir.tar $DIRECTORY
				mkdir $DIRECTORY/modeldir
				tar -xvf $DIRECTORY/modeldir.tar -C $DIRECTORY/modeldir
				rm $DIRECTORY/modeldir.tar
			
			fi
		
		done
		
		# Collate!
		LossferatuRunner collateoneshots $PARENT_DIRECTORY
		"""
	}
	
	static func runOneShotChildExperiment(at path: String, generation: Int) -> String {
		return """
		\(shabang)
		\(scriptDir)
		cd $DIRECTORY
		cd ..
		
		# How many children to create.
		CHILDREN=${1:-1}
		echo "Children: $CHILDREN"
		
		# Which generation config to use.
		GENERATION=${2:-\(generation)}
		
		# Whether to download the modeldir.
		DOWNLOAD_AFTERWARDS=${3:-false}
		
		# Run child.
		mkdir children
		mkdir children/Gen${GENERATION}
		PARENT_DIRECTORY="$(pwd)/children/Gen${GENERATION}"
		
		# Init and start.
		for run in {1..${CHILDREN}}
		do
		  DIRECTORY="$PARENT_DIRECTORY/childOneShot_$(date +%s)"
		  echo "- Starting new experiment in directory:"
		  echo $DIRECTORY
		  LossferatuRunner init $DIRECTORY \(path.replacingOccurrences(of: "_best_candidate", with: "_Gen${GENERATION}_best_candidate"))
		  LossferatuRunner start $DIRECTORY
		  sleep 2
		done
		
		# Check and finish.
		for DIRECTORY in $PARENT_DIRECTORY/childOneShot_*
		do
			JOBID=$(cat ${DIRECTORY}/one_shot/job_names.txt)
		
			\(commandFinishGeneration)
		
			# Download the modeldir if required.
			if [ "$DOWNLOAD_AFTERWARDS" = true ] ; then

				export AWS_ACCESS_KEY_ID=research
				export AWS_SECRET_ACCESS_KEY=**********************
				aws s3 cp --endpoint-url=http://minio.somewhere.com:9000 s3://YOURBUCKET-datasets/experiments/${JOBID}/modeldir.tar $DIRECTORY
				mkdir $DIRECTORY/modeldir
				tar -xvf $DIRECTORY/modeldir.tar -C $DIRECTORY/modeldir
				rm $DIRECTORY/modeldir.tar
			
			fi
		done
		
		# Collate!
		LossferatuRunner collateoneshots $PARENT_DIRECTORY
		"""
	}
	
	private static var shabang = "#!/bin/zsh"
	private static var commandFinishGeneration = """
		until LossferatuRunner check $DIRECTORY; do echo "Retrying check in 20 seconds... ($(basename $DIRECTORY))"; sleep 20; done
		"""
	private static var scriptDir = "DIRECTORY=$(cd `dirname $0` && pwd)"
}
