//
//  MathematicaCode.swift
//  LossferatuCore
//
//  Created by Santiago Gonzalez on 1/6/20.
//  Copyright © 2020 Santiago Gonzalez. All rights reserved.
//

import Foundation

struct MathematicaCode {
	static func analysisNotebook(resultsCSV: String, candidatesCSV: String, best: SavedCandidate, experiment: ResultsManager) -> String {
		
		let resultsCSVEscaped = resultsCSV.replacingOccurrences(of: "\n", with: "\\n")
		let candidatesCSVEscaped = candidatesCSV.replacingOccurrences(of: "\n", with: "\\n")
		
		let isTemporalLossFunction = experiment.config.experiment?.taylorGloConfig?.inputVariables?.contains(.trainingCompleted) ?? false
		let numInputVariables = experiment.config.experiment?.taylorGloConfig?.inputVariables?.count ?? 2
		let evolvedLearningRateDecayPoints = experiment.config.experiment?.taylorGloConfig?.evolvedLearningRateDecayPoints ?? 0
		
		let trainingConfig = experiment.config.evaluation.baseTrainingConfig
		let evolutionPlotScaleCut: String = {
			switch trainingConfig.target {
			case .mnist: return "0.99"
			case .cifar10: return trainingConfig.epochs == trainingConfig.fullTrainingEpochs ? "0.9" : "0.8"
			case .cifar100: return "0.7"
			case .svhn: return "0.8"
			case .aircraft: return "0.5"
			case .crispr: return "0.8"
			default: return "0.5"
			}
		}()
		
		let epochCountLabel = String(trainingConfig.epochs)
		
		return
			"isTreeAnal = \((experiment.config.experiment?.treeGloConfig ?? nil != nil) ? "True" : "False");\n" +
			"isTaylorAnal = \((experiment.config.experiment?.taylorGloConfig ?? nil != nil) ? "True" : "False");\n" +
			"resultsCSVStr = \"\(resultsCSVEscaped)\";\n" +
			"lossFnsCandidatesCSVStr = \"\(candidatesCSVEscaped)\";\n" +
				"bestLossFn[x_,y_\(isTemporalLossFunction ? ",t_" : "")] := \(best.lossFunctionMathematica ?? "\"ERROR\"");\n" +
			"bestTensorflowStr = \"\(best.lossFunctionTensorFlow ?? "ERROR")\";\n" +
			"populationSize = \"\((experiment.config.experiment!.taylorGloConfig as? GenerationalExperimentConfig ?? experiment.config.experiment!.treeGloConfig as? GenerationalExperimentConfig)!.populationSize)\";\n" +
			"\n\n" + // TODO: yikes, this is hideous.
		"""
		resultsCSV = ImportString[resultsCSVStr];
		Print[resultsCSV[[1,All]]]
		resultsCSV = resultsCSV[[2;;-1,All]];
		candidatesCSV = ImportString[lossFnsCandidatesCSVStr];

		(* Styling, oh yeah! So shiny! *)
		vp = {3, -1, 1};
		style3d = Directive[Yellow, Specularity[White, 10]];
		niceBlue = RGBColor[.0,.64,1];
		niceGreen = RGBColor[.38,.85,.21];
		niceYellow = RGBColor[.97,.73,0];
		niceRed = RGBColor[1,.15,0];
		SetOptions[{Plot,Plot3D,StackedListPlot},BaseStyle->{FontFamily->"Helvetica Neue"}];

		(* Calculate some evolution results. *)
		generationAverage = resultsCSV[[All,3]]; (* All generations. *)
		generationBest = resultsCSV[[All,2]]; (* All generations. *)
		allTimeBest = Join[generationBest[[1;;1]],Map[Max[generationBest[[1;;#1]]]&, Range[2,Length[generationBest]]]]; (* All generations. *)
		newBestGenerations = Map[FirstPosition[allTimeBest, #1] - 1 &, DeleteDuplicates[allTimeBest]];
		generationJobFailures = resultsCSV[[All,8]]; (* All generations. *)
		generationInvariantViolations = resultsCSV[[All,9]]; (* All generations. *)
		generationJobSuccesses = ToExpression /@ ConstantArray[populationSize, Length[generationAverage]] - generationInvariantViolations - generationJobFailures; (* All generations. *)

		
		(* Useful tools. *)
		taylor3lossfn[x_, y_, p_] := p[[1]] (y - p[[8]]) + p[[2]] (y - p[[8]])^2 + p[[3]] (y - p[[8]])^3 + p[[4]] (x - p[[7]]) (y - p[[8]]) + p[[5]] (x - p[[7]]) (y - p[[8]])^2 + p[[6]] (x - p[[7]])^2 (y - p[[8]]);
		binme[fn_,x_,y_]:= -(fn[x,y]+fn[1-x,1-y]);
		binmeTemporal[fn_,x_,y_,t_] := -(fn[x, y, t] + fn[1 - x, 1 - y, t]);
		logBinary[x_,y_]:= -x*Log[y] -(1-x)*Log[1-y];
		jetfn[u_?NumericQ]:=Blend[{{0,RGBColor[0,0,9/16]}, {1/9,Blue}, {23/63,Cyan}, {13/21,Yellow}, {47/63,Orange}, {55/63,Red}, {1,RGBColor[1/2,0,0]}},u]/;0<=u<=1;

		(* Process best loss function. *)
		\(isTemporalLossFunction ?
			"""
			bestLossFnBinary[xx_,yy_,tt_] = binmeTemporal[bestLossFn,xx,yy,tt];
			bestLossMinimumX1T0 = FindMinimum[bestLossFnBinary[1,y,0],{y,0.0001,0.9999}];
			bestLossMinimumX1T1 = FindMinimum[bestLossFnBinary[1,y,1],{y,0.0001,0.9999}];
			"""
		:
			"""
			bestLossFnBinary[xx_,yy_] = binme[bestLossFn,xx,yy];
			bestLossMinimumX1 = FindMinimum[bestLossFnBinary[1,y],{y,0.0001,0.9999}];
			"""
		)
		binplot = Plot3D[{
			\(isTemporalLossFunction ?
				"""
				bestLossFnBinary[1,y,t] - bestLossFnBinary[1,1,t]
				"""
			:
				"""
				bestLossFnBinary[x,y]
				"""
			)
		},\(isTemporalLossFunction ? "{y,0,1},{t,0,1}" : "{x,0,1},{y,0,1}"),PlotPoints->60, PlotStyle->style3d];
		presentBinplot = Show[binplot,ViewPoint->vp,  AxesLabel->{\(isTemporalLossFunction ? "\"y_0\",\"t\"" : "\"x\",\"y\""),"loss"},
		LabelStyle->Directive[Black], TicksStyle->Directive[Small], PlotLabel-> "Binary Classification Loss (Best Validation Accuracy)"];

		(* Best loss function comparison plot. *)
		colorfn = ColorData["BlueGreenYellow"];
		lossFnComparisonPlot = Plot[{
			\(isTemporalLossFunction ?
				"""
				(*bestLossFnBinary[1,y,0] - bestLossMinimumX1T0[[1]],
				bestLossFnBinary[1,y,1] - bestLossMinimumX1T1[[1]]*)
				
				bestLossFnBinary[1,y,0] - bestLossFnBinary[1,1,0],
				bestLossFnBinary[1,y,0.25] - bestLossFnBinary[1,1,0.25],
				bestLossFnBinary[1,y,0.5] - bestLossFnBinary[1,1,0.5],
				bestLossFnBinary[1,y,0.75] - bestLossFnBinary[1,1,0.75],
				bestLossFnBinary[1,y,1] - bestLossFnBinary[1,1,1]
				"""
			:
				"""
				logBinary[1,y],
				bestLossFnBinary[1,y] - bestLossFnBinary[1,1] (*- bestLossMinimumX1[[1]]*)
				"""
			)
			}, {y,0,1},
			\(isTemporalLossFunction ?
				"""
				PlotStyle -> (colorfn /@ {0,0.25,0.5,0.75,1}),
				"""
			:
				"""
				PlotStyle->{niceBlue,niceGreen,niceYellow},
				"""
			)
			PlotTheme->{"Web","HeightGrid","ThickLines"},
			Frame->True,
			Axes->True,
			PlotLegends->LineLegend[{
			\(isTemporalLossFunction ?
				"""
				"Best Loss (t=0)",
				"Best Loss (t=0.25)",
				"Best Loss (t=0.5)",
				"Best Loss (t=0.75)",
				"Best Loss (t=1)"
				"""
			:
				"""
				"Log Loss",
				"Best Loss"
				"""
			)
			},LegendMargins->{-5,0}],
			LabelStyle->Directive[Medium, Black],
			ImageSize->320,
			FrameLabel->{"Predicted Label (\\!\\(\\*SubscriptBox[\\(y\\), \\(0\\)]\\))","Loss at \\!\\(\\*SubscriptBox[\\(x\\), \\(0\\)]\\)=1"}
			,PlotPoints->2000
		];

		(* Big evolution plot. *)
		evPlotStart = 0.2;
		evPlotCut = \(evolutionPlotScaleCut);
		evPlotUpperScale = 0.01 * 50 / (1 - evPlotCut);
		evPlotUpperPadding = 0; (*0.05;*)
		evPlotScalingFn[t1_, t2_, gap_: 1/10][x_] := Piecewise[{{x, x <= t1}, {t1 + If[t2 - t1 == 0, gap, gap/(t2 - t1) (x - t1)],
			t1 <= x <= t2}, {t1 + gap + (x - t2)*evPlotUpperScale, x >= t2}}];
		evPlotInverseScalingFn[t1_, t2_, gap_: 1/10][x_] := InverseFunction[evPlotScalingFn[t1, t2, gap]][x]
		evPlotTicks = Join[Range[0, evPlotCut, 0.2], Range[evPlotCut, 1, (1 - evPlotCut)/5]];
		evPlotTicksFrame = Join[Range[0, evPlotCut, 0.2], {evPlotCut, evPlotCut + (1 - evPlotCut)/2, 1}];
		evolutionPlot = ListLinePlot[{
				Function[point, Style[point, PointSize[0.015]]] /@ Join[Map[{#1[[1]], 1.0}&, newBestGenerations], {{Length[generationAverage]-1, -0.5}}],
				MapIndexed[{#2 - 1, #1}&, allTimeBest],
				MapIndexed[{#2 - 1, #1}&, generationBest],
				MapIndexed[{#2 - 1, #1}&, generationAverage]},
			DataRange->{0,Length[generationAverage]-1},
			PlotRange->{{-0.01*Length[generationAverage],Length[generationAverage]-1 + 0.01*Length[generationAverage]},{evPlotStart,1 + evPlotUpperPadding / evPlotUpperScale}},
			ScalingFunctions->{"Linear",{evPlotScalingFn[evPlotCut, evPlotCut, 0], evPlotInverseScalingFn[evPlotCut, evPlotCut, 0]}},
			Epilog->Line[{{-0.01*Length[generationAverage],evPlotCut+0.000001}, {Length[generationAverage]-1+0.01*Length[generationAverage],evPlotCut+0.000001}}],
			Ticks->{Automatic, evPlotTicks},
			GridLines->{Automatic, evPlotTicks},
			PlotStyle->{niceRed,niceYellow,niceGreen,niceBlue},
			PlotTheme->{"Web","HeightGrid","ThickLines"},
			GridLinesStyle->{Directive[Automatic,Dotted,Gray],Directive[Automatic,Thin,Gray]},
			PlotRangeClipping->False,
			FrameTicks->{Automatic, evPlotTicksFrame},
			
			(*Axes->True,
			PlotLegends->LineLegend[{
				"New Best",
				"All-Time Best",
				"Generation Best",
				"Generation Average"
			},LegendMargins->{-5,0}],
			Joined->{False,True,True,True},
			LabelStyle->Directive[Medium, Black],
			ImageSize->320,
			GridLines->Automatic,
			FrameLabel->{"Generation","\(epochCountLabel)-epoch Validation Accuracy"}*)
			Axes->True,
			PlotLegends->Placed[LineLegend[{
				"New Best",
				"All-Time Best",
				"Generation Best",
				"Generation Average"
				},LegendMargins->{-5,0},
				LegendFunction->(Framed[#1, FrameMargins -> 5, Background->White] & )], {0.785,0.3}],
			Joined->{False,True,True,True},
			LabelStyle->Directive[Medium, Black],
			ImageSize->380,
			GridLines->Automatic,
			FrameLabel->{"Generation","\(epochCountLabel)-epoch Validation Accuracy"}
		];

		(* Candidate status plot. *)
		candidateStatusPlot = StackedListPlot[{generationJobFailures, generationInvariantViolations, generationJobSuccesses},
			PlotTheme->{"Web","HeightGrid","ThickLines"},
			PlotStyle->{niceRed, niceYellow, niceGreen},
			Frame->True,
			Axes->True,
			DataRange->{0,Length[generationAverage]-1},
			PlotLegends->LineLegend[{
				"Training Failures",
				"Invariant Violations",
				"Successful Evaluations"
			},LegendMargins->{-5,0}],
			LabelStyle->Directive[Medium, Black],
			GridLinesStyle->{Directive[Automatic,Dotted,Gray],Directive[Automatic,Thin,LightGray]},
			ImageSize->320,
			GridLines->Automatic,
			FrameLabel->{"Generation","Candidates"}
		];
		
		(* Loss Params Dimensionality Reduction *)
		mat = candidatesCSV[[All,3;;-1]];
		generations = candidatesCSV[[All,1]];
		maxGenerations = Max[generations]
		generations = generations / maxGenerations;
		pca = DimensionReduce[mat,Method->"TSNE"];(*DimensionReduce[mat,Method\\[Rule]"TSNE"]; *)(*PrincipalComponents[N[mat]][[All,1;;2]];*)

		colorfn = jetfn; (*ColorData["Rainbow"];*)
		colors = colorfn /@ generations;
		zipped = MapThread[f[#1,#2] &,{pca,colors}];
		pca = Function[x,Style[x[[1]],x[[2]]]]/@zipped;

		plotLegend[{min_,max_},n_,col_, numfn_]:= (*  https://mathematica.stackexchange.com/questions/1300/listplot-with-each-point-a-different-color-and-a-legend-bar  *)Graphics[MapIndexed[{{col[#1],Rectangle[{0,#2[[1]]-1},{1,#2[[1]]}]},{Black,Text[NumberForm[numfn[N@#1],{4,2}],{4,#2[[1]]-.5},{1,0}]}}&,Rescale[Range[n],{1,n},{min,max}]],Frame->True,FrameTicks->None,PlotRangePadding->.5, FrameLabel->"Generation"]
		biplot = ListPlot[pca,PlotStyle->PointSize[0.01],AxesLabel->{"PC1","PC2"}];
		biplotleg=plotLegend[Through[{Min,Max}[generations]],16,colorfn, Function[x,Floor[maxGenerations*x]]];
		paramsDimensionalityReductionOverTime = Grid[{{Show[biplot,ImageSize->{Automatic,250}],Show[biplotleg,ImageSize->{Automatic,250}]}}];
		
		(* Generation Best Loss Params Over Evolution *)
		bestMat = Select[candidatesCSV, #1[[3]] == "True" &][[All,3;;\(3 + best.lossFunctionTaylor!.count)]];
		bestParamsPlot = ListLinePlot[Transpose[bestMat],
			PlotTheme->{"Web","HeightGrid","ThickLines"},
			PlotStyle->{Thickness[0.007]},
			PlotLegends->LineLegend[{
		\( ((numInputVariables..<best.lossFunctionTaylor!.count) + Array(0..<numInputVariables)).map { "\"\\!\\(\\*SubscriptBox[\\(\\[Theta]\\), \\(\($0)\\)]\\)\""}.joined(separator: ",") )
			},LegendMargins->{-5,0}],
			Frame->True,
			Axes->True,
			LabelStyle->Directive[Medium, Black],
			ImageSize->320,
			GridLines->Automatic,
			GridLinesStyle->{Directive[Automatic,Dotted,Gray],Directive[Automatic,Thin,LightGray]},
			FrameLabel->{"Generation","Parameter Value"}
		];
		
		\(evolvedLearningRateDecayPoints > 0 ?
			"""
			(* Generation Best LR Schedule Params Over Evolution *)
			bestLRScheduleMat = Select[candidatesCSV, #1[[3]] == "True" &][[All,\(-evolvedLearningRateDecayPoints);;-1]];
			bestLRScheduleParamsPlot = ListLinePlot[Transpose[bestLRScheduleMat],
				PlotTheme->{"Web","HeightGrid","ThickLines"},
				PlotStyle->{Thickness[0.007]},
				PlotLegends->LineLegend[{
				\( (0..<evolvedLearningRateDecayPoints).map { "\"Decay \($0)\""}.joined(separator: ",") )
				},LegendMargins->{-5,0}],
				Frame->True,
				Axes->True,
				LabelStyle->Directive[Medium, Black],
				ImageSize->320,
				GridLines->Automatic,
				GridLinesStyle->{Directive[Automatic,Dotted,Gray],Directive[Automatic,Thin,LightGray]},
				FrameLabel->{"Generation","Parameter Value"}
			];
			"""
		:
			""
		)
		
		(* Best Functions *)
		bestFunctions = Function[row, Function[{x,y\(isTemporalLossFunction ? ",t" : "")}, \(isTemporalLossFunction ? "taylor4lossfnTemporal" : "taylor3lossfn")[x,y\(isTemporalLossFunction ? ",t" : ""),row]]] /@ bestMat;
		bestFunctions = Function[bfn, binme\(isTemporalLossFunction ? "Temporal" : "")[bfn,1,y\(isTemporalLossFunction ? ",0" : "")]] /@ bestFunctions;
		bfcolors = colorfn /@ Range[0,1, 1/(Length[bestFunctions]-1)];
		bfstyle = Function[color, {Thickness[0.004],color}] /@ bfcolors;
		bestFunctionsRelativeLoss = Evaluate[Function[r, -(r - (r/.y -> 1)) ]   /@  (bestFunctions)];
		bfplot = Plot[bestFunctionsRelativeLoss, {y,0,1}, PlotStyle->bfstyle,
			PlotTheme->{"Web","HeightGrid","ThickLines"},
			Frame->True,
			Axes->True,
			LabelStyle->Directive[Medium, Black],
			ImageSize->320,
			GridLines->Automatic,
			GridLinesStyle->{Directive[Automatic,Dotted,Gray],Directive[Automatic,Thin,LightGray]},
			FrameLabel->{"Predicted Label (\\!\\(\\*SubscriptBox[\\(y\\), \\(0\\)]\\))","Relative Loss at \\!\\(\\*SubscriptBox[\\(x\\), \\(0\\)]\\)=1"}
		];
		bfplotfull = Grid[{{Show[bfplot,ImageSize->{Automatic,200}],Show[biplotleg,ImageSize->{Automatic,200}]}}];
		
		
		
		
		
		(* Zero Training Error Attraction *)
		c1[p_] := -2*p[[2]]*p[[4]] + 2*p[[1]]*p[[2]]*p[[7]] + p[[3]] -
		   p[[6]]*p[[1]] + p[[8]]*p[[1]]*p[[1]] + 3*p[[5]]*p[[2]]*p[[2]];
		ch[p_] := 2*p[[4]] - 2*p[[7]]*p[[1]] - 6*p[[5]]*p[[2]];
		chh[p_] := 3*p[[5]];
		chy[p_] := 2*p[[7]];
		cy[p_] := -2*p[[2]]*p[[7]] + p[[6]] - 2*p[[8]]*p[[1]];
		cyy[p_] := p[[8]];

		greaterThanZeroToEnsureZeroErrAttractor = ((-1 +
			 e) e (E^((-1 + e) e (gnt - gt)) - E^((
			 e ((-1 + e) gt (-1 + n) + gnt (-1 + e (-3 + n) + n)))/(-1 +
			   n)^2)))/((-1 + e) E^((-1 + e) e (gnt - gt)) -
		   e E^((e ((-1 + e) gt (-1 + n) + gnt (-1 + e (-3 + n) + n)))/(-1 +
			  n)^2));

		lutResolution = 1000;
		lutEps = Table[i, {i, 0, 1, 1/lutResolution}];
		lutTargets = 1 - # & /@ lutEps;
		lutNontargets = #/9 & /@ lutEps;
		
		params = bestMat[[\(best.generation)+1]][[2;;]]
		(* CONVERT PARAMS TO PAPER FORM (i.e., vars first) *)
		paperParamsBest = {Join [ params[[3;;-1]],params[[1;;2]]]}
		
		lutTargetGammasBase =
		  c1[paperParamsBest[[1]]] + (ch[paperParamsBest[[1]]] + chy[paperParamsBest[[1]]]) * # +
			 chh[paperParamsBest[[1]]] * #^2 + cy[paperParamsBest[[1]]] + cyy[paperParamsBest[[1]]] & /@
		   lutTargets;
		lutNontargetGammasBase =
		  c1[paperParamsBest[[1]]] + (ch[paperParamsBest[[1]]] + chy[paperParamsBest[[1]]]) * # +
			 chh[paperParamsBest[[1]]] * #^2 + cy[paperParamsBest[[1]]] + cyy[paperParamsBest[[1]]] & /@
		   lutNontargets;
		lutConstraintValuesBase =
		  MapThread[
		   greaterThanZeroToEnsureZeroErrAttractor /. {n -> \(trainingConfig.target == .cifar100 ? "100" : "10" ), e -> (1 - #1),
			   gt -> #2, gnt -> #3} &, {lutTargets, lutTargetGammasBase,
			lutNontargetGammasBase}];

		attractionPlot = ListLinePlot[{lutConstraintValuesBase}, DataRange->{0,1}, FrameLabel->{"Deviation from Memorization (\\[Epsilon])","Attraction Towards Zero Training"},Frame->True,PlotRange->{{0,0.9},{-1,1}},PlotStyle->niceBlue,LabelStyle->Directive[Medium, Black],ImageSize->320]


		
		
		
		
		
		(* "The Notebook", featuring nobody, because this is software, dude. Coming soon to a Mathematica installation near you! *)
		notebook = UsingFrontEnd[CreateDocument[{
		TextCell["\(experiment.config.name)", "Title"],
			
			TextCell["Best Result", "Section"],
			ExpressionCell[Button["Copy TensorFlow Loss String", CopyToClipboard[bestTensorflowStr]]],
			ExpressionCell[Button["Copy Job ID", CopyToClipboard[\"\(best.jobName ?? "")\"]]],
			TextCell["Generation:    \(best.generation)", "Text"],
			TextCell["Validation Accuracy:    " <> ToString[Last[allTimeBest]], "Text"],
			\(isTemporalLossFunction ?
				"""
				TextCell["Minimum (t=0) at:    " <> ToString[bestLossMinimumX1T0[[2]]], "Text"],
				TextCell["Minimum (t=1) at:    " <> ToString[bestLossMinimumX1T1[[2]]], "Text"],
				"""
			:
				"""
				TextCell["Minimum at:    " <> ToString[bestLossMinimumX1[[2]]], "Text"],
				"""
			)
			ExpressionCell[presentBinplot],
			ExpressionCell[lossFnComparisonPlot],
			ExpressionCell[attractionPlot],
			
			TextCell["Generational Results", "Section"],
			ExpressionCell[Button["Copy Results CSV", CopyToClipboard[resultsCSVStr]]],
			ExpressionCell[evolutionPlot],
			ExpressionCell[candidateStatusPlot],
			ExpressionCell[paramsDimensionalityReductionOverTime],
			ExpressionCell[bestParamsPlot],
			\(evolvedLearningRateDecayPoints > 0 ? "ExpressionCell[bestLRScheduleParamsPlot]," : "")
			ExpressionCell[bfplotfull],
		
			TextCell["Dissertation Save", "Section"],
			ExpressionCell[InputField[Dynamic[userExpName],String]],
			ExpressionCell[Button["Save!",
				Export["/Users/santiagogonzalez/Library/Mobile Documents/com~apple~CloudDocs/DISSERTATION/images/taylorglo_experiments/" <> userExpName <> "_evolution.pdf",evolutionPlot];
				Export["/Users/santiagogonzalez/Library/Mobile Documents/com~apple~CloudDocs/DISSERTATION/images/taylorglo_experiments/" <> userExpName <> "_bestfnbinary.png",lossFnComparisonPlot,ImageResolution->300];
				Export["/Users/santiagogonzalez/Library/Mobile Documents/com~apple~CloudDocs/DISSERTATION/images/taylorglo_experiments/" <> userExpName <> "_bestfns.png",bfplotfull,ImageResolution->300];
				Export["/Users/santiagogonzalez/Library/Mobile Documents/com~apple~CloudDocs/DISSERTATION/images/taylorglo_experiments/" <> userExpName <> "_attraction.pdf",attractionPlot]
			]]
		
		
		},
		WindowTitle->"Experiment Notebook"
		]];
		"""
	}
}
