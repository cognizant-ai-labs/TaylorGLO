# TaylorGLO
### **Santiago Gonzalez and Risto Miikkulainen (2021). [Optimizing Loss Functions Through Multivariate Taylor Polynomial Parameterization](http://nn.cs.utexas.edu/?gonzalez:gecco21). In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO-2021) (also [arXiv:2002.00059](https://arxiv.org/abs/2002.00059)).**

Metalearning of deep neural network (DNN) architectures and hyperparameters has become an increasingly important area of research. Loss functions are a type of metaknowledge that is crucial to effective training of DNNs, however, their potential role in metalearning has not yet been fully explored. Whereas early work focused on genetic programming (GP) on tree representations, this paper proposes continuous CMA-ES optimization of multivariate Taylor polynomial parameterizations. This approach, TaylorGLO, makes it possible to represent and search useful loss functions more effectively. In MNIST, CIFAR-10, and SVHN benchmark tasks, TaylorGLO finds new loss functions that outperform functions previously discovered through GP, as well as the standard cross-entropy loss, in fewer generations. These functions serve to regularize the learning task by discouraging overfitting to the labels, which is particularly useful in tasks where limited training data is available. The results thus demonstrate that loss function optimization is a productive new avenue for metalearning.

## Components

The system is composed of two key components that interact with each other:

* **Losssferatu** is the parallelized experiment host that runs evolutionary processes, manages results, and coordinates candidate evaluation. Lossferatu can run for extended periods of time without human intervention, much like its namesake Nosferatu.
* **Fumanchu** is a generic, model-agnostic neural network training and evaluation component built in Torch with a unified interface. One experiment may involve hundreds of unique invocations of Fumanchu. More informally, Fumanchu treats models as cattle, rather than as pets; the inspiration for being named after Fu Manchu the bull.

More details are available in Chapter 3 of Santiago Gonzalez's dissertation: [Improving Deep Learning Through Loss-Function Evolution](http://nn.cs.utexas.edu/?gonzalez:diss20).

## Getting Started and Navigating the Codebase

* `LossferatuRunner`, the binary executable for Lossferatu, can be compiled by invoking `build_and_release.sh` in the `LossferatuRunner` directory.
	* Lossferatu has [SwiftGenetics](https://github.com/sgonzalez/SwiftGenetics) and [SwiftCMA](https://github.com/sgonzalez/SwiftCMA) as dependencies.
* `LossferatuRunner` usage:
	* **Running experiments:**
	* `$ LossferatuRunner init EXPERIMENT_DIR CONFIG.json`
	* `$ LossferatuRunner start EXPERIMENT_DIR`
	* `$ LossferatuRunner check EXPERIMENT_DIR`
	* **Postprocessing results:**
	* `$ LossferatuRunner analyze EXPERIMENT_DIR`
	* `$ LossferatuRunner collateoneshots EXPERIMENTS_DIR`
	* `$ LossferatuRunner resummarize EXPERIMENTS_DIRS_DIR`
	* `$ LossferatuRunner resummarizegenerational EXPERIMENTS_DIR`
	* `$ LossferatuRunner ttest EXPERIMENTS_DIR_1 EXPERIMENTS_DIR_2`
	* **Miscellaneous:**
	* `$ LossferatuRunner getinvocation CONFIG.json`
	* `$ LossferatuRunner studiolog JOB_NAME (parse)`
	* `$ LossferatuRunner test`
* Example invocations for `LossferatuRunner` are shown in [LossferatuRunner/README.md](https://github.com/cognizant-ai-labs/TaylorGLO/blob/main/fumanchu/README.md).
* Example invocations for Fumanchu are shown in [fumanchu/TRAINING.md](https://github.com/cognizant-ai-labs/TaylorGLO/blob/main/fumanchu/TRAINING.md).
	* Dependencies for Fumanchu are listed in [fumanchu/requirements.txt](https://github.com/cognizant-ai-labs/TaylorGLO/blob/main/fumanchu/requirements.txt).
* Functional Lossferatu experiment configuration files can be found in the [experiments](https://github.com/cognizant-ai-labs/TaylorGLO/blob/main/experiments) directory
* Running evolution end-to-end requires functional [StudioML](https://studio.ml) infrastructure with the [studio-go-runner](https://github.com/studioml/studio-go-runner).
	* Notably, due to Lossferatu's modularity, it can be readily adapted to other infrastructure (take note of the `"evaluator": "studio"` field in experiment configs and `TrainingInterface.swift`).

## Citation

If you use **TaylorGLO** in your research, please cite it with the following BibTeX entry:

```
@article{taylorglo,
	author = {Gonzalez, Santiago and Miikkulainen, Risto},
	year = {2021},
	month = {07},
	pages = {},
	title = {Optimizing Loss Functions Through Multivariate Taylor Polynomial Parameterization},
	journal = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO-2021)}
}
```
