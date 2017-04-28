//Some constants
const mixtures = 5;
var priorDistribution;
var priorMultinomial;

var model = {}; //To store model information

let trainingExamples = []; //Examples seen so far
const contextLength = 20; //Memoryless afterwards
const updateEvery = 5; //Update model every 5 key downs

/**
A few probability distribution definitions
**/
function DirichletDistribution(vectorParams) {
	this.params = vectorParams;
}

function MultinomialDistribution(vectorParams) {
	this.params = vectorParams;
}

//Computes P(\theta)
DirichletDistribution.prototype.p = function(vector) {
	return vector.reduce(function(element, currentProduct, i){
		return currentProduct * Math.pow(element, this.params[i]);
	}, 1);
}

MultinomialDistribution.prototype.p = function(outcome) {
	return this.params[outcome];
}


//E-step for EM algorithm - returns a N x K matrix of responsibilities
function E() {
	let responsibilitiesMatrix = [];
	//Return an N x K matrix
	for (let i = 0; i < trainingExamples.length; i++) {
		responsibilitiesMatrix[i] = [];
		for (let j = 0; j < mixtures; j++) {
			if (model[j+1].p(trainingExamples[i]) == 0) {
				responsibilitiesMatrix[i].push(Math.log(priorMultinomial[j]) + 1/260);
			}
			else {
				responsibilitiesMatrix[i].push(Math.log(priorMultinomial[j]) + Math.log(model[j+1].p(trainingExamples[i])));
			}
		}
	}

	//Convert to normal probabilities and renormalize
	for (let i = 0; i < trainingExamples.length; i++) {
		for (let j = 0; j < mixtures; j++) {
			responsibilitiesMatrix[i][j] = Math.exp(responsibilitiesMatrix[i][j]);
		}
	}

	///Renormalize
	for (let i = 0; i < responsibilitiesMatrix.length; i++) {
		let normalizingConstant = responsibilitiesMatrix[i].reduce(function(element, currentSum) {
			return currentSum + element;
		}, 0);
		for (let j = 0; j < responsibilitiesMatrix[i].length; j++) {
			responsibilitiesMatrix[i][j] /= normalizingConstant;
		}
	}

	return responsibilitiesMatrix;
}

//M step for EM algorithm
function M() {
	let responsibilitiesMatrix = E();
	let N = 0; //Sum of all responsibilities
	for (let i = 0; i < trainingExamples.length; i++) {
		for (let j = 0; j < mixtures; j++) {
			N += responsibilitiesMatrix[i][j];
		}
	}

	//Update the prior multinomial
	for (let i = 0; i < mixtures; i++) {
		let responsibilitiesSum = 0;
		for (let j = 0; j < trainingExamples.length; j++) {
			responsibilitiesSum += responsibilitiesMatrix[j][i];
		}
		priorMultinomial[i] = responsibilitiesSum/N;
	}

	//Update the Multinomial - derivation is just the frequency of it
	for (let i = 1; i <= mixtures; i++) {
		for (let j = 0; j < 26; j++) {
			//Find occurences in training examples
			let update = 0;
			let freq = 0;
			for (let a = 0; a < trainingExamples.length; a++) {
				if (trainingExamples[a] == j) {
					update += responsibilitiesMatrix[a][i - 1];
					freq += 1;
				}
			}
			model[i].params[j] = update*freq;
		}
		//Renormalize
		let normalizationSum = model[i].params.reduce(function(element, currentSum) {
			return currentSum + element;
		}, 0);
		model[i].params = model[i].params.map(function(element) {
			return element/normalizationSum;
		});
	}
}

function init() {
	//Set up prior distribution

	priorDistribution = new DirichletDistribution([0.2, 0.2, 0.2, 0.2, 0.2]);
	priorMultinomial = [0.18, 0.28, 0.36, 0.15, 0.11];

	//Set up my distributions in the model
	for (let i = 1; i <= mixtures; i++) {
		let vectorProbabilities = []
		for (let j = 0; j < 26; j++) {
			vectorProbabilities[j] = 1/26; //By default
		}
		model[i] = new MultinomialDistribution(vectorProbabilities);
	}
}

init();
trainingExamples = [2, 5, 2];
for (let i =0; i < 20; i++) {
	M();
}
console.log(priorMultinomial)
// // var i = 0;
// // //For the actual site
// // window.onkeydown = function(event) {
// // 	console.log(event);
// // 	let keyValue = event.key.charCodeAt(0) - "a".charCodeAt(0);
// // 	if (keyValue >= 0 && keyValue <= 25) {
// // 		trainingExamples.push(keyValue);
// // 		//Run the M-step every so often
// // 		if (i == 0) {
// // 			for (let k = 0; k < 20; k++) {
// // 				M();
// // 			}
// // 			console.log(priorMultinomial);
// // 		}
// // 		i = (i+1)%updateEvery;
// // 	}
// }