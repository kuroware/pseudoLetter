//Some constants
const mixtures = 5;
var priorDistribution;
var priorMultinomial = [];

var model = {}; //To store model information

let trainingExamples = []; //Examples seen so far
const contextLength = 20; //Memoryless afterwards
const updateEvery = 5; //Update model every 5 key downs


//HACKY way to get a quick renormalize function
Array.prototype.normalize = function() {
	let normalizeConstant =  this.reduce(function(element, currentSum) {
		return currentSum + element;
	}, 0);

	for (let i = 0; i < this.length; i++) {
		this[i] /= normalizeConstant;
	}
}

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
		// //Renormalize
		model[i].params.normalize();	
	}
}

//Compute the predictive distribution implicitly and samples
// sum of p(z|X)(x'|z, X) - X is a set of observations
function predictive() {
	//Sample from mixtures - sample a mixture probability from the conditionals
	let conditionalMixtures = [];
	for (let k = 0; k < mixtures; k++) {
		//p(z|X) = p(z)p(X|z = k)
		let p_X_given_z = 0;
		for (let i = 0; i < trainingExamples.length; i++) {
			p_X_given_z += Math.log(model[k+1].p(trainingExamples[i]));
		}
		let p_z_given_X = Math.log(priorMultinomial[k]) + p_X_given_z;
		conditionalMixtures.push(Math.exp(p_z_given_X));
	}
	conditionalMixtures.normalize();
	console.log(conditionalMixtures);

	//Sample from this distribution
	let x = Math.random();
	let curSum = 0;
	let samplingOutcome = 1;
	for (let i = 0; i < conditionalMixtures.length; i++) {
		curSum += conditionalMixtures[i];
		if (curSum > x) {
			samplingOutcome = i + 1;
			break;
		}
	}
	//Sample from the multinomial 
	x = Math.random();
	curSum = 0;
	samplingLetterOutcome = 0;
	for (let i = 0; i < model[samplingOutcome].params.length; i++) {
		curSum += model[samplingOutcome].params[i];
		if (curSum > x) {
			samplingLetterOutcome = i;
			break;
		}
	}
	return samplingLetterOutcome;
}

function init() {
	//Set up prior distribution

	priorDistribution = new DirichletDistribution([0.2, 0.2, 0.2, 0.2, 0.2]);
	for (let i = 0; i < mixtures; i++) {
		priorMultinomial.push(Math.random());
	}
	priorMultinomial.normalize();

	//Set up my distributions in the model
	for (let i = 1; i <= mixtures; i++) {
		let vectorProbabilities = []
		for (let j = 0; j < 26; j++) {
			vectorProbabilities[j] = Math.random(); //By default
		}
		model[i] = new MultinomialDistribution(vectorProbabilities);
	}
}

init();
var i = 0;
//For the actual site
let predictedLetter = null;
let predictedCorrect = 0;
let totalPredictions = 0;
let previousAccuracy = 0;
window.onkeydown = function(event) {
	console.log(event);
	let keyValue = event.key.charCodeAt(0) - "a".charCodeAt(0);
	if (keyValue >= 0 && keyValue <= 25) {
		//Accumulate some training examples
		trainingExamples.push(keyValue);

		if (trainingExamples.length > 15) {
			//Check to see if we were right

			let predColor = 'red';
			if (predictedLetter != null && predictedLetter == keyValue) {
				predictedCorrect += 1;
				predColor = 'green';
			}

			//Log the predictions
			let predictionSummary = document.createElement("div");
			predictionSummary.style.textAlign = 'center';
			predictionSummary.style.color = predColor;
			predictionSummary.style.marginBottom = '1px';
			predictionSummary.innerHTML = "Predicted " + String.fromCharCode(predictedLetter + "a".charCodeAt(0)) + " and read " + String.fromCharCode(keyValue + "a".charCodeAt(0));
			document.getElementById("predictions").appendChild(predictionSummary);

			if (document.getElementById("predictions").childNodes.length > 10) {
				document.getElementById("predictions").removeChild(document.getElementById("predictions").childNodes[0]);
			}

			//Run the M-step every so often
			if (i == 0) {
				for (let k = 0; k < 20; k++) {
					M();
				}
				console.log(priorMultinomial);
			}
			i = (i+1)%updateEvery;
			//Calculate predictive
			predictedLetter = predictive();
			totalPredictions++;

			let color = (predictedCorrect/totalPredictions) > previousAccuracy ? 'green' : 'red';
			previousAccuracy = predictedCorrect/totalPredictions;
			console.log(String.fromCharCode(predictedLetter + "a".charCodeAt(0)));
			document.getElementById("predicted-correct").innerHTML = predictedCorrect;
			document.getElementById("total-predictions").innerHTML = totalPredictions;
			document.getElementById("accuracy").innerHTML = (predictedCorrect/totalPredictions * 100).toFixed(2) + "%";
			document.getElementById("accuracy").style.color = color;
			document.getElementById("stats").style.display = "block";
		}

	}
}