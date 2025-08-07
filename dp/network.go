package dp

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type NeuralNetwork struct {
	numberOfLayers  int
	layerDimensions []int

	dataCache []*mat.Dense
	weights   []*mat.Dense
	biases    []*mat.VecDense
}

func NewNeuralNetwork(numberOfLayers int, layerDimensions []int) *NeuralNetwork {
	if numberOfLayers <= 0 {
		panic("Number of layers must be greater than zero")
	}
	if len(layerDimensions) != numberOfLayers+1 {
		panic("Number of layers must match the number of dimension layers (n+1)")
	}
	for _, layerDimension := range layerDimensions {
		if layerDimension <= 0 {
			panic("Layer dimension must be greater than zero")
		}
	}
	nn := &NeuralNetwork{
		numberOfLayers:  numberOfLayers,
		layerDimensions: layerDimensions,

		dataCache: make([]*mat.Dense, numberOfLayers+1),
		weights:   make([]*mat.Dense, numberOfLayers),
		biases:    make([]*mat.VecDense, numberOfLayers),
	}

	for i := 0; i < numberOfLayers; i++ {
		nn.weights[i] = generateRandomMatrix(layerDimensions[i+1], layerDimensions[i])
		nn.biases[i] = generateRandomVector(layerDimensions[i+1])
	}

	return nn
}

// FeedForward receives an input and propagates the information through the network until its output
func (nn *NeuralNetwork) FeedForward(input *mat.Dense) *mat.Dense {
	data := mat.NewDense(input.RawMatrix().Rows, input.RawMatrix().Cols, input.RawMatrix().Data)

	for i := 0; i < nn.numberOfLayers; i++ {
		nn.dataCache[i] = mat.NewDense(data.RawMatrix().Rows, data.RawMatrix().Cols, data.RawMatrix().Data)
		data = applyLayer(nn.dataCache[i], nn.weights[i], nn.biases[i])
	}

	nn.dataCache[len(nn.dataCache)-1] = mat.NewDense(data.RawMatrix().Rows, data.RawMatrix().Cols, data.RawMatrix().Data)

	return data
}

// applyLayer performs the calculation for a neural network layer over an input
func applyLayer(input mat.Matrix, weight mat.Matrix, bias mat.Vector) *mat.Dense {
	result := new(mat.Dense)
	result.Mul(weight, input)
	// add bias
	result.Apply(func(i, j int, v float64) float64 {
		return v + bias.AtVec(i)
	}, result)
	// apply activation function
	result.Apply(func(_, _ int, v float64) float64 {
		return sigmoid(v)
	}, result)
	return result
}

// BackPropagate does the calculation of the gradient for each layer, and apply the alfa scalar to both weight and biases
func (nn *NeuralNetwork) BackPropagate(expectedOutput *mat.Dense, alfa float64) {

	outputLayer := nn.dataCache[nn.numberOfLayers]

	currentError := mat.NewDense(outputLayer.RawMatrix().Rows, outputLayer.RawMatrix().Cols, nil)
	currentError.Sub(outputLayer, expectedOutput)

	for layerNumber := nn.numberOfLayers - 1; layerNumber >= 0; layerNumber-- {
		layerOutput, layerInput := nn.dataCache[layerNumber+1], nn.dataCache[layerNumber]

		numberOfOutputs, numberOfSamples := layerOutput.Dims()
		numberOfInputs, _ := layerInput.Dims()

		dC_dZ := mat.NewDense(numberOfOutputs, numberOfSamples, nil)

		// step 1. calculate dC/dZ = (current error) * sigmoid'(Z)
		dC_dZ.CloneFrom(currentError)

		dC_dZ.Apply(func(i, j int, v float64) float64 {
			a := layerOutput.At(i, j)
			return v * a * (1 - a)
		}, dC_dZ)

		dC_dZ.Scale(1/float64(numberOfSamples), dC_dZ)
		if r, c := dC_dZ.Dims(); r != numberOfOutputs || c != numberOfSamples {
			panic("Expected number of dimensions to be the same as the number of dimensions")
		}

		//  step 2. calculate dC/dW = dC/dZ * dZ/dW
		dC_dW := mat.NewDense(numberOfOutputs, numberOfInputs, nil)
		dC_dW.Mul(dC_dZ, layerInput.T())
		if r, c := dC_dW.Dims(); r != numberOfOutputs || c != numberOfInputs {
			panic("Expected number of dimensions to be the same as the number of dimensions")
		}

		//  step 3. calculate dC/dB = dC/dZ * dZ/dB np.sum(dC/dZ3, axis=1, keepdims=True)
		dC_dBValues := make([]float64, numberOfOutputs)
		for i := 0; i < numberOfOutputs; i++ {
			dC_dBValues[i] = floats.Sum(mat.Row(nil, i, dC_dZ))
		}
		dC_dB := mat.NewVecDense(numberOfOutputs, dC_dBValues)

		//  step 4. calculate propagator dC/dA^[Li-1] = dC/dZ * dZ/dA
		dC_dA := mat.NewDense(numberOfInputs, numberOfSamples, nil)
		dC_dA.Mul(nn.weights[layerNumber].T(), dC_dZ)

		dC_dW.Scale(alfa, dC_dW)
		nn.weights[layerNumber].Sub(nn.weights[layerNumber], dC_dW)

		dC_dB.ScaleVec(alfa, dC_dB)
		nn.biases[layerNumber].SubVec(nn.biases[layerNumber], dC_dB)

		currentError = dC_dA
	}
}

// CalculateCost uses the binary cross entropy loss function
func CalculateCost(predictionOutput *mat.Dense, correctAnswers *mat.Dense) float64 {
	lossFunc := func(want, got float64) float64 {
		return -((want * math.Log(got)) + ((1 - want) * math.Log(1-got)))
	}

	losses := mat.NewDense(predictionOutput.RawMatrix().Rows, predictionOutput.RawMatrix().Cols, nil)
	losses.Apply(func(i, j int, v float64) float64 {
		return lossFunc(correctAnswers.At(i, j), v)
	}, predictionOutput)

	losses.Scale(1/float64(predictionOutput.RawMatrix().Cols), losses)

	summedLosses := floats.Sum(losses.RawMatrix().Data)

	return summedLosses
}

func generateRandomMatrix(numberOfRows, numberOfColumns int) *mat.Dense {
	data := make([]float64, numberOfRows*numberOfColumns)
	for i := 0; i < numberOfRows*numberOfColumns; i++ {
		data[i] = rand.Float64()
	}
	return mat.NewDense(numberOfRows, numberOfColumns, data)
}

func generateRandomVector(numberOfItems int) *mat.VecDense {
	data := make([]float64, numberOfItems)
	for i := 0; i < numberOfItems; i++ {
		data[i] = rand.Float64()
	}
	return mat.NewVecDense(numberOfItems, data)
}

// used as activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
