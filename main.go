package main

import (
	"fmt"
	"neural-network-basics/dp"

	"gonum.org/v1/gonum/mat"
)

func main() {

	// infrastructure definition
	numberOfLayers := 2
	layerDimensions := []int{2, 4, 1}

	fmt.Printf("Number of layers %d \n", numberOfLayers)
	fmt.Printf("Layer dimensions: %v \n", layerDimensions)

	initialInput := mat.NewDense(layerDimensions[0], 4, []float64{
		0, 0, 1, 1,
		0, 1, 0, 1,
	})
	correctAnswers := mat.NewDense(layerDimensions[numberOfLayers], 4, []float64{0, 1, 1, 0})

	neuralNetwork := dp.NewNeuralNetwork(numberOfLayers, layerDimensions)

	alfa := 1.0
	epochs := 10000
	costs := make([]float64, epochs)

	for i := 0; i < epochs; i++ {
		result := neuralNetwork.FeedForward(initialInput)

		costs[i] = dp.CalculateCost(result, correctAnswers)

		neuralNetwork.BackPropagate(correctAnswers, alfa)

		if i%20 == 0 {
			fmt.Printf("epoch %d: cost %.4f\n", i, costs[i])
			fmt.Print(mat.Formatted(result))
			fmt.Println()
		}
	}

	fmt.Println(mat.Formatted(correctAnswers))
}
