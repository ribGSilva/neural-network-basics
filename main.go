package main

import (
	"fmt"
	"neural-network-basics/dp"

	"gonum.org/v1/gonum/mat"
)

//TIP <p>To run your code, right-click the code and select <b>Run</b>.</p> <p>Alternatively, click
// the <icon src="AllIcons.Actions.Execute"/> icon in the gutter and select the <b>Run</b> menu item from here.</p>

func main() {

	// infrastructure definition
	numberOfLayers := 3
	layerDimensions := []int{2, 3, 3, 1}

	fmt.Printf("Number of layers %d \n", numberOfLayers)
	fmt.Printf("Layer dimensions: %v \n", layerDimensions)

	initialInput := mat.NewDense(layerDimensions[0], 10, []float64{
		150, 70,
		254, 73,
		312, 68,
		120, 60,
		154, 61,
		212, 65,
		216, 67,
		145, 67,
		184, 64,
		130, 69,
	})
	correctAnswers := mat.NewDense(layerDimensions[numberOfLayers], 10, []float64{0, 1, 1, 0, 0, 1, 1, 0, 1, 0})

	neuralNetwork := dp.NewNeuralNetwork(numberOfLayers, layerDimensions)

	alfa := 0.1
	epochs := 1000
	costs := make([]float64, epochs)

	for i := range epochs {
		result := neuralNetwork.FeedForward(initialInput)

		costs[i] = dp.CalculateCost(result, correctAnswers)

		neuralNetwork.BackPropagate(correctAnswers, alfa)

		if i%20 == 0 {
			fmt.Printf("epoch %d: cost %.4f\n", i, costs[i])
			fmt.Print(mat.Formatted(result))
			fmt.Println()
		}
	}
}
