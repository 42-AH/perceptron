package main

import (
    "math/rand"
    "fmt"
)

type Perceptron struct {
    inputLayer [2]float64
    weights    [2]float64
    bias       float64
    prediction float64
    target     float64
    ebl        float64
}

func (p *Perceptron) cost() float64 {
    return (p.target - p.prediction) * (p.target - p.prediction) / 2
}

func (p *Perceptron) createPerceptron() {
    for i := 0; i < len(p.inputLayer); i++ {
        p.inputLayer[i] = rand.Float64()
        p.weights[i] = rand.Float64()
    }
    p.bias = rand.Float64()
}

func (p *Perceptron) feedforward() float64 {
    p.prediction = 0
    for i := 0; i < len(p.weights); i++ {
        p.prediction += p.inputLayer[i] * p.weights[i]
    }
    p.prediction += p.bias
    return p.prediction
}

func (p *Perceptron) backprop() {
    learningRate := 0.5
    error := p.target - p.prediction
    for i := 0; i < len(p.weights); i++ {
        p.weights[i] += learningRate * error * p.inputLayer[i]
    }
    p.bias += learningRate * error
}

func main() {
    p := Perceptron{}
    p.createPerceptron()

    for i := 0; i < 10000; i++ {
        p.target = p.inputLayer[0] + p.inputLayer[1]
        p.feedforward()
        p.backprop()
        if p.cost() < 0.0001 {
            p.inputLayer[0] = rand.Float64()
            p.inputLayer[1] = rand.Float64()
            p.target = p.inputLayer[0] + p.inputLayer[1]
            p.feedforward()
            p.ebl = p.cost()
            fmt.Println(p.ebl)
        }
    }
}
