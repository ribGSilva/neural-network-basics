package main_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNan(t *testing.T) {
	naN := math.NaN()

	result := naN * 0

	assert.True(t, result == 0)
}
