package main

import (
	"fmt"
	"math/big"

	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// --------------------------------------------------
// CONFIG — tweak for benchmarking
// --------------------------------------------------
// Small demo network: input D, hidden H, output 1.
const (
	D      = 3  // input dimension
	H      = 2  // hidden dimension
	Samples = 2 // number of training samples
	Epochs  = 100 // number of epochs to unroll inside the circuit
)

// Learning rate (as scalar in the field)
var lr = big.NewInt(1) // simple lr=1 for demo; scale up/down as needed

var (
	fieldMod = ecc.BN254.ScalarField()
	inv2     = new(big.Int).ModInverse(big.NewInt(2), fieldMod)
	inv4     = new(big.Int).ModInverse(big.NewInt(4), fieldMod)
)

// Sigmoid approximation: σ(x) ≈ 1/2 + x/4; derivative ≈ 1/4.
// We hardcode these constants in the circuit via field elements.

// Circuit holds public data and public final weights; intermediate weights/acts are witness.
type TrainCircuit struct {
	// Public inputs: dataset and initial weights; final weights to assert.
	X        [Samples][D]frontend.Variable `gnark:",public"`
	Y        [Samples]frontend.Variable    `gnark:",public"`
	W1Init   [H][D]frontend.Variable       `gnark:",public"`
	W2Init   [H]frontend.Variable          `gnark:",public"`
	// Final weights kept private (not constrained as public outputs).
	W1Final  [H][D]frontend.Variable
	W2Final  [H]frontend.Variable
}

// helper: sigmoid approx: 0.5 + x/4
func sigmoidApprox(api frontend.API, x frontend.Variable) frontend.Variable {
	return api.Add(inv2, api.Mul(inv4, x))
}

// helper: sig' approx: 1/4 (constant)
func sigmoidDerivApprox(api frontend.API) frontend.Variable {
	return inv4
}

func (c *TrainCircuit) Define(api frontend.API) error {
	// Initialize weights from public initial
	var W1 [H][D]frontend.Variable
	var W2 [H]frontend.Variable
	for i := 0; i < H; i++ {
		for j := 0; j < D; j++ {
			W1[i][j] = c.W1Init[i][j]
		}
		W2[i] = c.W2Init[i]
	}

	lrVar := lr
	quarter := sigmoidDerivApprox(api) // 1/4

	// Unroll training for Epochs
	for epoch := 0; epoch < Epochs; epoch++ {
		// Accumulate gradients over all samples (full batch)
		var gradW1 [H][D]frontend.Variable
		var gradW2 [H]frontend.Variable
		// init grads to zero
		for i := 0; i < H; i++ {
			for j := 0; j < D; j++ {
				gradW1[i][j] = 0
			}
			gradW2[i] = 0
		}

		for s := 0; s < Samples; s++ {
			// Forward: hidden preact = W1 * x
			var hLin [H]frontend.Variable
			for i := 0; i < H; i++ {
				sum := frontend.Variable(0)
				for j := 0; j < D; j++ {
					sum = api.Add(sum, api.Mul(W1[i][j], c.X[s][j]))
				}
				hLin[i] = sum
			}
			// Hidden act
			var hAct [H]frontend.Variable
			for i := 0; i < H; i++ {
				hAct[i] = sigmoidApprox(api, hLin[i])
			}
			// Output: yhat = sum_i W2[i] * hAct[i]
			yhat := frontend.Variable(0)
			for i := 0; i < H; i++ {
				yhat = api.Add(yhat, api.Mul(W2[i], hAct[i]))
			}

			// Loss grad wrt yhat: dL/dyhat = 2*(yhat - y)
			dLdy := api.Mul(2, api.Sub(yhat, c.Y[s]))

			// Grad W2: dL/dW2[i] = dL/dyhat * hAct[i]
			for i := 0; i < H; i++ {
				gradW2[i] = api.Add(gradW2[i], api.Mul(dLdy, hAct[i]))
			}

			// Backprop to hidden: dL/dhAct[i] = dL/dyhat * W2[i]
			// dL/dhLin[i] = dL/dhAct[i] * sigmoid' ≈ dL/dyhat * W2[i] * 1/4
			var dLdhLin [H]frontend.Variable
			for i := 0; i < H; i++ {
				dLdhAct := api.Mul(dLdy, W2[i])
				dLdhLin[i] = api.Mul(dLdhAct, quarter)
			}
			// Grad W1: dL/dW1[i][j] = dL/dhLin[i] * x_j
			for i := 0; i < H; i++ {
				for j := 0; j < D; j++ {
					gradW1[i][j] = api.Add(gradW1[i][j], api.Mul(dLdhLin[i], c.X[s][j]))
				}
			}
		}

		// Weight update: W := W - lr * grad
		for i := 0; i < H; i++ {
			for j := 0; j < D; j++ {
				W1[i][j] = api.Sub(W1[i][j], api.Mul(lrVar, gradW1[i][j]))
			}
			W2[i] = api.Sub(W2[i], api.Mul(lrVar, gradW2[i]))
		}
	}

	return nil
}

// --------------------------------------------------
// Helper: build a tiny dataset and run the same training in Go
// --------------------------------------------------
func sigmoidApproxHost(x *big.Int, p *big.Int) *big.Int {
	// σ(x) ≈ 1/2 + x/4 mod p using modular inverses
	term := new(big.Int).Mul(x, inv4)
	term.Mod(term, p)
	out := new(big.Int).Add(inv2, term)
	out.Mod(out, p)
	return out
}

func trainHost(xData [Samples][D]*big.Int, yData [Samples]*big.Int, W1 [H][D]*big.Int, W2 [H]*big.Int, epochs int, p *big.Int) ([H][D]*big.Int, [H]*big.Int) {
	lrMod := new(big.Int).Set(lr)
	lrMod.Mod(lrMod, p)
	for ep := 0; ep < epochs; ep++ {
		// inv4 already global
		// zero grads
		var gW1 [H][D]*big.Int
		var gW2 [H]*big.Int
		for i := 0; i < H; i++ {
			for j := 0; j < D; j++ {
				gW1[i][j] = big.NewInt(0)
			}
			gW2[i] = big.NewInt(0)
		}
		for s := 0; s < Samples; s++ {
			var hLin [H]*big.Int
			for i := 0; i < H; i++ {
				sum := big.NewInt(0)
				for j := 0; j < D; j++ {
					tmp := new(big.Int).Mul(W1[i][j], xData[s][j])
					tmp.Mod(tmp, p)
					sum.Add(sum, tmp)
					sum.Mod(sum, p)
				}
				hLin[i] = sum
			}
			var hAct [H]*big.Int
			for i := 0; i < H; i++ {
				hAct[i] = sigmoidApproxHost(hLin[i], p)
			}
			yhat := big.NewInt(0)
			for i := 0; i < H; i++ {
				tmp := new(big.Int).Mul(W2[i], hAct[i])
				tmp.Mod(tmp, p)
				yhat.Add(yhat, tmp)
				yhat.Mod(yhat, p)
			}
			dLdy := new(big.Int).Sub(yhat, yData[s])
			dLdy.Mod(dLdy, p)
			dLdy.Mul(dLdy, big.NewInt(2))
			dLdy.Mod(dLdy, p)
			for i := 0; i < H; i++ {
				tmp := new(big.Int).Mul(dLdy, hAct[i])
				tmp.Mod(tmp, p)
				gW2[i].Add(gW2[i], tmp)
				gW2[i].Mod(gW2[i], p)
			}
			var dLdhLin [H]*big.Int
			for i := 0; i < H; i++ {
				dLdhAct := new(big.Int).Mul(dLdy, W2[i])
				dLdhAct.Mod(dLdhAct, p)
				dLdhLin[i] = new(big.Int).Mul(dLdhAct, inv4)
				dLdhLin[i].Mod(dLdhLin[i], p)
			}
			for i := 0; i < H; i++ {
				for j := 0; j < D; j++ {
					tmp := new(big.Int).Mul(dLdhLin[i], xData[s][j])
					tmp.Mod(tmp, p)
					gW1[i][j].Add(gW1[i][j], tmp)
					gW1[i][j].Mod(gW1[i][j], p)
				}
			}
		}
		// update
		for i := 0; i < H; i++ {
			for j := 0; j < D; j++ {
				tmp := new(big.Int).Mul(lrMod, gW1[i][j])
				tmp.Mod(tmp, p)
				W1[i][j].Sub(W1[i][j], tmp)
				W1[i][j].Mod(W1[i][j], p)
			}
			tmp := new(big.Int).Mul(lrMod, gW2[i])
			tmp.Mod(tmp, p)
			W2[i].Sub(W2[i], tmp)
			W2[i].Mod(W2[i], p)
		}
	}
	return W1, W2
}

func main() {
	fmt.Println("Compiling NN training circuit...")
	var circuit TrainCircuit
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		panic(err)
	}
	fmt.Println("Number of constraints:", ccs.GetNbConstraints())

	// Build tiny dataset and run host training for witness
	mod := ecc.BN254.ScalarField()
	var xData [Samples][D]*big.Int
	var yData [Samples]*big.Int
	// simple points
	xData[0] = [D]*big.Int{big.NewInt(1), big.NewInt(0), big.NewInt(1)}
	xData[1] = [D]*big.Int{big.NewInt(0), big.NewInt(1), big.NewInt(1)}
	yData[0] = big.NewInt(1)
	yData[1] = big.NewInt(0)

	var W1Init [H][D]*big.Int
	var W2Init [H]*big.Int
	for i := 0; i < H; i++ {
		for j := 0; j < D; j++ {
			W1Init[i][j] = big.NewInt(1) // small init
		}
		W2Init[i] = big.NewInt(1)
	}

	W1Final, W2Final := trainHost(xData, yData, W1Init, W2Init, Epochs, mod)

	// Fill witness/public
	var wit TrainCircuit
	for s := 0; s < Samples; s++ {
		for j := 0; j < D; j++ {
			wit.X[s][j] = new(big.Int).Set(xData[s][j])
		}
		wit.Y[s] = new(big.Int).Set(yData[s])
	}
	for i := 0; i < H; i++ {
		for j := 0; j < D; j++ {
			wit.W1Init[i][j] = new(big.Int).Set(W1Init[i][j])
			wit.W1Final[i][j] = new(big.Int).Set(W1Final[i][j])
		}
		wit.W2Init[i] = new(big.Int).Set(W2Init[i])
		wit.W2Final[i] = new(big.Int).Set(W2Final[i])
	}

	prover, err := frontend.NewWitness(&wit, ecc.BN254.ScalarField())
	if err != nil {
		panic(err)
	}
	public, err := prover.Public()
	if err != nil {
		panic(err)
	}

	pk, vk, err := groth16.Setup(ccs)
	if err != nil {
		panic(err)
	}
	proof, err := groth16.Prove(ccs, pk, prover)
	if err != nil {
		fmt.Println("Prove error:", err)
		return
	}
	if err := groth16.Verify(proof, vk, public); err != nil {
		fmt.Println("Verify error:", err)
	} else {
		fmt.Println("Proof verified: true")
	}
}
