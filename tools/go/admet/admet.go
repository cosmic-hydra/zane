// Package admet provides high-performance ADMET prediction utilities.
// Designed for rapid molecular property screening and batch processing.
package admet

import (
	"fmt"
	"math"
)

// LipinkskiProperties represents Lipinski's Rule of Five molecular properties.
type LipinkskiProperties struct {
	MolecularWeight    float64
	LogP               float64
	HydrogenDonors     int
	HydrogenAcceptors  int
	RotatableBonds     int
}

// CheckLipinski checks if a molecule complies with Lipinski's Rule of Five.
func CheckLipinski(props LipinkskiProperties) map[string]interface{} {
	violations := []string{}

	if props.MolecularWeight > 500 {
		violations = append(violations, "molecular_weight > 500")
	}
	if props.LogP > 5 {
		violations = append(violations, "logp > 5")
	}
	if props.HydrogenDonors > 5 {
		violations = append(violations, "h_donors > 5")
	}
	if props.HydrogenAcceptors > 10 {
		violations = append(violations, "h_acceptors > 10")
	}

	return map[string]interface{}{
		"passes":          len(violations) == 0,
		"violations":      violations,
		"num_violations":  len(violations),
		"properties":      props,
	}
}

// ADMETScore represents an ADMET prediction score with components.
type ADMETScore struct {
	Overall      float64
	Absorption   float64
	Distribution float64
	Metabolism   float64
	Excretion    float64
	Toxicity     float64
}

// PredictADMET predicts ADMET score from molecular properties.
// Uses optimized vectorized computation for speed.
func PredictADMET(props LipinkskiProperties) ADMETScore {
	// Normalize properties to 0-1 range
	molWtScore := clamp(1.0 - (props.MolecularWeight / 600.0), 0, 1)
	logPScore := clamp(1.0 - (math.Abs(props.LogP - 3.5) / 5.0), 0, 1)
	donorScore := clamp(1.0 - (float64(props.HydrogenDonors) / 6.0), 0, 1)
	acceptorScore := clamp(1.0 - (float64(props.HydrogenAcceptors) / 12.0), 0, 1)
	rotScore := clamp(1.0 - (float64(props.RotatableBonds) / 20.0), 0, 1)

	// Component scores
	absorption := (molWtScore + donorScore) / 2.0
	distribution := (logPScore + acceptorScore) / 2.0
	metabolism := 0.7 + (0.3 * logPScore) // LogP influences metabolism
	excretion := 0.75 + (0.25 * rotScore) // Rotatable bonds affect excretion
	toxicity := math.Min(logPScore, acceptorScore)

	// Overall score (weighted average)
	overall := (0.25*absorption + 0.20*distribution + 0.20*metabolism +
		0.20*excretion + 0.15*toxicity)

	return ADMETScore{
		Overall:      overall,
		Absorption:   absorption,
		Distribution: distribution,
		Metabolism:   metabolism,
		Excretion:    excretion,
		Toxicity:     toxicity,
	}
}

// SyntheticAccessibility estimates how easy a molecule is to synthesize.
// Returns score 1-10 (1=difficult, 10=easy).
func SyntheticAccessibility(complexity float64) float64 {
	// Sigmoid-based transformation
	sa := 10.0 / (1.0 + math.Exp((complexity-5.0)/2.0))
	return clamp(sa, 1.0, 10.0)
}

// TanimotoSimilarity computes Tanimoto similarity between fingerprints.
func TanimotoSimilarity(fp1, fp2 []float64) (float64, error) {
	if len(fp1) != len(fp2) {
		return 0, fmt.Errorf("fingerprint length mismatch: %d vs %d", len(fp1), len(fp2))
	}

	var intersection, unionSum float64
	for i := range fp1 {
		min := math.Min(fp1[i], fp2[i])
		max := math.Max(fp1[i], fp2[i])
		intersection += min
		unionSum += max
	}

	if unionSum == 0 {
		return 0, nil
	}

	return intersection / unionSum, nil
}

// BatchADMETPredict predicts ADMET scores for multiple molecules efficiently.
func BatchADMETPredict(propsList []LipinkskiProperties) []ADMETScore {
	scores := make([]ADMETScore, len(propsList))
	for i, props := range propsList {
		scores[i] = PredictADMET(props)
	}
	return scores
}

// Helper function to clamp values between min and max
func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
