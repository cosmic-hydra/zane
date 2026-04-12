// package main - CLI tool for rapid ADMET predictions
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"zane/tools/go/admet"
)

func main() {
	// Command-line flags
	molWeight := flag.Float64("mw", 0, "Molecular weight (Da)")
	logP := flag.Float64("logp", 0, "LogP partition coefficient")
	hbd := flag.Int("hbd", 0, "Hydrogen bond donors")
	hba := flag.Int("hba", 0, "Hydrogen bond acceptors")
	rotBonds := flag.Int("rb", 0, "Rotatable bonds")
	batch := flag.String("batch", "", "JSON file with batch properties")
	output := flag.String("output", "json", "Output format (json|csv|text)")

	flag.Parse()

	if *batch != "" {
		handleBatch(*batch, *output)
		return
	}

	if *molWeight == 0 {
		fmt.Fprintf(os.Stderr, "Usage: admet -mw 350.0 -logp 3.5 -hbd 2 -hba 5 -rb 8\n")
		os.Exit(1)
	}

	// Single prediction
	props := admet.LipinkskiProperties{
		MolecularWeight:   *molWeight,
		LogP:              *logP,
		HydrogenDonors:    *hbd,
		HydrogenAcceptors: *hba,
		RotatableBonds:    *rotBonds,
	}

	// Check Lipinski
	lipinskiResult := admet.CheckLipinski(props)

	// Predict ADMET
	admetScore := admet.PredictADMET(props)

	// Output results
	result := map[string]interface{}{
		"lipinski": lipinskiResult,
		"admet": map[string]float64{
			"overall":      admetScore.Overall,
			"absorption":   admetScore.Absorption,
			"distribution": admetScore.Distribution,
			"metabolism":   admetScore.Metabolism,
			"excretion":    admetScore.Excretion,
			"toxicity":     admetScore.Toxicity,
		},
		"sa_score": admet.SyntheticAccessibility(float64(*rotBonds + *hba)),
	}

	if *output == "json" {
		jsonBytes, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			log.Fatalf("JSON marshaling error: %v", err)
		}
		fmt.Println(string(jsonBytes))
	} else {
		fmt.Printf("Lipinski Pass: %v\n", lipinskiResult["passes"])
		fmt.Printf("ADMET Score: %.3f\n", admetScore.Overall)
		fmt.Printf("SA Score: %.1f\n", result["sa_score"])
	}
}

func handleBatch(batchFile string, output string) {
	data, err := os.ReadFile(batchFile)
	if err != nil {
		log.Fatalf("Error reading batch file: %v", err)
	}

	var propsList []admet.LipinkskiProperties
	if err := json.Unmarshal(data, &propsList); err != nil {
		log.Fatalf("Error parsing JSON: %v", err)
	}

	scores := admet.BatchADMETPredict(propsList)

	if output == "json" {
		jsonBytes, err := json.MarshalIndent(scores, "", "  ")
		if err != nil {
			log.Fatalf("JSON marshaling error: %v", err)
		}
		fmt.Println(string(jsonBytes))
	} else {
		fmt.Println("Molecule\tADMET\tStatus")
		for i, score := range scores {
			status := "PASS"
			if score.Overall < 0.5 {
				status = "FAIL"
			}
			fmt.Printf("Mol_%d\t%.3f\t%s\n", i, score.Overall, status)
		}
	}
}
