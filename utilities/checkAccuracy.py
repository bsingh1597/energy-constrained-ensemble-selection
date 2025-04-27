import torch
import sys
import os
from typing import List, Optional

# Project-specific imports
FRAMEWORKS_PATH = '/home/myid/bs83243/mastersProject/EnsembleBench/EnsembleBench/frameworks'
if os.path.abspath(FRAMEWORKS_PATH) not in sys.path:
    sys.path.append(os.path.abspath(FRAMEWORKS_PATH))

from pytorchUtility import (calAccuracy)

def load_and_evaluate_model(model_path: str) -> None:
    """
    Loads a PyTorch model, calculates accuracy, and prints results.

    Args:
        model_path (str): Path to the PyTorch model file.
    """

    predictionVectorsList: List[torch.Tensor] = []
    labelVectorsList: List[torch.Tensor] = []
    tmpAccList: List[torch.Tensor] = []

    try:
        prediction = torch.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error: Could not load model from {model_path}.  Details: {e}")
        return

    if 'predictionVectors' not in prediction or 'labelVectors' not in prediction:
        print(f"Error: The loaded model from {model_path} does not contain 'predictionVectors' and/or 'labelVectors'.")
        return

    predictionVectors = prediction['predictionVectors']
    predictionVectorsList.append(torch.nn.functional.softmax(predictionVectors, dim=-1))
    labelVectors = prediction['labelVectors']
    labelVectorsList.append(labelVectors.cpu())
    tmpAcc = calAccuracy(predictionVectors, labelVectors)[0].cpu()  # Store the accuracy directly
    tmpAccList.append(tmpAcc) # append the accuracy to the tmpAccList


    print(f"Prediction Vectors Size: {predictionVectors.size()}")
    print(f'Length of tmpAccList: {len(tmpAccList)}')
    print(f'Accuracy: {tmpAcc}')

def main(model_path: Optional[str] = None):
    """
    Main function to load and evaluate a model.

    Args:
        model_path (str, optional): Path to the PyTorch model file.
                                     If None, the function will exit.
    """
    if not model_path:
        print("Error: Model path must be provided as a command-line argument.")
        sys.exit(1)  # Exit with a non-zero code to indicate an error

    load_and_evaluate_model(model_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        main(model_path)
    else:
        print("Usage: python script_name.py <model_path>")
        sys.exit(1)