# First, define the models and paths in a structured way
valPath = "/home/myid/bs83243/mastersProject/energy_constraint_ensemble/valFile/imagenet"
sortedValPath = "/home/myid/bs83243/mastersProject/energy_constraint_ensemble/ValFileSorted/imagenet"

# Create dictionary mapping model names to their file names
MODEL_FILES = {
    'densenet201': 'densenet201.pt',
    'resnet18': 'resnet18.pt',
    'resnet152': 'resnet152.pt',
    'resnext50': 'resnext50.pt',
    'vgg19_bn': 'vgg19_bn.pt',
    'efficientnet_b0': 'efficientnet_b0.pt',
    'inception_v3': 'inception_v3.pt',
    'squeezeNet1_1': 'squeezeNet1_1.pt',
    'alexnet': 'alexnet.pt',
    'vgg16': 'vgg16.pt'
}

# Load and sort all models
sorted_models = {}
reference_labels = None

for model_name, filename in MODEL_FILES.items():
    print(f"Processing {model_name}...")
    
    # Load model
    model_path = os.path.join(valPath, filename)
    val_file = torch.load(model_path, map_location="cuda:1")
    
    # Get labels and convert to numpy
    label_vectors = val_file["labelVectors"].cpu().numpy()
    
    # Store reference labels from first model
    if reference_labels is None:
        reference_labels = label_vectors
        sorted_indices = np.argsort(label_vectors)
    
    # Sort using the same indices for all models
    sorted_prediction_vectors = val_file['predictionVectors'][sorted_indices]
    sorted_label_vectors = val_file['labelVectors'][sorted_indices]
    
    # Create sorted dictionary
    sorted_dict = {
        'predictionVectors': sorted_prediction_vectors,
        'labelVectors': sorted_label_vectors
    }
    
    # Save sorted model
    output_path = os.path.join(sortedValPath, filename)
    torch.save(sorted_dict, output_path)
    print(f"Saved sorted {model_name} to {output_path}")
    
    # Store for verification
    sorted_models[model_name] = sorted_label_vectors

# Verify all models are sorted the same way
print("\nVerifying sort consistency...")
reference_model = list(sorted_models.keys())[0]
for model_name in sorted_models:
    if model_name != reference_model:
        match = np.all(sorted_models[reference_model] == sorted_models[model_name])
        print(f"Labels match between {reference_model} and {model_name}: {match}")