
from transformers import pipeline

# Create the feature extraction pipeline
feature_extraction_pipeline = pipeline(
    task="feature-extraction",
    model="model_name",
    device=0  # Use 0 for GPU, -1 for CPU
)

# List of input sequences
input_texts = [
    "Input text 1",
    "Input text 2",
    "Input text 3",
    # Add more input texts here
]

# Tokenize and pad the input sequences using the tokenizer
# Set padding=True to pad the sequences to the same length
# Set truncation=True to truncate sequences longer than the maximum length
# Set max_length to the desired maximum length of the sequences
tokenized_inputs = feature_extraction_pipeline.tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=128,  # Set your desired maximum sequence length
    return_tensors="pt"  # Return PyTorch tensors for easy processing
)

# Move the tokenized inputs to the appropriate device (CPU or GPU)
tokenized_inputs = {k: v.to(feature_extraction_pipeline.device) for k, v in tokenized_inputs.items()}

# Process the padded sequences in a batch with the pipeline
features_batch = feature_extraction_pipeline(**tokenized_inputs)