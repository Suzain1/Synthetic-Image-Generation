using Flux
using NPZ

# Define a simple CNN model
model = Chain(
    Conv((3, 3), 1 => 16, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 16 => 32, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(32 * 54 * 54, 2),       # Fully connected layer (assuming 2 output classes)
    softmax
)

# Load the preprocessed image from the .npy file
function load_preprocessed_image(file_path)
    img_array = npzread(file_path)
    img_array = reshape(img_array, 224, 224, 1, 1)  # Reshape for Flux (H, W, Channels, Batch)
    return Float32.(img_array)
end

# Path to saved .npy file
image_path = "preprocessed_image_1.npy"

# Load image and run forward pass
input_image = load_preprocessed_image(image_path)
output = model(input_image)

println("Model Output:\n", output)

#=
flux_model:
- Julia version: 
- Author: suzai
- Date: 2025-02-06
=#
