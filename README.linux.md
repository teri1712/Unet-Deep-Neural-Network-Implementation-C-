# Raw Implementation for UNet using C/C++

This project implements UNet and EfficientNet-UNet (Eff-UNet) models for image segmentation using LibTorch in C++. The project is designed to be built with CMake and requires OpenCV.

## Directory Structure

```
Unet/
├── build/                # CMake build directory
│   └── Release/          # Executables are generated here
├── dataset/
│   ├── final-dataset/    # Preprocessed images
│   ├── oversample/       # Oversampled images
│   └── validate/         # Validation images
│       ├── Team A/       # Images for IoU evaluation for class 'Team A'
│       └── ...
├── libtorch/             # LibTorch library
├── models/               # Trained model files (.pt)
├── pred/                 # Predicted images after each training epoch
├── stat/                 # Training statistics (CSV files)
├── CmakeLists.txt        # CMake configuration file
├── header.h              # Header file with all declarations
├── data.cpp              # Data loading and processing
├── models.cpp            # Model implementations (UNet, Eff-UNet)
├── train.cpp             # Training logic
├── utils.cpp             # Utility functions
├── TestModel.cpp         # Source for TestModel executable
├── TrainModel.cpp        # Source for TrainModel executable
├── build.sh              # Build script
├── train.sh              # Training script
└── test.sh               # Testing script
```

## Requirements

*   **Operating System:** Linux
*   **C++ Standard:** C++17
*   **Build System:** CMake
*   **Libraries:**
    *   OpenCV (latest version)
    *   LibTorch 2.2.2 (CPU version). You can download it from the official PyTorch website.

## Building the Project

1.  **Download and Extract LibTorch:**
    *   Download the LibTorch 2.2.2 CPU version for Linux from the [PyTorch website](https://pytorch.org/get-started/locally/).
    *   Extract the downloaded archive and place the `libtorch` directory inside the `Unet` directory, as shown in the directory structure above.

2.  **Run the Build Script:**
    *   Make the build script executable:
        ```bash
        chmod +x build.sh
        ```
    *   Execute the script, passing the path to your OpenCV installation:
        ```bash
        ./build.sh <path_to_your_opencv_lib>
        ```
        Replace `<path_to_your_opencv_lib>` with the actual path to your OpenCV library installation (e.g., `/usr/local/lib/cmake/opencv4`).

3.  **Executables:**
    *   After a successful build, you will find the `TrainModel` and `TestModel` executables in the `Unet/build/Release` directory.

## Usage

The provided `train.sh` and `test.sh` scripts simplify the execution process. Make sure they are executable first:
```bash
chmod +x train.sh
chmod +x test.sh
```

### `train.sh`

This script trains the UNet or Eff-UNet model.

**Usage:**

```bash
./train.sh <model_name> <learning_rate>
```

**Arguments:**

*   `<model_name>`: The name of the model to train (`Unet` or `EffUnet`).
*   `<learning_rate>`: The initial learning rate (e.g., `0.001`).

### `test.sh`

This script loads a pre-trained model and performs image segmentation.

**Usage:**

```bash
./test.sh <model_name> <path_to_model.pt> <input_image_path> <output_directory_path>
```

**Arguments:**

*   `<model_name>`: The name of the model to use (`Unet` or `EffUnet`).
*   `<path_to_model.pt>`: The path to the pre-trained model file (`.pt`).
*   `<input_image_path>`: The path to the input image.
*   `<output_directory_path>`: The path to the directory where the segmented output image will be saved.

## Pre-trained Models

Links to pre-trained UNet and Eff-UNet models are provided in the `pre-built.txt` file. You can download these models and use them with the `test.sh` script to perform image segmentation without retraining.
