## Overview

This pipeline processes .sxm files through several analysis stages:

1. **Data Loading** - Load and parse SXM files
2. **Feature Extraction** - Calculate image properties and statistics
3. **Clustering** - Group similar images using Gaussian Mixture Models
4. **Image Processing** - Apply cluster-specific filters and corrections
5. **Structural Analysis** - Extract molecular backbones and measure geometric properties

## Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   ```

2. **Create a virtual environment (your-name is of your own choice)**
   ```bash
   python -m venv your-name
   your-name\Scripts\activate.bat
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Setting Up Paths

The pipeline uses a `paths.json` file to manage data locations. Create this file in your project root:

```json
{
  "input_paths": {
    "general_path": "y://DATA//STM//STM_DATA//"
  },
  "output_paths": {
    "saving_path": "Z:/Results/Batch_processing"
  },
  "info_folder": {
  "info_path":"(NKN)2//20250706_LHe_Ag111"
  }
}
```

**Important**: Update these paths to match your system:
- `input_paths`: Where your STM data files (.sxm) are stored
- `output_paths`: Where you want processed results saved
- `info_folder`: Specific experiment folder within base_data



## Usage

### Basic Usage

1. **Prepare your data**
   - Ensure your .sxm files are in the directory specified in `paths.json`
   - Update the `current_experiment` path in `paths.json`

2. **Try the jupyter notebook**


3. **Check results**
   - Results will be saved in your desired destination
   - Each cluster gets its own folder with visualizations

### Output Structure

The pipeline creates the following output structure:
```
results/current_experiment/
├── cluster_0/
│   ├── classified_unprocessed_0.png
│   ├── classified_processed_0.png
│   ├── otsu_background_0.png
│   ├── distance_transform_0.png
│   └── backbones_0.png
├── cluster_1/
│   └── ...
├── processed_data.pkl
└── cluster_dict.pkl
```

### Understanding the Outputs

- **classified_unprocessed**: Raw images grouped by similarity
- **classified_processed**: Images after applying cluster-specific filters
- **otsu_background**: Background-removed images using Otsu thresholding
- **distance_transform**: Distance transform visualizations
- **backbones**: Molecular backbone extraction and measurements
- **processed_data.pkl**: Complete processed dataset
- **cluster_dict.pkl**: Cluster assignments

## Key Features

### Automated Clustering
- Uses Gaussian Mixture Models 

### Image Processing
- Automatic plane correction and drift compensation
- Adaptive filtering based on image characteristics
- Background removal and noise reduction

### Structural Analysis
- Molecular backbone extraction using distance transforms
- Geometric measurements (length, end-to-end distance, contour ratios)
- Physical unit conversions (pixels to nanometers)

## Troubleshooting

### Common Issues

1. **"File not found" errors**
   - Check your `paths.json` file paths
   - Ensure the directory structure matches expectations
   - Verify .sxm files exist in the specified location

2. **Import errors**
   - Make sure you installed all requirements: `pip install -r requirements.txt`
   - Activate your virtual environment
   - Check Python version compatibility

3. **Memory issues with large datasets**
   - Process smaller batches of files
   - Reduce image resolution in processing steps
   - Close unnecessary applications

4. **Empty clusters or poor clustering**
   - Check if your images have sufficient variation
   - Adjust clustering parameters in the code
   - Ensure images are properly loaded (not corrupted)

## Notes for New Users

- **Start Small**: Try with 5-10 images first to understand the workflow
- **Check Paths**: Most issues stem from incorrect path configuration
- **Virtual Environment**: Always use a virtual environment to avoid conflicts
- **Data Backup**: Keep backups of original data files
- **Patience**: Large datasets can take significant time to process

## Support

For questions or issues:
1. Check this README first
2. Review error messages carefully
3. Verify your environment setup
4. Test with minimal data first
