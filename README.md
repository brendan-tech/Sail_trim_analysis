# Sail Trim Analysis Readme

## Introduction

This repository contains a Python script for analyzing sail images to measure various properties such as camber, draft, and twist. The script processes sail images, identifies 3 draft lines, and calculates relevant sail characteristics. This readme file provides an overview of the code and its usage.

## Prerequisites

Before using this script, ensure that you have the following prerequisites installed:

- Python (3.6 or higher)
- OpenCV (Open Source Computer Vision Library)
- NumPy
- Matplotlib
- pandas

You can install these dependencies using pip:

```bash
pip install opencv-python numpy matplotlib pandas
```

## Usage

1. Clone the repository to your local machine.

2. Place the sail image you want to analyze in the same directory as the script, or provide the file path to the image in the 'img_inputs.txt' file.

3. Modify the 'img_inputs.txt' file with the desired configuration parameters. The available parameters are as follows:

   - `min_d`: Maximum pixel separation to count as the same object.
   - `p_min`: Minimum percentage of pixels to detect to consider an object of interest.
   - `ppl`: Points per line.
   - `ew`: Distance from ends of the stripe to start and end points.
   - `s_c`: Color of stripes to analyze (0 for orange, 1 for flo_yellow).
   - `path`: Path to the image file to be analyzed.

4. Run the script using the following command:

   ```bash
   python sail_image_analysis.py
   ```

5. The script will process the image, identify lines of interest, calculate sail characteristics, and display the results.

## Functionality

The script performs the following tasks:

1. Loads the sail image and applies color filtering to isolate the desired stripes.
2. Identifies lines of interest based on the configured color.
3. Calculates camber, draft, and twist for the identified lines.
4. Plots the results, including the identified lines, mean markers, and sail characteristics.

## Output

The script generates visual outputs that include plots showing the identified lines, mean markers, and sail characteristics (camber, draft, and twist). It also prints the calculated values to the console.

## Example

An example sail image analysis can be found in the 'example' directory, along with the input configuration file 'img_inputs.txt'.

## License

This code is provided under the [MIT License](LICENSE). Feel free to use and modify it for your own purposes.

## Author

- **Author**: Brendan

For any questions or issues, please contact the author @ brendan123lynch@gmail.com.

**Note**: This readme provides an overview of the code and its usage. For detailed explanations of the functions and algorithms used, please refer to the code comments and documentation within the script.
