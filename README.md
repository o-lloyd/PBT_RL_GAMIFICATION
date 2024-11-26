# Proton Beam Therapy Reinforcement Learning

## Overview

This repository provides a framework for setting up reinforcement learning (RL) on patient data to create personalized treatment plans using Proton Beam Therapy (PBT). The current implementation includes data processing and beam modeling components essential for simulating and analyzing proton beam interactions with patient-specific density maps.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing](#data-processing)
  - [Running Beam Models](#running-beam-models)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)
- [Quick Start Example](#quick-start-example)
- [Contact](#contact)

## Features

- **Data Processing:** Converts and prepares DICOM data for use in beam modeling.
- **Beam Modeling:** Simulates proton beam interactions with patient-specific density maps and visualizes dose distributions.
- **Extensible Framework:** Designed to incorporate reinforcement learning algorithms for optimizing treatment plans.

## Requirements

- **Operating System:** Windows
- **Python Version:** 3.8 or higher
- **Libraries:**
  - `numpy`
  - `matplotlib`
  - `pydicom`
  - `opencv-python`
  - `scipy`
  - `pickle`
  - `protonbeam` (custom module)

## Installation

1. **Clone the Repository**

    ```bash:clone_repository.sh
    git clone https://github.com/yourusername/proton-beam-therapy-rl.git
    cd proton-beam-therapy-rl
    ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

    ```bash:setup_virtualenv.sh
    python -m venv venv
    venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash:install_dependencies.sh
    pip install -r requirements.txt
    ```

    *If `requirements.txt` is not available, you can install the necessary libraries manually:*

    ```bash:manual_install.sh
    pip install numpy matplotlib pydicom opencv-python scipy
    ```

    *Ensure that the custom `ProtonBeam` module is available in your Python path.*

## Usage

### Data Processing

Before running the beam simulation, you need to process the patient DICOM data.

1. **Prepare DICOM Data**

    Ensure that your DICOM files are placed in the appropriate directory as specified in `data_processing/data_processing.py`. By default, the base path is set to:

    ```python:data_processing/data_processing.py
    base_path = r"data"
    ```

2. **Run Data Processing Script**

    Execute the `data_processing/data_processing.py` script to process the DICOM data and generate the necessary pickle files.

    ```bash:data_processing/data_processing.py
    python data_processing/data_processing.py
    ```

    This will create processed data files, such as:

    - `pickles/example_dicom_processed_data.pkl`

### Running Beam Models

After processing the data, you can run the beam simulation to generate and visualize proton beam dose distributions.

1. **Run Beam Test Script**

    Execute the `test_beam.py` script to simulate the proton beam and visualize the results.

    ```bash:test_beam.py
    python test_beam.py
    ```

    This script performs the following:

    - Loads the processed DICOM and beam data.
    - Simulates a proton beam at specified coordinates and angle.
    - Visualizes the dose distribution on the density map.
    - Generates dose vs. water equivalent distance plots.

## File Structure
```bash:file_structure.sh
   

proton-beam-therapy-rl/
│
├── data_processing/
│ └── data_processing.py
│
├── pickles/
│ ├── example_dicom_processed_data.pkl
│ └── beam_data.pkl
│
├── structures_to_display.py
│
├── test_beam.py
│
├── ProtonBeam/
│ ├── init.py
│ └── ProtonBeamV4.py
│
├── requirements.txt
│
└── README.md

```


- **data_processing/**: Contains the data processing scripts.
- **pickles/**: Stores processed data files.
- **structures_to_display.py**: Defines structures for display purposes.
- **test_beam.py**: Script for beam simulation and visualization.
- **ProtonBeam/**: Custom module for proton beam modeling.
- **requirements.txt**: Lists Python dependencies.
- **README.md**: Project documentation.


## License

This project is licensed under the [MIT License](LICENSE).

## Quick Start Example

1. **Run Data Processing**

    ```bash:data_processing/data_processing.py
    python data_processing/data_processing.py
    ```

2. **Run Beam Simulation**

    ```bash:test_beam.py
    python test_beam.py
    ```

Ensure that all dependencies are installed and the data paths in the scripts are correctly set according to your system.



For further information, please contact [hayden@htibbals.com](mailto:hayden@htibbals.com).