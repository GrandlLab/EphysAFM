# EphysAFM

## Description
This is a python package written to analyze atomic force microscopy and electrophysiology data collected in tandem from the manuscript "The energetics of rapid cellular mechanotransduction". An example analysis pipeline can be found in the Jupyter notebook file "EphysAFM.ipynb".

![EphysAFM](/assets/afm-ephys.gif)

## Prerequisites

Before you begin make sure you have Python 3.7 or higher. It may work with other versions but I have not tested it with these particular packages.

## Installation

```
git clone https://github.com/neuro-myoung/EphysAFM.git
```

The environment.yml file has all the necessary packages and appropriate versions. Using conda create a virtual environment with the necessary packages using the environment.yml file by going to the folder in your command line and typing the following command:

```
conda env create -f environment. yml
```

The environment will be named ephysafm by default. Activate the new environment whenever you want to run the program by entering in your command line the following:

```
conda activate ephysafm
```

Descriptions of individual functions can be found in the different modules `loadFile.py`,`plotData.py`, `preprocess.py`, and `summarize.py`.  

## Contributing
To contribute to **sader_calibration**, follow these steps:

1. Fork this repository.
2. Create a branch: git checkout -b *branch_name*.
3. Make your changes and commit them: git commit -m '*commit_message*'
4. Push to the original branch: git push origin *project_name* *location*
5. Create a pull request.

Alternatively see the GitHub [documentation](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) on creating a pull request.

## Contributors

[@neuro-myoung](https://github.com/neuro-myoung)

## Contact

If you want to contact me you can reach me at michael.young@duke.edu

## License
This project uses an [MIT License](https://opensource.org/licenses/MIT)