# smart_shirt

## Code for data processing and ML implemenation of the smart-shirt project.

### Files 

- **main.py**: File containing main code for data processing and ML algorithm implementation.
- **smart_shirt_jupyter_nb.ipybn**: Jupyter notebook version of main.py. 
- **smart_shirt_environment.yml**: Virtual environment file to run jupyterlab with dependencies.


### Instructions to run the the jupyter notebook:
- Install Anaconda/Miniconda if you don't have it already: https://docs.conda.io/en/latest/miniconda.html 
- Clone this repository to your local.
- Create and activate a Python virtual environment:
    - If you installed Anaconda/Miniconda, use `conda` (on Windows, these commands should be run in **Anaconda Prompt**):

        ```shell
        $ cd smart_shirt-main
        ~/smart_shirt-main user$ conda env create -f smart_shirt_environment.yml
        ~/smart_shirt-main user$ conda activate smart_shirt
        ```

- Launch the notebook on JupyterLab:
    ```shell
    (smart_shirt) ~/smart_shirt-main user$ jupyter lab
    ``` 
