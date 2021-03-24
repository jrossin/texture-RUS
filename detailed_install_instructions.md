
## SETTING UP YOUR ENVIRONMENT FOR RUNNING THE CODES (Windows/Anaconda)
1. Download Anaconda3
    - Ideally place in a folder with no spaces in it's $PATH$. (Eg. Do not use C:\Users\Lab Computer\Anaconda3); C:\Users\labcomp\Anaconda3 would be better
    - [anaconda download link](https://www.anaconda.com/products/individual)
    - Open Anaconda Prompt after installation has completed
2. Using github
    - clone both the SMCpy package and the Texture Inversion codes
    - [github download link](https://git-scm.com/downloads)
    - Open git bash
    - Navigate to a directory that you would like to be the "working directory" for these codes
    - in git bash, this is done using the "cd" command, followed by the path of interest; for example, cd C:\Users\working_directory_textureinversion
    - all git directories will now pop up as folders within the folder "working_directory_textureinversion"
    - run commands 'git clone https://github.com/nasa/SMCPy.git SMCPy', 'git clone https://github.com/jrossin/texture-RUS.git tex'
3. Navigate to the created folder from the end of step 2 (*working_directory_textureinversion*).
    - Create a virtual environment for python based on the appropriate *environment.yml* file in the git repo
    - if working with an anaconda installation in windows 'conda env create -f environmentwinconda.yml'
    - if working with an anaconda install on linux/ ubuntu 'conda env create -f environmentubuntu20.yml'
    - This will initialize a new conda environment with name py3smcpy (specified inside the virtual environment, can be changed if desired)
    - The *requirements.txt* file can alternatively be used to see the minimum versions of the packages needed to run this code
    - A less intensive option may be to simply to clone the git repos as-is, link them up correctly in conda, and continue through the steps to the bottom. When it comes time to run the model, simply install the packages that do not exist as errors occur.
4. Activate (enter) the virtual environment you created, and download these packages into it.
    - 'activate *virtualenvname*
    - Download Mpi4py if you desire to parallelize your simulations
    - 'conda install mpi4py' OR 'conda install -c intel mpi4py'
5. Link SMCpy directory to the anaconda path
    - Use the following command and replace *PATHTOGITDIRECTORY* with the location of the local SMCPy git repository
    - 'conda develop *PATHTO-SMCPy-LOCAL*'
    - This will create a *conda.pth* file within the virtual environment, linking the SMCPy file path to any file that runs in anaconda - adding other files to this text document will enable you to link directories to the system path in the future

## PREPARING A RUN/ RUNNING THE CODES
1. Ensure that all errors are cleared when navigating/ opening anaconda, these are generally system specific.

2. User input
    - handled within the *run_'modeltype'.py* file.
    - Changing the variable "problem_type" within the run_interface.py file will change from a forward model evaluation to an inverse model evaluation.
    - All variables relevant for either a forward or inverse model are in the header (dimensions, density), or within the respective 'if' statement for each condition
    - The *run_'modeltype'.py* file should be editable within any text editor, and simply saving it prior to running will update the values.
    - Running the actual file occurs in the anaconda prompt, once navigated to the folder that contains *run_'modeltype'.py*
3. Refer to readme.md for running the file within conda
