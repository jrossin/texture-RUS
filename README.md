# texture-RUS
Inverse determination of texture and elastic constants from resonant frequencies (gathered by RUS) by pairing with the inverse solving python package SMCPy

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
- handled within the *run_interface.py* file.
- Changing the variable "problem_type" within the run_interface.py file will change from a forward model evaluation to an inverse model evaluation.
- All variables relevant for either a forward or inverse model are in the header (dimensions, density), or within the respective 'if' statement for each condition
- The *run_interface.py* file should be editable within any text editor, and simply saving it prior to running will update the values.
- Running the actual file occurs in the anaconda prompt, once navigated to the folder that contains *run_interface.py*
3. All models: required information
    - Sample parallelepiped dimensions(x,y,z), in units of meters
    - Polynomial order of the approximation (8-16, even numbers only), 10 or 12 is a good place to start
    - Density of the material in question (units of kg/m^3)
    - For a texture forward or inverse model, the 3 single crystal elastic constants of the microscopically cubic material, inputted as sc11 (C11), sc12 (C12), and sc44(C44)

4. Running a forward model
    - cd to the  directory with the *run_interface.py* file.
    - Specify:
    - List of elastic constants (Cij, in GPa) or texture coefficients (Vijkl,can be converted from Clmn), to simulate the resultant resonance frequencies of the specimen
    - Number of desired output frequencies (50-100 is generally sufficient)
    - 'python run_interface.py'

5. Running an inverse model:
    - cd to the  directory with the *run_interface.py* file.
    - List of experimental resonance frequencies in a file (feel free to copy and edit an existing one, such as freq_R2_CoNi_20deg_ab_70_polished), with your own list of resonance frequencies in kHz. The first line of the file should contain the number of resonance frequencies you want to use for your inversion. Update the 'filename' variable in run_interface.py to your new filename
    - Ranges of of elastic constants (Cij, in GPa) or texture coefficients (Clmn or Vijkl) to solve for during the inference - this is contained in the 'params' dict()
    - The number of chains (independent simulations) to run. Typically 1 is sufficient with SMC, though more are possible for parameter studies
    - SMCPy specific parameters such as particle count, timesteps, and mcmc (markoc chain monte carlo) steps must be specified. These can be understood from the publication [Efficient Sequential Monte-Carlo Samplers for Bayesian Inference](10.1109/TSP.2015.2504342) by Nguyen et al.; They are also well summarized within the SMCPy package instructions. The parameters are linked to the complexity of your inversion, and higher numbers require greater time but yield greater precision of results.
6. Parallel command for the inverse model (this can be computationally intensive)
    This will utilize the mpi4py package and a set number of cores (*number of cores* as an integer)
    'mpiexec -n *number of cores* python run_interface.py'

