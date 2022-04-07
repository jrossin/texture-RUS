# texture-RUS
Inverse determination of texture and elastic constants from resonant frequencies (gathered by RUS) by pairing with the inverse solving python package SMCPy
Published at https://doi.org/10.1016/j.actamat.2021.117287
"Rossin, J., Leser, P., Pusch, K., Frey, C., Murray, S. P., Torbet, C. J., ... & Pollock, T. M. (2021). Bayesian inference of elastic constants and texture coefficients in additively manufactured cobalt-nickel superalloys using resonant ultrasound spectroscopy. Acta Materialia, 220, 117287."
Written by Jeff Rossin (jrossin@engineering.ucsb.edu)

## clone both the SMCpy package and the Texture Inversion codes
    - 'git clone https://github.com/nasa/SMCPy.git'
    - 'git clone https://github.com/jrossin/texture-RUS.git'

## RUNNING THE CODES
1. All models: required information
    - Sample parallelepiped dimensions(x,y,z), in units of meters
    - Polynomial order of the approximation (8-16, even numbers only), 10 or 12 is a good place to start
    - Density of the material in question (units of kg/m^3)
    - For a texture forward or inverse model, the 3 single crystal elastic constants of the microscopically cubic material, inputted as sc11 (C11), sc12 (C12), and sc44(C44)

2. Running a forward model
    - cd to the  directory with the *run_texture_forward.py* or  *run_cij_forward.py* file.
    - Specify:
    - List of elastic constants (Cij, in GPa) or texture coefficients (Vijkl,can be converted from Clmn), to simulate the resultant resonance frequencies of the specimen
    - Number of desired output frequencies (50-100 is generally sufficient)
    - 'python run_interface.py'

3. Running an inverse model:
    - cd to the directory with the *run_texture_inverse.py* or  *run_cij_inverse.py* file.
    - List of experimental resonance frequencies in a file (feel free to copy and edit an existing one, such as freq_R2_CoNi_20deg_ab_70_polished), with your own list of resonance frequencies in kHz. The first line of the file should contain the number of resonance frequencies you want to use for your inversion. Update the 'filename' variable in run_interface.py to your new filename
    - Ranges of of elastic constants (Cij, in GPa) or texture coefficients (Clmn or Vijkl) to solve for during the inference - this is contained in the 'params' dict()
    - The number of chains (independent simulations) to run. Typically 1 is sufficient with SMC, though more are possible for parameter studies
    - SMCPy specific parameters such as particle count, timesteps, and mcmc (markoc chain monte carlo) steps must be specified. These can be understood from the publication [Efficient Sequential Monte-Carlo Samplers for Bayesian Inference](10.1109/TSP.2015.2504342) by Nguyen et al.; They are also well summarized within the SMCPy package instructions. The parameters are linked to the complexity of your inversion, and higher numbers require greater time but yield greater precision of results. A good place to start is particles-3000, timesteps-30, MCMCsteps - 6.
4. Parallel command for the inverse model -will utilize the mpi4py package and across a number of cores (*number of cores* as an integer)
    'mpiexec -n *number of cores* python run_interface.py'

## full instructions for installing and running the git repos/ anaconda are given in 'detailed_install_instructions.md'
