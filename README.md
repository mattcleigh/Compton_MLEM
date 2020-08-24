-------------------------------------------------------------------

     ============================================================
      UCT_MLEM - A MLEM Compton Camera Image Reconstruction Code 
     ============================================================

                            README FILE
                            -----------

 This GPU-based image reconstruction code takes data produced by a Compton
 camera and uses the MLEM image reconstruction technique to produce three-
 dimensional images of the gamma source.

-------------------------------------------------------------------

Primary Author: Matt Leigh - University of Cape Town
Supervisor: Steve Peterson - University of Cape Town
                      
Latest revision: M.Leigh, 24 May 2018
Written and compiled using NVCC and CUDA 8.0 (released July 2017)

-------------------------------------------------------------------

 1- PRE-REQUISITES

    This code was written using the CUDA Toolkit developed by NVIDIA to run code on their
    proprietary NVIDIA GPUs.  There are two primary requirements for running the UCT_MLEM
    code.
    
    1. An NVIDIA GPU card - A complete of compatible NVIDIA GPU can be found here: 
    https://developer.nvidia.com/cuda-gpus
    
    2. A functioning installation of the CUDA Toolkit.  This code was written using the CUDA 8.0
    Toolkit (Released July 2017).  There have been subsequent updates to the CUDA Toolkit, 
    and the UCT_MLEM code should be compatible with future versions of CUDA.  The CUDA
    Toolkit can be downloaded here: https://developer.nvidia.com/cuda-downloads
    

 2- CUDA INSTALLATION

    Installing the CUDA Toolkit can be tricky, so here are a few tips that might help:

    1. Make sure the NVIDIA drivers are up to date and functioning properly.
    
    The NVIDIA drivers can be found on the NVIDIA website (link below).  This is useful for determining the 
    most recent version of the drivers for your particular GPU card, but might not be
    the best way of updating the drivers for your particular system.
    
    For Ubuntu, I found the following instructions most helpful.
    
    - Add the following repository - it provides open-source version of the NVIDIA drivers
      % sudo add-apt-repository ppa:graphics-drivers

    - Update NVIDIA drivers using System Settings
      % System Settings -> Software & Updates -> Additional Drivers
        - After it loads the drivers, there should be an open source version of the latest nvidia driver
        - For example: Using NVIDIA binary driver - version 390.59 from nvidia-390 (open source)
      % Apply Changes

    - Reboot and check NVIDIA drivers functioning properly - try the following commands from
        command line.  If any of them fail, then the installation did not work.  Although,
        if it fails, you will probably be alerted sooner because your screen will not look right.
      + cat /proc/driver/nvidia/version
      + nvidia-smi
      + sudo lshw -c video
    
    - If failed, purge drivers, reboot and start-over [try something different]
      % sudo apt-get remove --purge nvidia-*
    
    - If it works, then you are ready for the next step.

    2. Install on the CUDA Toolkit and not CUDA Toolkit with drivers.

    The official NVIDIA install instructions (link below) recommend installing the CUDA Toolkit
    with the drivers, but I found that this just kills the NVIDIA drivers, particularly if 
    the version aren't the same, so installing just the toolkit makes the most sense.
    
    In Ubuntu, use this command
      % sudo apt-get install cuda-toolkit-8-0
    instead of 
      % sudo apt-get install cuda-8-0
      
    The best way to test your CUDA installation is to run the sample codes, but the quickest way
    is to run the compiler (nvcc) using:
      % nvcc -V
    If you get something back, then it is working.  Also a good idea to run the NVIDIA
    drivers test again to make sure that those weren't broken during install.

    3. Resources

    NVIDIA Drivers: http://www.nvidia.com/Download/index.aspx?lang=en-us.
    NVIDIA Install Instructions: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
    The CUDA Toolkit can be downloaded here: https://developer.nvidia.com/cuda-downloads


 3- UCT_MLEM STRUCTURE

    There are currently three versions of the UCT_MLEM code available in this repository. They all include the following key steps:
    - Import the input parameters from a configuration Setup.txt file. The file is explained in more detail below.
    - Volxelise the source space.
    - Calculate iteration 0 by using back projection
    - Import all the data from the compton camera and store each scattering event as a row in a "cone matrix".
    - Calculate the MLEM system matrix by integrating the cone-of-origin pdfs through each voxel
    - Iterate the voxelised source space using the MLEM algorithm.
    - Apply a noise reducing cut-off at the end of each iteration.
    - Store the estimated gamma distribution in a CSV or DAT file.
    
    Importing system parameters, scattering data and saving the output is all done on the CPU. Calculating the system matrix,
    performing the MLEM iteration and alpplying noise reducing cut-off are steps done in parallel on the GPU. Therefore, there is a lot of 
    memory transfer between the CPU and GPU during there execution of this code, where the requred objects are copied across. 
    
    Each of the three versions listed in the repository differ slightly in their approach. However, only the first two listed here are in 
    working condition. The other one is still under construction. 
    
    MLEM_Reformed                     - This is the standard method where the system matrix is calculated by integrating the scattering pdf
                                        through each voxel. The scattering pdf is generated using a gaussian distribution centered at the
                                        measured scattering angle and uses its uncertainty for the standard devieation.
                                        The voxel shaped "reformed" into a spherical co-ordinate volume unit, making integration faster.
                                        The entire system matrix is stored on the GPU, so memory costs will make this method not ideal for 
                                        any system with less than 8G of GRAM. Even then, resolution is capped pretty low. 
                                       
    MLEM_NO_MATRIX                    - This is the advised method. Here the system matrix is never stored and 
                                        voxel-cone interactions are recalculated on the fly during iteration. To save time there is no
                                        integration, rather the pdf is evaluated at a single point at the center of the voxels. If the 
                                        voxels are small enough, this does not effect reconstruction. The code executes only slightly slower
                                        than the matrix method and is capable of much higher resolutions. 
    
    (under construction)
    MLEM_Reformed_Energy_Integration  - This method is the same as the MLEM_Reformed one, exept the pdf is generated by assuming a
                                        gaussian error on each energy deposition rather than a gaussian on the scattering angle. 
                                        Currently having issues with the integration. May be scrapped.

    Each version of the UCT_MLEM code contains the following structure and files.
    
    Headers/                    - Contains header files
    Source/                     - Contains source files
      CPUFunctions.cu           - Functions that are run on the CPU
      GraphicsCardFunctions.cu  - Functions that are run on the GPU
      Parallel_Reform.cu        - Main code that reads input, ends data to GPU and creates output
    Reform                      - Executable binary produced after compiling
    Setup.txt                   - Configuration file
    makefile                    - Makefile used for compiling code

    There are also four supporting folders in the repository listed below with a brief description.
    
    Images/      - Folder where images are saved from Plotting scripts
    Input/       - Folder for UCT_MLEM input files
      Filtered/  - Folder for filtered CSV input files [UCT_MLEM looks here for input files]
      Raw/       - Folder for raw CSV input files
      Filter.py  - Python script to filter cones and convert data to MATT input data structure
    Plotting/    - Folder where different plotting scripts are stored (6 in total)
    Output/      - Folder where UCT_MLEM output is stored


 4- SETUP.TXT

    The UCT_MLEM code can be configured without needing to recompile by using the 
    Setup.txt file.  An example of this configuration file can be seen here.  At startup,
    the UCT_MLEM code will read the values from the Setup.txt file to set the reconstruction
    parameters. The ordering of the lines is important as the code reads each value after
    the equal sign in order. It is important that nothing in the Setup.txt files are 
    altered other than the values after the equal signs. This is an exaple of the Setup.txt file
    taken from the MLEM_NO_MATRIX repository.
    
    __________________________________________________________________________
    Setup.txt
    
      File Naming
    1)    Name of the input file              = 180315-run7-cs137_at_05_-05_11mm-2hrs-BigE-UnPhy-ComL_2x
    2)    Added Modifier to the output name   = test
    3)    Load previous iteration             = 180315-run7-cs137_at_05_-05_11mm-2hrs-BigE-UnPhy-ComL_2x_C57344_x100y100z1_I10

      Defining the Source Space
    -    X-limits = -50.0 , 50.0
    -    Y-limits = -50.0 , 50.0
    -    Z-limits = -50.0 , 50.0

    -    The number of voxels per dimension x,y,z = 30 , 30 , 30
    -    The number of theads per iteration block = 450
    -    The number of iteration blocks           = 30
      THE NUMBER OF VOXELS MUST EQUAL THE NUMBER OF ITERATION THREADS

      Defining the Event Data
    -    The total number of imported cones       = 28160
    -    The number of theads per cone block      = 256
    -    The number of cone blocks                = 110
      THE NUMBER OF CONES MUST EQUAL THE NUMBER OF CONE THREADS

      Run Length and Saves
    -    The total number of iterations     = 500
    -    The number of iterations per save  = 500

      Other Parameters
    -    The Noise Reducing Cuttoff as a fraction of the maximum value  = 0.001
    -    The assumed uncertainty in the energy measurements in MeV      = 0.1

      Method of Saving Data (Can have more than one selected)
    -    DAT files in the same format as those produced by CORE = 0
    -    CSV files to be used in Matthew's plotting codes       = 1
     
    __________________________________________________________________________
    
    
    
    The following is a description on each of these values, listed in the order they 
    appear in the Setup.txt file:
    
    Name of the input file:
    -   This is the name of the input file which the code looks for in the Input/Filtered
        repository. It contains a list of the scatters in a CSV file. The format is explained
        in detail in the following section.
        
    Added Modifier to the output name
    -   This is a modifier to the name of the output file which is saved by the MLEM code. See
        the section on DATA OUTPUT to see how the output files are named. Can be left blank.
        
    Load previous iteration
    -   Here you are able to load and start iteration from where you last left off. The features of 
        your new iteration (number of cones, which cones, voxel dimensions etc) must be itentical.
        This is if you do not have enough time to complete your entire iteration plan or the program
        crashes during iteration. During setup, it will look for a file with this name in the Output
        folder. Can be left blank.
    
    Defining the Source Space
    -   The next few values define the "Source Space". This is a rectangular volume which is 
        voxelised and used for the reconstruction. The values listed here are the edges of the
        volume in each dimension. The leading value must always be smaller than the later for 
        each axis. 

    The number of voxels per dimension x,y,z:
    -   This is simply the desited number of voxels the Source Space will be partitioned into
        for each dimension. These values do not have to be equal and thus the code allows for 
        non-cubic voxels. This can be thought of as the resolution of the space.
   
    The number of theads per iteration block, and
    The number of iteration blocks: 
    -   These two values define how the CUDA will manage the many parallel threads during the
        iteration phase. CUDA executes a batch of threads together called a "block". Each block
        contains a fixed number of threads and is executed simultaneosly. More than one block 
        may be executed by the GPU at a given time. The way that this code works, is that during
        the iteration phase, each voxel will be given its own thread in a particular block. Thus
        it is crucial for the succes of the code that the total number of threads (the number of
        blocks times the number of threads in each block) and the total number of voxels
        in the source space (the product of the number of voxels in each dimension) is exactly the
        same.
        In addition to this, the blocks are filled on the GPU with threads in wraps of 32. 
        Therefore, the code will execute faster if the number of threads per block is a multiple of
        32. Also, if there are less threads per block then more blocks can be fit on the GPU.
        I find that having 256 threads per block is the sweet spot for both of these conditions.
        Although it should be stated that the time saved is usually very small and 32 often is 
        not a factor of the number of voxels, so this is more of an optional step. Though it is 
        good to keep the size of the block under 500.

     The total number of imported cones:
     -   The total number of imported scattering events. Must not exceed the number available 
         in the input file. 
     
    The number of theads per cone block and,
    The number of cone blocks:
    -   This is for the part of the code where each cone/scattering-event is now assigned a thread.
        The product of these two values must equal the number of cones. See the above entry for the 
        number of voxels as to why.

    The total number of iterations:
    -   The maximum number of MLEM iterations the code will execute.
    
    The number of iterations per save:
    -   The esitmated gamma distribution is saved after the first iteration, the last iteration, and 
        after any multiple of this value.
        

    The Noise Reducing Cuttoff as a fraction of the maximum value:
    -   After each iteration, the value of the highest voxel is found. Any entry below this fraction
        of it is then zeroed out. This saves time as the code is designed to skip already zeroed voxels
        and the MLEM iteration on its own will only make an entry approch zero, not reach zero. As more 
        voxels are zeroed out one will notice the iteration accelerate cosiderably. Therefore, the eatimated
        time remaining (printed on the console during execution) is usually much too high during the first
        few iterations.
    
    
    The assumed uncertainty in the energy measurements in MeV:
    -   This is a massive part of the performance of the code. It is the estimated uncertainty associated with 
        the amount of energy deposited at each scattering location. It is independant of the scatter. It will
        influence the uncertainty o the scattering angle (which is dependant on which scatter we are looking at).
        If this is too high then the image is blurred out, and looses all detail. If it is too low then the image will 
        be exeptionally noisy and erouneous hotspots will occur during iteration.

     Method of Saving Data:
    -   For each of these values just indiacate a 1 or a 0 for whether or not you want the data saved using your
        selected format. Both options can be selected.


 5- DATA INPUT
 
    The CORE file format is used by all versions of the UCT_MLEM code except the Energy_Integration.
      
    CORE Input File Format [identical to input file format used by CORE]:
       
    Energy 1, X-Pos 1, Y-Pos 1, Z-Pos 1, Energy 2, X-Pos 2, Y-Pos 2, Z-Pos 2
    
    - energy in MeV and postion in mm
 
 
 6- DATA OUTPUT
 
    There are two primary output file formats, both stored in a CSV format.  It is possible
    to switch between the two formats in the Setup.txt file.  The DAT format mimics the output.dat
    output from the CORE reconstruction code.  The CSV format is the original output designed
    by Matt Leigh.  All of the plotting scripts in the Plotting folder use the CSV output format
    
    DAT Output Format
    
    [Header] - example: 5 bins in X, 6 and 7 in Y and Z, respectively.
    Row 1 lists the number of bins in x y z 
    Row 2 is the bin edges for X (5 bins have 6 bin edges)
    Row 3 is the bin edges for Y (6 bins have 7 bin edges)
    Row 4 is the bin edges for Z (7 bins have 8 bin edges)
 
    [Data] - The bin count rows are x*y in length and the go
    z1: x1y1, x2y1, x3y1, x4y1, x5y1, x1y2, . . . , x3y6, x4y6, x5y6
    z2: x1y1, x2y1, x3y1, x4y1, x5y1, x1y2, . . . , x3y6, x4y6, x5y6
    . . .
    . . .
    . . .
    z7: x1y1, x2y1, x3y1, x4y1, x5y1, x1y2, . . . , x3y6, x4y6, x5y6
    
    CSV Output Format: Stores values of non-zero voxels only. The positions listed are the centers voxel in mm.
                       Also stored is the i,j,k index numbers of the voxel in the space. Each of which run from 0 to 
                       the total number of divisions in the x, y, z dimension respectively.
                       Each voxel gets its own line in the following format: 

    bin_count , x_position , y_position , z_position , i , j , k
    
    Both files are saved in the Output repository with the following naming method. The names will
    apply the origional name of the inout file, the modifier listed in the Setup.txt file, the number 
    of scattering events used in the reconstruction, the number of voxels in each dimension and the
    number of MLEM iterations that took place before the file was saved. Files names are therefore 
    structured as (without brackets):
    
        (input file name)(name modifier)_C(number of scattering events)_x(voxels along x-axis)_y(voxels along y-axis)_z(voxels along z-axis)_I(iterations)
 
 7- HOW TO RUN

    - Compile using nvcc [executable created in the same folder]
        % make

    - Execute UCT_MLEM 
        % ./Reform


 8- KNOWN BUGS

	= MLEM_NO_MATRIX - Currently can only read in double scatter events - Will add ability to read triple scatters in future
    
    Never run a reconstruction where the source space does not include all possible source of gammas caputed by the Compton camera.
    For example, dont set the source space to only include one point source if there were two present during the exposure. This will
    cause all cones from source outside the space to push the distribution to the edge of the space, this will cause catastophic effects
    on the reconstructed image. 
    
    When executing a reconstruction of high energy gammas (eg: Cobalt 60) then the image is usually very noisy. What will occur is that
    the image is still reconstructed, but alot of the distribution is pushed into the corners of the image, creating hotspots only a couple 
    of pixels wide. When looking at a simple heatmap of the space after reconstruction, these corner hotspots are so bright that it can mean that
    sometimes the actual source is not visible at all despite being present. Therefore plotting fuctions, such as Method.py, automatically get rid
    the voxels which are close to an edge.
    
    If the input-file is not named correctly or does not exist there will be a segmentation fault after the propmt:
    - Importing Input Data to Cone-Matrix:




-------------------------------------------------------------------
