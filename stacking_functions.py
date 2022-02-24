import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
import matplotlib as mpl
from scipy.optimize import curve_fit
import scipy.constants as sc
from numba import jit
import os


### ----------- Calculation Functions  ----------- ###

def pbfactor(px, py, freq, pix_size, cx, cy, pdim):
    """
    Calculate the multiplicative correction for frequency-dependent primary beam attentuation.
    """
    
    # θPB = 42/νGHz https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/fov
    FWHM = 42./(freq/1000.)
    FWHM_sigma = (FWHM/(2*np.sqrt(2*np.log(2))))
    delx = ((px-cx)*pdim)/60.
    dely = ((py-cy)*pdim)/60.
    separation = np.sqrt((delx*delx)+(dely*dely))
    return 1./(np.exp(-1*((separation**2)/(2*(FWHM_sigma**2)))))


def sum_to_flux(pix_sum, a, b, pix_dim):
    """
    Calculate flux from a pixel sum, for a specified beam and area being summed over.
    """
    
    beam_area = (np.pi*a*b)
    return pix_sum*(pix_dim*pix_dim/beam_area)


def calculate_distance(freq):
    """
    Calculate the distance [Mpc] to a galaxy for a given frequency.
    Assume a "FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)" cosmology.
    """
    
    z = (1420.40575/freq) - 1
    return cosmo.luminosity_distance(z).value


def HImass_calc(datafile, box, major, minor, pixdim, chan_width):
    """
    Calculate the HI mass from stacked cube in line luminosity/beam units.
    The box parameter represents the entirety of what should be included,
    so we need to add 1 to the ends to keep things nice and pythonic.
    """
    
    # Fetch the data.
    hdulist = fits.open(datafile)
    data = hdulist[0].data[0]
    hdulist.close()
    
    # Get the sum.
    pixsum = np.sum(data[box[2][0]:box[2][1]+1,box[1][0]:box[1][1]+1,box[0][0]:box[0][1]+1])
    
    # Get the per pixel error and calculate the error for the given box size.
    pixstd = np.std(data)
    dims = (box[:,1]-box[:,0])+1
    pixsumstd = pixstd*np.sqrt(dims[0]*dims[1]*dims[2])

    # Return the mass and mass error in proper units.
    HImass = 2.36e5*sum_to_flux(pixsum, major, minor, pixdim)*chan_width
    HImasserr = 2.36e5*sum_to_flux(pixsumstd, major, minor, pixdim)*chan_width
    
    return np.array([HImass, HImasserr])


### ----------- HI Stacking Functions  ----------- ###

def read_catalog(fname, rowskip, delim, coluse, lowf, highf, isz, lowdec, highdec, lowra, highra, color, stmass):
    """
    Read in a catalog and produce the data array required for the stack.
    Catalog is required to be in the form of id, ra, dec, z.
    """
    
    # Read in the data.
    inpraw = np.loadtxt(fname, skiprows=rowskip, delimiter=delim, usecols=coluse)
    
    # Turn axis into frequency if catalog is in redshift.
    if isz:
        inpraw[:,3] = 1420.40575/(1+inpraw[:,3])
    
    # Get the data in the proper window
    inptemp = []
    for ii in range(0,len(inpraw)):
        if inpraw[ii][3] > lowf and inpraw[ii][3] < highf:
            if inpraw[ii][1] > lowra and inpraw[ii][1] < highra:
                if inpraw[ii][2] > lowdec and inpraw[ii][2] < highdec:
                    if color[0] == True and stmass[0] == False: # Add based on only a color selection.
                        if inpraw[ii][4] > color[1] and inpraw[ii][4] < color[2]:
                            inptemp.append(inpraw[ii])
                    elif color[0] == False and stmass[0] == True: # Add based only on a stellar mass selection.
                        if inpraw[ii][5] > stmass[1] and inpraw[ii][5] < stmass[2]:
                            inptemp.append(inpraw[ii])
                    elif color[0] == True and stmass[0] == True: # Add based on color and stellar mass selection.
                        if inpraw[ii][4] > color[1] and inpraw[ii][4] < color[2]:
                            if inpraw[ii][5] > stmass[1] and inpraw[ii][5] < stmass[2]:
                                inptemp.append(inpraw[ii])
                    else:
                        inptemp.append(inpraw[ii])
    # Return the final array.                   
    return np.array(inptemp)


def construct_pixel_catalog(specdata, lowfreq, highfreq, pixdim, cx, cy, chwidth, ra, dec):
    """
    1. Remove sources that would exist in edge channels.
    2. Convert RA, DEC, FREQ into specified cube coordinates.
    """
    
    ### Get the galaxies in the usabale velocity bin. ###
    subcat_list = []
    
    for ii in range(0, len(specdata)):
        if specdata[ii][3] > (lowfreq+5.) and specdata[ii][3] < (highfreq-5.):
            subcat_list.append(specdata[ii])
    subcat = np.array(subcat_list)
    print('You are attempting to stack '+str(len(subcat))+' galaxies.')
    ### Convert to pixel coordinates. ###
    subcat[:,1] = ((((ra-subcat[:,1])*3600.)/pixdim)+cx).astype(int)
    subcat[:,2] = ((((subcat[:,2]-dec)*3600.)/pixdim)+cy).astype(int)
    subcat[:,3] = (((subcat[:,3]-lowfreq)/chwidth)).astype(int)
    
    return subcat


def extract_cubelet(HIdata, source, startfreq, pix_size, cx, cy, chwidth, Dexp):
    """
    Extract a subcube around an individual HI source with dimensions 45x64x64.
    """
    
    # Define the bounds for the cube extraction.
    low_chan = int(source[3] - 22)
    high_chan = int(source[3] + 23)
    low_dec = int(source[2] - 32)
    high_dec = int(source[2] + 32)
    low_ra = int(source[1] - 32)
    high_ra = int(source[1] + 32)
    # Extract the data.
    temp=HIdata[low_chan:high_chan,low_dec:high_dec,low_ra:high_ra]
    
    # Calculate Primary Beam correction.
    freq_0 = startfreq + (source[3]*chwidth)
    pb = pbfactor(source[1], source[2], freq_0, pix_size, cx, cy,pix_size)
    
    # Put the data into units of Flux x Dist^2 and return it
    dist = calculate_distance(freq_0)
    
    # Create final return products.
    data = temp*dist*dist*pb
    weight = 1./(np.std(temp)*np.std(temp)*(dist**Dexp)*pb)
    
    # Return the data and the weights.
    return [data, weight]


def extract_all_cubelets(HIdata, sources, startfreq, pix_size, cx, cy, chan, Df):
    """
    Pull all the sub cubes around each source.
    """
    
    # Pull all the data.
    source_list = []
    weights = np.zeros(len(sources))
    for ii in range(0, len(sources)):
        cubelet_data = extract_cubelet(HIdata,sources[ii],startfreq, pix_size, cx, cy, chan, Df)
        source_list.append(cubelet_data[0])
        weights[ii] = cubelet_data[1]
        
    # Put the data into a nice array.
    dim1,dim2,dim3,dim4 = len(source_list),source_list[0].shape[0],source_list[0].shape[1],source_list[0].shape[2]
    allsources = np.zeros((dim1,dim2,dim3,dim4))
    for ii in range(0, len(source_list)):
        allsources[ii] = source_list[ii]
    
    return [allsources, weights]


def stack_cubes(HIdata, slist, sfreq, psize, cra, cdec, cw, Dpow):
    """
    Stack cubes via a weighted average.
    """
    
    stackinp = extract_all_cubelets(HIdata, slist, sfreq, psize, cra, cdec, cw, Dpow)
    allcubes =  stackinp[0] # The cubes around the target galaxy.
    allweights = stackinp[1] # The weights for each cube.
    
    # Stack the cubes.
    stacked_cube = np.zeros((45,64,64))
    for ii in range(0,len(allcubes)):
        temp = (allcubes[ii]*allweights[ii])/np.sum(allweights)
        stacked_cube += temp
        
    return stacked_cube

def stack_inpfile(inpfile, HIdata, freqs, pdim, center, cwide, pra, pdec, Dlaw):
    """
    Stack cubes with an input file.
    """
    
    # Convert the input source catalog into pixel units.
    cubecat = construct_pixel_catalog(inpfile, freqs[0], freqs[1], pdim, center[0], center[1], cwide, pra, pdec)
    
    # Do the stack.
    stacked = stack_cubes(HIdata, cubecat, freqs[0], pdim, center[0], center[1], cwide, Dlaw)
    
    return stacked


def convert_stack_cube(newimgfile, stackinput, template):
    """
    Convert the stacked cube into a FITS file that CASA can use.
    """
    
    # Copy the template desired with the output name.
    cmd = 'scp -r '+template+' '+newimgfile
    os.system(cmd)

    # Open the new image in update mode.
    newhdu = fits.open(newimgfile, mode='update')

    # Define the new frequency data.
    chanwidth = newhdu[0].header['CDELT3']
    start = 1420405752-(chanwidth*((stackinput.shape[0])/2.))

    # Update the FITS header to ensure it is correct.
    newhdu[0].header['NAXIS1'] = stackinput.shape[1]
    newhdu[0].header['NAXIS2'] = stackinput.shape[2]
    newhdu[0].header['NAXIS3'] = stackinput.shape[0]
    newhdu[0].header['CRVAL3'] = 1420405752
    newhdu[0].header['CRPIX3'] = np.floor((stackinput.shape[0])/2.)

    # Update the data.
    newhdu[0].data[0] = stackinput

    newhdu.flush()
    newhdu.close()
    
    