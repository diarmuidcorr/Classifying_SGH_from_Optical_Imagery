#script used to download all optical satellite imagery from a chosen sensor.
#The satellite imagery download script, FELS, is from Vasco Nunes:
#SOURCE: https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud
#Compiled by Diarmuid Corr, d.corr@lancaster.ac.uk

from fels import run_fels
from multiprocessing import Pool



#run_fels installed from: https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud,
#see documentation for tips.
for grid in grids:
def fels_function(grid):
    run_fels(grid, sensor, start_MS, end_MS, output= output_dir, cloudcover = cloud_cover,
             includeoverlap = True, latest=False,excludepartial=False, overwrite=False,
             outputcatalogs=os.path.expanduser(catalogue_path)
             ) #see the documentation
    return

def pool_function(grids):
    pool = Pool(n_proc)
    pool.map(fels_function, grids)
    pool.close()
    pool.join()
    return

if __name__ == '__main__':
    #define global variables:
    n_proc = 12 #example of the number of processors to be used.
    #Should not exceed number of values in grids variable or CPUs on the server used.
    
    grids = ['list_of_grids', 'commas_separated'] #e.g. '22WEV' for S2; '203031' for L8, L7.

    #start_MS should be the day before the intended search period, 
    #taking the form: '2017-04-30' for intended start date: '2017-05-01'
    #end_MS should be the day after the intended search period, 
    #taking the form: '2017-10-01' for intended start date: '2017-09-30'
    #make sure to break this into individual melt-seasons or you’ll just download all the imagery.

    start_MS =  '2017-04-30' #day before the search window i.e. 1st Jan 2017
    end_MS =  '2017-10-01' #day after the search window i.e. 31st Mar 2017

    cloud_cover =  20 #max cloud cover %
    catalogue_path =  '' #path to outputted index file, this is where the function searches for tiles
    #index files can be large ~5-10GB each. Make sure the chosen directory exists.

    output_dir = '' # string for the path to the output directory. Where files will be downloaded to.

    sensor = 'sensor abbreviation' #e.g. 'L7', 'L8', 'S2' etc.
    
    pool_function(grids)
