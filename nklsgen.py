import numpy as np
import itertools
import random


def bin_to_str(code):
    """
    Convert list of binary values to str of binary values
    
    Parameters
    ----------
    code : list
        list of binary values [0,1,0,0,1,1,0,1]
        
    Returns
    ----------
    string : str 
        str of binary values ['01001101']
        
    Examples
    ----------
    >>> code = [1,1,0,0,0,1]
    >>> bin_to_str(code)
    '110001'
    """
    
    string = str()
    for each in code:
        string += (str(int(each)))
    return string

def str_to_bin(code):
    """
    Convert str of binary values to list of binary values
    
    Parameters
    ----------
    string : str 
        string of binary values ['01001101']

    Returns 
    ----------
    code : list
        list of binary values [0,1,0,0,1,1,0,1]
            
    Examples
    ----------
    >>> code = '1100010'
    >>> bin_to_str(code)
    [1,1,0,0,0,1,0]
    """
    
    bin_ = []
    for each in code:
        bin_.append(int(each))
    return bin_


def normalize_landscape(landscape):
    """
    This function normalizes landscape values such that:
        min(landscape.values()) == 0 and max(landscape.values()) == 1
        
    Parameters
    landscape : dict
        dictionary in which:
            landscape.keys() are strings of binary values
            landscape.values() are floats
    
    Returns
    ----------
    landscape : dict
        dictionary in which:
        landscape.keys() are strings of binary values
        landscape.values() are float values in [0,1]
    """
    
    vals = list(landscape.values())
    min_, max_ = min(vals), max(vals)
    diff = max_ - min_
    return dict(zip(list(landscape.keys()), [((x - min_) / diff) for x in vals]))


def generate_interaction_matrix(N, K, from_dist=False):
    """
    Create random interaction matrix given N, K
    
    Parameters
    ----------
    N : int
        number of alleles
    K : int
        number of interactions between alleles (same for every allele)
    
    Returns 
    ----------
    interaction_matrix : np.array
        (N)x(N) matrix with random interactions
        
    Examples
    ----------
    >>> int_matrix_random(5,5)
    array([[1., 1., 0., 0., 0.],
           [1., 1., 0., 0., 0.],
           [0., 0., 1., 0., 1.],
           [0., 1., 0., 1., 0.],
           [0., 0., 1., 0., 1.]])
    """
    
    interaction_matrix = np.zeros((N,N))
    
    for primary_allele in list(range(N)):
        
        alleles = list(range(N))
        alleles.remove(primary_allele)
        
        random.shuffle(alleles)
        alleles.append(primary_allele)
        
        sec_alleles = alleles[-(K+1):]
        
        for sec_allele in sec_alleles:
            interaction_matrix[primary_allele, sec_allele] = 1
            
    return interaction_matrix.astype('int')


def prep(N, K, base=2):
    """
    This function is for neatly setting up objects for landscape generation
    Creates a proto-landscape, interaction_matrix, landscape_key
    
    Parameters
    ----------
    N : int
        number of alleles
    K : int
        number of interactions between alleles (same for every allele)
    base : int
        number of states each allele can be in (normal NK landscapes have base==2)

    Returns
    ----------
    proto_landscape: np.array
        ([base]**N)x(N) matrix of random values [0,1]
    int_matrix : np.array
        (N)x(N) matrix with random interactions
    ls_key : np.array
        (1)x(N) array for converting genotype string into base10 index 

    Example
    ----------
    >>> prep(N=5, K=1)
    array([[1., 1., 0., 0., 0.],
           [1., 1., 0., 0., 0.],
           [0., 0., 1., 0., 1.],
           [0., 1., 0., 1., 0.],
           [0., 0., 1., 0., 1.]])
    array([[0.24095055, 0.52126665, 0.5182196 , 0.26316811, 0.66344476],
           [0.46878655, 0.11950298, 0.77278226, 0.16061041, 0.18268662],
           [0.45490393, 0.10435652, 0.46649772, 0.75401804, 0.78114005],
           ...,
           [0.56060879, 0.5895411 , 0.4252403 , 0.76356792, 0.09401038],
           [0.62082816, 0.8230607 , 0.51474512, 0.03132471, 0.56050153],
           [0.36132159, 0.12410364, 0.88080286, 0.11930435, 0.10359306]])
    array([16,  8,  4,  2,  1])
    """
    
    int_matrix = generate_interaction_matrix(N,K)
    
    proto_landscape = np.random.rand(base**N, N)
    
    ls_key = np.power(base, np.flip(np.arange(N)))
    
    return int_matrix, proto_landscape, ls_key
    

def calc_fit(N, proto_landscape, int_matrix, genotype, ls_key):
    """
    Calculates fitness of current position given proto-landscape
    
    Parameters 
    ----------
    N : int
        number of genes
    proto_landscape : np.array
        (2**N)x(N) dimensional list of random values [0,1]
    int_matrix : np.array
        (N)x(N) matrix with random interactions
    genotype : np.array
        (1)x(N) binary string representing position (position on landscape)
    ls_key : np.array
        (1)x(N) array for converting genotype string into base10 index
        
    Returns
    ----------
    fitness_vector : np.array
        fitness values of each allele in provided genotype given int_matrix
        
    Example
    ----------
    >>> calc_fit(N, proto_landscape, int_matrix, genotype, key)
    array([0.77557927, 0.79878075, 0.85783928, 0.67450553, 0.6081521 ])
    """
    
    fitness_vector = np.zeros(N)
    for allele in np.arange(N):
        active_interactions = genotype * int_matrix[allele]
        fitness_vector[allele] = proto_landscape[np.sum(active_interactions * ls_key), allele]
        
    return fitness_vector


def nk_landscapes(N, K, num_of_ls=1, identical=False, simple_output=True, base=2):
    """
    This function generates random NK landscapes with indices (genotypes + fitness_values)
    
    Parameters
    ----------
    N : int
        number of alleles
    K : int
        number of interactions between alleles (same for every allele)
    num_of_ls : int
        number of landscapes to generate
    identical : boolean
        are landscapes the same?
    simple_output : boolean
        If true, output is [genotype, mean_fitness]
        If false, output is [genotype, allele_fitness, mean_fitness]
    base : int
        number of states each allele can be in (normal NK landscapes have base==2)
    
    Returns
    ----------
    landscapes : np.array that contains (num_of_ls) np.arrays
        
        Each array shape is (base**N)x(base*N+1) in shape
        Where each row in each array records a genotype, allele fitness values, and average fitness
        
        landscapes[0][0] => [0, 1, 1, 0, 0, 0.43, 0.57, 0.12, 0.98, 0.72, 0.56]
        landscapes[0][0][0, 0:N] => [0, 1, 1, 0, 0] (genotype)
        landscapes[0][0][0, N:2*N] => [0.43, 0.57, 0.12, 0.98, 0.72] (fitness values for each allele)
        landscapes[0][0][0, 2*N] => [0.56]
        
    Example
    ----------
    >>> generate_landscapes(N, K)
    [array([[0.        , 0.        , 0.        , 0.        , 0.        ,
             0.71047101, 0.72516974, 0.68606963, 0.50738105, 0.91240935,
             0.70830015],
            [0.        , 0.        , 0.        , 0.        , 1.        ,
             0.71047101, 0.72516974, 0.68606963, 0.50738105, 0.05950145,
             0.53771857],
            [0.        , 0.        , 0.        , 1.        , 0.        ,
             0.87194581, 0.72516974, 0.77717397, 0.28544239, 0.71054667,
             0.67405572], 
    """
    
    # Prepare empty landscapes array (SIMPLE OUTPUT)
    landscapes = np.zeros((num_of_ls, base**N, N+1))
    
    # Main procedure
    for ls in list(range(num_of_ls)):
        
        # Runs when you want landscapes generated from DIFFERENT int_matrix/proto_landscapes
        int_matrix, proto_landscape, ls_key = prep(N, K)
    
        # For each genotype, calculate allele_fitness AND mean_fitness values
        row = 0
        for genotype_ in itertools.product(range(base), repeat=N):
            genotype = np.array(genotype_)
            allele_fitness = calc_fit(N, proto_landscape, int_matrix, genotype, ls_key)
            mean_fitness = np.array([np.mean(allele_fitness)])
            
            # Assign genotype, and mean_fitness to landscape in landscapes array                
            landscapes[ls][row] = np.concatenate((genotype, mean_fitness))
            
            # Next row index
            row += 1
      
    return landscapes


def generate_landscapes(N, K, num_of_ls=1, preprocessed=True):
    """
    This function wraps the generation function and does some processing on landscapes
    
    Parameters
    ----------
    N : int
        number of alleles
    K : int
        number of interactions between alleles (same for every allele)
    num_of_ls : int
        number of landscapes to generate
    preprocessed : boolean
        Reformats each landscape in the landscapes np.array from np.array to dict where:
            landscapes[i].keys() => strings of binary values '01101101'
            landscapes[i].values() => [0,1] normalized floats 
        
    Returns
    ----------
    landscapes_ : np.array of np.arrays
    OR 
    landscapes : np.array of dicts
    """
    
    landscapes_ = nk_landscapes(N, K, num_of_ls, identical=False, simple_output=True, base=2)
    
    if not preprocessed:
        return landscapes_
    
    landscapes = []
    
    for ls in landscapes_:
        cut = ls.shape[1] - 1
        keys_, values_ = np.split(ls.copy(), [cut], axis=1)
        keys = [bin_to_str(code) for code in keys_.astype('int').tolist()]
        values = [round(val[0], 10) for val in values_]
        
        landscapes.append(normalize_landscape(dict(zip(keys, values))))
        
    return landscapes
