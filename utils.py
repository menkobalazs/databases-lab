#==============#
### PACKAGES ###

# commmon packages
import numpy as np
import pandas as pd
from datetime import datetime # get current time
import matplotlib.pyplot as plt
from tqdm import tqdm # progress bar
from IPython.display import clear_output # clear output in a loop

# Picture downloading related packages
import requests
from io import BytesIO
from PIL import Image

# slider and animation tools
import ipywidgets as widgets
from matplotlib import animation 

# astronomy related packages
from astroquery.sdss import SDSS

# SQL related packages
import psycopg2
import psycopg2.extras


#===============#
### CONSTANTS ###

# fontsize for plots
FS = 12

# image parameters
IMG_SIZE = 128 # img.shape --> 128x128x3
IMG_SCALE = 0.3

# Connection settings 
pgsql_settings = {
    'pguser' : 'menkobalazs1',
    'pgpasswd' : 'O67UT7',
    'pghost' : 'postgres-datasci.db-test',
    'pgport' : 5432,
    'pgdb' : 'menkobalazs1_project', 
    'schema' : 'public'
}


#===============#
### FUNCTIONS ###

# SQL related function
def connect_from_settings(settings):
    """ Create connection to the data base. """
    return psycopg2.connect(
        host = settings['pghost'],
        port = settings['pgport'],
        database = settings['pgdb'],
        user = settings['pguser'],
        password = settings['pgpasswd'],
        options=f'--search_path={settings["schema"]}'
    )

# SQL related function
def run_query(query):
    """ Run an SQL query. """
    connection = connect_from_settings(pgsql_settings)
    cursor = connection.cursor(cursor_factory = psycopg2.extras.DictCursor)
    cursor.execute(query)
    dict_res = cursor.fetchall()
    df = pd.DataFrame(dict_res,columns=list(dict_res[0].keys())) 
    cursor.close()
    connection.close()
    return df

# SDSS related fuction
def get_sdss_image(ra, dec, scale=IMG_SCALE, size=IMG_SIZE):
    """
    Retrieve a JPEG image from the SDSS (Sloan Digital Sky Survey) SkyServer.

    Parameters:
    - ra (float, required): Right Ascension in degrees.
    - dec (float, required): Declination in degrees.
    - scale (float, default=0.15): Scale in arcseconds per pixel.
    - size (int, default=128): Desired size of the output image in pixels (height and width).

    Returns:
    - List[int]: A flattened list of pixel values if the image is successfully retrieved.
    - None: If the image cannot be found.
    """
    url = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/" + \
          f"getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return np.array(image).reshape(-1).astype(int).tolist()
    else:
        print(f"Cannot find image with ra={ra} and dec={dec}. Status code: {response.status_code}")
        return None
    
# Image processing related function
def reshape_images(img, channels=3):
    """ 
    Reshapes a flat list or 1D array of pixel data into a 3D image array.

    Parameters:
        img (list or numpy array): The input flat array representing image pixel data.
                                   The length of `img` must be a perfect square multiplied
                                   by the number of channels (e.g., 3 for RGB images).
        channels (int): Number of channels in the image (default is 3 for RGB).

    Returns:
        numpy.ndarray: A reshaped 3D array with dimensions (height, width, channels), where
                       height and width are equal, calculated as the square root of the 
                       total number of pixels divided by the number of channels.
    """
    size = np.sqrt(len(img)/channels).astype(int)
    return np.array(img).reshape(size, size ,3)

# Data processing related function
def query_a_needed_set(needed_indices):
    """
    Queries a subset of data based on specified indices and processes image data.

    Parameters:
        needed_indices (list of int): A list of indices specifying the rows to query
                                      from the `redshift` table.

    Returns:
        pandas.DataFrame: A DataFrame containing the queried data with the following columns:
                          - 'specobjid': Unique identifier for the spectrum object.
                          - 'ra': Right Ascension coordinate.
                          - 'dec': Declination coordinate.
                          - 'z': Redshift value.
                          - 'zerr': Error in the redshift value.
                          - 'veldisp': Velocity dispersion.
                          - 'veldisperr': Error in the velocity dispersion.
                          - 'picture': List of normalized pixel values for the image
                                       (scaled to [0, 1] by dividing by 255).

    Raises:
        ValueError: If `needed_indices` is not a list or is empty.
    """
    needed_set = run_query(f"""
        CREATE TEMP TABLE temp_idx (idx INT);
        INSERT INTO temp_idx (idx) VALUES ({needed_indices});

        SELECT specobjid, ra, dec, z, zerr, veldisp, veldisperr, picture
           -- ARRAY(SELECT unnest(picture) / 255.0) AS normalized_picture -- possible selection, but too slow
        FROM redshift 
        JOIN temp_idx ON redshift.index = temp_idx.idx
    """)
    needed_set["picture"]  = needed_set["picture"].apply(lambda x: [i / 255 for i in x])
    return needed_set

# Data processing related function
def get_data(subset, column):
    """ 
    Extracts and processes data from a specific column of a dataset subset.

    Parameters:
        subset (dict-like): A dictionary-like object (e.g., pandas DataFrame or similar)
                            containing the data.
        column (str): The column name to extract data from. Must be one of the following:
                      - 'picture': Extracts and reshapes image data. Assumes the column
                                   contains flat image arrays that can be reshaped using
                                   `reshape_images`.
                      - Other valid columns: 'specobjid', 'ra', 'dec', 'z', 'zerr',
                                             'veldisp', 'veldisperr'.

    Returns:
        numpy.ndarray: 
            - For 'picture': A 4D array of shape `(n_samples, height, width, channels)`,
              where `height` and `width` are inferred from the first image.
            - For other columns: A 1D array of the values in the specified column.

    Raises:
        ValueError: If the `column` parameter is not one of the valid column names.
    """    
    if column == 'picture':
        shape = reshape_images(subset['picture'][0]).shape
        return np.array(subset['picture'].tolist()).reshape(-1, *shape)
    elif column in ['specobjid', 'ra', 'dec', 'z', 'zerr', 'veldisp', 'veldisperr']:
        return np.array(subset[column].tolist())
    else:
        valid_columns = ['specobjid', 'ra', 'dec', 'z', 'zerr', 'veldisp', 'veldisperr', 'picture']
        raise ValueError(f"column param must be in {valid_columns}, not '{column}'")

def plot_history(train, val, history_score='', ylim=(None, None), save_as=''):
    """ Plot CNN history scores. """
    print(f'Train {history_score}: {np.round(train[-1]*100,3)}%')
    print(f'Validation {history_score}: {np.round(val[-1]*100,3)}%')
    plt.figure(figsize=(9,6))
    plt.plot(train, '--o', c='r', label=f'train {history_score}')
    plt.plot(val, '--o', c='b', label=f'validation {history_score}')
    plt.title(history_score[0].upper() + history_score[1:] + ' of the Model', fontsize=FS+3)
    plt.xlabel('epochs', fontsize=FS)
    plt.ylabel(history_score, fontsize=FS)
    plt.legend(fontsize=FS)
    plt.grid(ls='dotted')
    plt.ylim(*ylim)
    if save_as:
        plt.savefig(f"figures/{save_as}.pdf", bbox_inches='tight')
    plt.show()
    return None

def plot_predicted_vs_true_values(pred, valid, save_as=''):
    """ Plot predicted vs true label and its histogram. """   
    plt.figure(figsize=(9,6))
    plt.plot([-1,1], [-1,1], '-', c='r', alpha=0.4)
    plt.scatter(pred, valid, marker='o', color='b', alpha=0.5)
    plt.xlim(0,0.8)
    plt.ylim(0,0.8)
    plt.title('Predicted and true redshift values', fontsize=FS+3)
    plt.xlabel(r'$z_{\text{predicted}}$', fontsize=FS)
    plt.ylabel(r'$z_{\text{true}}$', fontsize=FS)
    plt.grid(ls='dotted')
    if save_as:
        plt.savefig(f'figures/{save_as}.pdf', bbox_inches='tight')
    plt.show()
    
    
def plot_predicted_vs_true_histogram(pred, valid, save_as=''): 
    """ Plot predicted vs true label histogram. """   
    plt.figure(figsize=(9,6))
    plt.hist(pred, bins=100, color='b', alpha=0.5, label=r'$z_{\text{predicted}}$')
    plt.hist(valid, bins=100, color='r', alpha=0.5, label=r'$z_{\text{true}}$' )
    plt.title('Predicted and true redshifts on histogram', fontsize=FS+3)
    plt.ylabel('Number of data points', fontsize=FS)
    plt.xlabel('$z$', fontsize=FS)
    plt.legend(fontsize=FS)
    plt.grid(ls='dotted')
    if save_as:
        plt.savefig(f'figures/{save_as}.pdf', bbox_inches='tight')
    plt.show()
    return None



