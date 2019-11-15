# Copyrights (c) Charles Le Losq 2017
import numpy as np
import os, sys

import matplotlib
matplotlib.use('Agg') # non interactive renderer
from matplotlib import pyplot as plt

import pandas as pd
import rampy
import scipy

import pickle

from tqdm import tqdm

def data_prep(spectra_liste,choice,start_path,out_path,**kwargs):
    print('Preparing to treat the spectra in\n'+start_path+'\nand output the images in path\n'+out_path)

    # getting options
    switch_type = kwargs.get("switch_type","xy") # Switching between the plot or xy mode
    switch_baseline = kwargs.get("baseline",False) # In case the baseline is wanted
    delimiter = kwargs.get("delimiter",",") # In case the baseline is wanted

    # getting the index
    print("Getting the interesting data...")
    for j in tqdm(range(len(choice['mineral']))):
        if j == 0:
            idx_liste = spectra_liste.loc[spectra_liste['mineral'] == choice['mineral'][j]].index.values
        else:
            idx_liste2 = spectra_liste.loc[spectra_liste['mineral'] == choice['mineral'][j]].index.values
            idx_liste = np.hstack((idx_liste,idx_liste2))

    # For sklearn output
    xcommon = np.arange(200,1200,1.0)
    output_bulk_array = np.zeros((len(idx_liste),len(xcommon)))
    output_labels = []

    print("Processing the interesting data...")
    for i in tqdm(range(len(idx_liste))):
        idx = idx_liste[i]

        # testing if the folder mineral exists
        newpath = out_path+spectra_liste['mineral'][idx]
        if switch_type != 'sklearn':
            if not os.path.exists(newpath):
                os.makedirs(newpath)

        # testing if the spectrum corresponding to idx has been generated already
        # if test is false => generated and save figure
        if switch_type == "plot":
            out_path_spectrum = newpath+'/'+spectra_liste['filename'][idx]+'.png'
        elif switch_type == "xy":
            out_path_spectrum = newpath+'/'+spectra_liste['filename'][idx]+'.csv'

        #
        # Data importation, baseline treatment, and normalisation
        #
        spectrum = np.genfromtxt(start_path+spectra_liste['filename'][idx],delimiter=delimiter,skip_header=10)

        if switch_baseline == True:
            out1, out2 = rampy.baseline(spectrum[:,0],spectrum[:,1],np.array([[0,1600]]),"als",lam=10.0**5)
            x = spectrum[:,0]
            y = out1[:,0]
        else:
            x = spectrum[:,0]
            y = spectrum[:,1]

        if switch_type == "plot" or switch_type == "xy":
            if not os.path.isfile(out_path_spectrum):

                if switch_type == "plot":
                    x1 = np.where(x <= 200)
                    x2 = np.where((x>200) &  (x <= 400))
                    x3 = np.where((x>400) &  (x <= 600))
                    x4 = np.where((x>600) &  (x <= 800))
                    x5 = np.where((x>800) &  (x <= 1000))
                    x6 = np.where((x>1000) &  (x <= 1200))
                    x7 = np.where((x>1200) &  (x <= 1400))

                    plt.figure(figsize=(6,6))
                    plt.plot(x,y,'-',color='none')
                    plt.fill_between(x[x1], 0, y[x1],color='red')
                    plt.fill_between(x[x2], 0, y[x2],color='purple')
                    plt.fill_between(x[x3], 0, y[x3],color='green')
                    plt.fill_between(x[x4], 0, y[x4],color='cyan')
                    plt.fill_between(x[x5], 0, y[x5],color='orange')
                    plt.fill_between(x[x6], 0, y[x6],color='magenta')
                    plt.fill_between(x[x7], 0, y[x7],color='blue')

                    plt.xlim(0,1500)
                    plt.ylim(0,)

                    plt.axis('off')
                    # If we haven't already shown or saved the plot, then we need to
                    # draw the figure first...
                    #fig.canvas.draw()

                    plt.tight_layout()

                    plt.savefig(out_path_spectrum)

                    plt.close() # no need when using Agg renderer but still added to avoid any trouble

                    # IF YOU WANT TO SAVE THINGS IN AN ARRAY? WORK WITH THAT
                    # Now we can save it to a numpy array.
                    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                elif switch_type == "xy":
                    np.savetxt(out_path_spectrum,np.hstack((x,y)),delimiter=',')

        elif switch_type == "sklearn":
            # For the X output = the spectrum
            f = scipy.interpolate.interp1d(x,y,bounds_error=False,fill_value=0.0) # y values outside the observed x are put to 0.
            output_bulk_array[i,:] = f(xcommon)/np.max(f(xcommon))

            # For the Y output = the labels
            label = int(choice[choice['mineral'] == spectra_liste['mineral'][idx]]['label'])
            output_labels.append(label)
        else:
            NameError("Choose between sklearn, xy or plot")

    # in case we ask output for sklearn, we need to save it
    if switch_type == "sklearn":
        with open(out_path+'labels.pkl', 'wb') as fp:
                pickle.dump(output_labels, fp)
        np.save(out_path+'obs', output_bulk_array)

    print('Done')

def main():
    #
    # For generating Plots for 2D CNN
    #
    spectra_liste = pd.read_excel('./data/file_liste_excellent.xlsx',sheet_name='liste') # RRUUFF dataset
    choice = pd.read_excel('./data/file_liste_excellent.xlsx',sheet_name='subset_to_train')
    spectra_liste_naturals = pd.read_excel('./data/file_liste_excellent.xlsx',sheet_name='testnat') # anto's zircon and diamonds

    #
    # For 2D CNN
    #

    #data_prep(spectra_liste_predict,choice,r'../RRUFF/all_anto/',r'../RRUFF/test_anto/',switch_type='plot',baseline=True)

    #data_prep(spectra_liste,choice,r'../RRUFF/excellent_unoriented/',r'../RRUFF/all/',switch_type='plot')

    #
    # Not used now
    #

    ###data_prep(spectra_liste,choice,r'../RRUFF/excellent_unoriented/',r'../RRUFF/xy/all/',switch_type='xy',baseline=False)

    #
    # For scikit-learn treatment
    #

    #data_prep(spectra_liste,choice,r'./data/excellent_unoriented/',r'./data/excellent_unoriented/',switch_type='sklearn',baseline=True)

    data_prep(spectra_liste_naturals,choice,r'./data/To_Recognize/spectra/',r'./data/To_Recognize/',switch_type='sklearn',delimiter=" ",baseline=True)


if __name__ == "__main__":
    main()
