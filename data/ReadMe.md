# Datasets

## Training dataset

Training files are in `./excellent_unoriented/`, spectra are from the RRUFF database and available online. They are arranged in npy and pkl files for practical reasons:

  - `obs.npy` contains the X spectra;
  - `labels.pkl` contains the associated individual labels;
  - `array_labels.npy` contains the label probability in an array (i.e. for each sample n, a m vector with the associated label probability, so we have an n by m array at the end).

## Mixed Synthetic

The `./mixed_synthetic` folder contains the notebook `GenerateSynthetic.ipynb` used to generate the mixed datasets, used for instance in ./notebooks/CNN_1D.ipynb. The following files contain the data:

 - `signal_synthetic.npy` contains the X spectra;
 - `fractions_synthetic.npy`  contains the label probability in an array (i.e. for each sample n, a m vector with the associated label probability, so we have an n by m array at the end).

## Files to further test the method

Files in the To_Recognize folder from the IPGP Geomaterial Raman database:
- `obs.npy` contains the X spectra;
- `labels.pkl` contains the associated individual labels;
- `array_labels.npy` contains the label probability in an array (i.e. for each sample n, a m vector with the associated label probability, so we have an n by m array at the end).

**Warning :** In this case the labels are false labels as we don't know for sure the answer there...
