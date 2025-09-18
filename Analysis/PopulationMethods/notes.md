# Index of folders and scripts in PopulationMethods with comments

**important**
Please add the library path to your PYTHONPATH variable: `Analysis/PopulationMethods/lib`


**note on running stuff**
For interactive visalizations that run videos, you might need to do something like:

First, unstill bad installations of PyQt5, if any
```
pip uninstall -y PyQt5 PyQt5-Qt5 PyQt5-sip qtpy
```
Then install everything with conda/mamba
``` 
micromamba install -c conda-forge \
  pyqt=5 gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad\
  gst-plugins-ugly glib qtpy ffmpeg\
  plotly python-kaleido pyqtwebengine pyopengl
```

Okay, things are still now working on windows, so I am relying on the vlc player. First make sure VLC is intalled. Then run
```
pip install python-vlc 
```

## LDA - linear discriminant analysis

### lda.py

Petros' LDA script. Work in progress.

### ldaDF.py

Testing the new libraries on top of Petros' script.

### 01 LDA without PCA

First attempt by Dylan with LDA. It does not perform PCA on the components. Uses time lags. Performance is 60% to 70%, but I haven't plotted the confusion matrix. Easy to extract most relevant neurons.

### 02 LDA gridsearch

This is like 01, but I implement a grid search over parameters to check how performance depends on bin size, number of lag steps, etc. Also, I merge pup_grab and pup_retrieve in the same label.

Outcome: takes about 40 min. What matters the most is to cover a 0.2 s window, either by lags or using larger bins. (plot is on Notion). This in preparation of logistic regression and other methods with sparseness constraint, where I will grid-search over different sparseness coefficients.

## PCA - principal component analysis

### 01 PCA lag visualize

Perform PCA in different conditions and with or without data enhanced with time lags.

### 02 PCA by areas

PyQt app that allows to select any session, and plots first 3 PCs. User can select bin size, PAG area, and behaviour to visualize.

## Visualizations

Here I will do dashboards and more interactive things.

### 99 test movie files pqt5

Check which video files can be opened.

### 01 Visualize Simple Rasters

Web-based dashboard that allows to select a mouse/session combination, and plots the raster in the web-browser.

### 02 Movies Rasters

Main file: `movie_raster_onebehavior.py`
Here I use pyqtgraph to plot movie, behavior segments for a specific behavior, and raster plot. All updated dynamically.

### 03 Time behaviour movie raster

New iteration, where one can select specific labeled behaviour, and the movie
is played only around those behaviours. Units to show can be selected too,
through a dataframe that needs to be included as pickle file.

## Logistic regression

Also essentially linear, but has some advantages compared to the LDA. 
The most important one is that it can incorporate sparseness constraints.

### 01 Try constraints

`do_gridsearch` Grid search over parameters. Quite long.

`try_gpu` single fit, but using GPU optimization (with one line at the beginning). Much faster. Visualization of confusion matrix and which neurons are important.

### 02 Include no behaviour

`one_fit.py` is an improved version from the above, where I add random tags in the initial part of the simulation, and use it to train a tag called "loiter" indicating the mouse doing nothing relevant. The results are saved on a pickled file.  `one_fit_movie_dashboard.py`  visualizes the results with video of behavior.

### 03 Plot compare

Here I make summary tables to really capture the neurons that contribute the most to each label. I also compare performance where those high-score neurons are removed, versus removing other random neurons.

If I want to be stricter, I should remove neurons with similar rates, at least.

### 04 Hunting session

Here I specialize the code a bit more on hunting behaviors in a hunting session (i.e. 23)

### 05 Hunt and retrieval 29

Here I focus on session 20240529, with both 6 escapes and multiple pup retrieval trials. The goal is to successfully decode escape, and compare the neurons relevant for escape with the neurons relevant for pup retrieval.

### 06 Poster stuff

Here I work on the figures and the result for Petros poster (September 2025).

+ **fit and save**: K-fold logistic regression fitting. With focus on hunting and parenting behaviours, ignoring escape.
+ **01 get ranking df** : read the data from fit and save, and generate a dataframe that indicates which neurons are the most important for all behavioural tag.

