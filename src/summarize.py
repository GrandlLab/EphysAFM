import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
import feather
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def getThresh(gdf, df, roi, retrieve='index'):
    crosses = np.zeros(len(gdf))
    for i, g in gdf:
        ylim = df.thresh[i-1]
        xlim = df.tpeakf[i-1]
        temp = g.query('absi_blsub >= @ylim & ti <= @xlim & ti > @roi[0]').reset_index(drop=True)
        temp2 = g.query('absi_blsub <= @ylim & ti <= @xlim & ti > @roi[0]').reset_index(drop=True)
        if len(temp) == 0:
            crosses[i-1] = 0
        else:
            val = np.argmax(temp2.loc[:,"index"])
            if retrieve == 'work':
                crosses[i-1] = temp2.work[val]
            elif retrieve == 'force':
                crosses[i-1] = temp2.force[val]
            elif retrieve == 'time':
                crosses[i-1] = temp2.ti[val]
            else:
                crosses[i-1] = val
            
    return(pd.Series(crosses))

def splitSweep(sweep, roi):
  """
  This function finds the index of peak force and splits the sweep into
  an approach phase and a retract phase based on this index.
  """
  approach = sweep.query('ti <= @roi[0] + 150')
  retract = sweep.query('ti > @roi[0] + 150')
  return(approach, retract)

def linFit(x, a, b):
  """
  This function defines the linear fit.
  """
  return a*x + b

def findSteadyState(df, window):
  subset = df.query('ti >= @window[0] & ti <= @window[1]')
  return np.mean(subset.i_blsub)


def summarizeFile(path, roi=[750,1250], blsub=[50, 150]):
    dat = pd.read_feather(path)
    sensitivityDat = pd.read_csv(("_").join(path.split("_")[0:-2]) + '_sensitivity.csv', header=None)
    paramDat = pd.read_csv(("_").join(path.split("_")[0:-2]) + '_params.csv', header=0, index_col=0)
    nsweeps = max(dat.sweep)
    dat = dat.assign(absi_blsub = np.abs(dat.i_blsub)) 
    dat = dat.assign(iblsub_wthresh = butter_lowpass_filter(dat['i_blsub'], 1000, 25000, 6))


    grps = dat.groupby('sweep')
    agg_df = grps.apply(
          lambda x, roi=roi, blsub=blsub: pd.Series(
              {
                  'peakf' : np.max(x.query('ti > @roi[0] & ti < @roi[1]').force),
                  'peaki' : np.max(np.abs(x.query('ti > @roi[0] & ti < @roi[1]').i_blsub)), 
                  'peakw' : np.max(x.query('ti > @roi[0] & ti < @roi[1]').work),
                  'tpeakf' : x.tin0[x.query('ti > @roi[0] & ti < @roi[1]').force.idxmax()],
                  'tpeaki' : x.ti[np.abs(x.query('ti > @roi[0] & ti < @roi[1]').i_blsub).idxmax()],
                  'tpeakw' : x.tin0[x.query('ti > @roi[0] & ti < @roi[1]').work.idxmax()],
                  'wpeakf' : x.work[x.query('ti > @roi[0] & ti < @roi[1]').force.idxmax()],
                  'offset' : x.query('ti >= (@roi[0] + 5) & ti <= (@roi[0] + 15)').iblsub_wthresh.mean(),
                  'vstep' : x.query('ti >= @roi[0] & ti <= (@roi[0]+50)').v.mean(),
                  'f95' : x.query('ti >= 995 & ti <= 996').force.mean(),
                  'i95' : x.query('ti >= 995 & ti <= 996').i_blsub.mean(),
                  'leak' : x.query('ti >= @blsub[0] & ti <= @blsub[1]').i.mean(),
                  'stdev' : x.query('ti >= (@roi[0] + 5) & ti <= (@roi[0] + 15)').iblsub_wthresh.std(),
                  'vhold' : x.query('ti >= @blsub[0] & ti <= @blsub[1]').v.mean(),
              }
          )
    ).reset_index("sweep")

    agg_df = agg_df.assign(
        delay = agg_df['tpeaki'] - agg_df['tpeakf'],
        seal = agg_df['vhold']/agg_df['leak'],
        thresh = np.abs(agg_df['offset']) + 2*agg_df['stdev'],
        date = np.repeat(paramDat.loc['date'][0], nsweeps),
        cell = np.repeat(paramDat.loc['cell#'][0], nsweeps),
        Rs = np.repeat(paramDat.loc['Rs'][0], nsweeps),
        Cm = np.repeat(paramDat.loc['Cm'][0], nsweeps),
        Rscomp = np.repeat(paramDat.loc['Rscomp'][0], nsweeps),
        kcant = np.repeat(paramDat.loc['kcant'][0], nsweeps),
        dkcant = np.repeat(paramDat.loc['dkcant'][0], nsweeps),
        protocol = np.repeat(path.split("_")[-2], nsweeps),
        velocity = np.repeat(paramDat.loc['velocity'][0], nsweeps),
        construct = np.repeat(paramDat.loc['construct'][0], nsweeps),
        osm = np.repeat(paramDat.loc['mosm'][0], nsweeps),
        uniqueID = np.repeat(paramDat.loc['uniqueID'][0], nsweeps),
        path = np.repeat(path, nsweeps),
      )

    agg_df = agg_df.assign(
        threshind = getThresh(grps, agg_df, roi=roi),
        thresht = getThresh(grps, agg_df, roi=roi, retrieve="time"),
        fthresh = getThresh(grps, agg_df, roi=roi, retrieve="force"),
        wthresh = getThresh(grps, agg_df, roi=roi, retrieve="work")
      )


    agg_df = agg_df[[
          'path','uniqueID', 'date', 'construct', 'cell', 'protocol', 'sweep', 'velocity', 'kcant', 'dkcant',
          'osm', 'Rs', 'Rscomp', 'Cm', 'seal', 'vhold', 'vstep', 'peaki', 'tpeaki', 'peakf', 'tpeakf', 'f95', 'i95',
          'peakw', 'tpeakw', 'wpeakf', 'leak', 'offset', 'stdev', 'delay', 'thresh', 'threshind', 'thresht', 'fthresh', 'wthresh'
        ]]

    agg_df.to_csv(("_").join(path.split("_")[0:-1]) + '_summary.csv', mode='w')
    
def summarizeDirectory(folderPath, protocol, roi=[650,1250], window=[50, 150]):
  """
  This function will run the summarizeFile function on all files in a folder.
  """

  path_list = []
  for root, dirs, files in os.walk(folderPath):
    for file in files:
      if file.find(protocol + '_preprocessed.feather') != -1:
        path_list.append(os.path.join(root, file).replace("\\","/"))
    
  for path in path_list:
    print(path)
    summarizeFile(path, roi=roi, blsub=window)

def make_sweepfile(path):
  """
  This function takes all the sweeps in the aggregate file and concatenates their approach traces into a single
  file with unique identifiers for later analysis and plotting.
  """
  agg = pd.read_csv(path)
  appended_df = []
  for i, row in agg.iterrows():
    print(row.path)
    sweep_file = pd.read_feather(row.path)
    if (row.tpeakf >= 700.0) & (row.tpeakf < 1000.0):
        roi = [750, 1150]
    elif row.tpeakf >= 1000.0:
        roi = [1000, 1400]
    else:
        roi = [500, 900]
        
    rep_sweep = sweep_file.query("sweep == @row.sweep & ti >= @roi[0] & ti <= @roi[1]").reset_index(drop=True)
    [approach, retract] = splitSweep(rep_sweep, roi)
    
    approach = approach.assign(
        uniqueID=np.repeat(row['uniqueID'], np.shape(approach)[0]),
        sweep=np.repeat(row['sweep'], np.shape(approach)[0]),
        construct=np.repeat(row['construct'], np.shape(approach)[0]),
        absi_blsub=np.abs(approach['i_blsub']),
        absi_blsub_mech=np.abs(approach['i_blsub']) - row['offset']
    )
    

    appended_df.append(approach[['uniqueID','sweep', 'construct', 'position', 'absi_blsub', 'absi_blsub_mech', 'force', 'work', 'deflection']])

  appended_df = pd.concat(appended_df)
  output_filename = input('What would you like to call the aggregate sweep file? ')
  appended_df.to_csv(("/").join(path.split("/")[0:-1]) + '/' + output_filename + '.csv')
  print(("/").join(path.split("/")[0:-1]) + '/' + output_filename + '.csv')

def find_slopes(summaryPath, sweepPath, x, y):
  """
  This function will take a summary file and allow you to fit the slopes to all the associated approach traces.
  The slopes will then be added to the summary data file.
  """
  summary = pd.read_csv(summaryPath)
  sweep_file = pd.read_csv(sweepPath).reset_index(drop=True)

  def fitselect(xmin, xmax):
    """
    This function will fit a line to a manually selected region of the work-current plot. The fit will be
    added to the plot for visualization and the slope will be returned in pA/fJ.
    """
    indmin, indmax = np.searchsorted(temp[x], (xmin, xmax))
    indmax = min(len(temp[x]) - 1, indmax)

    subx = temp[x][indmin:indmax]
    suby = temp[y][indmin:indmax]
    try:
      poptLin = curve_fit(linFit, subx, suby)[0]
    except RuntimeError:
      print("Error - curve fit failed")

    from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
    at = AnchoredText("Sensitivity: " + str(round(poptLin[0], 3))+" (pA/fJ)",
      prop=dict(size=8), frameon=True, loc=2,)

    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    fitY = poptLin[0]*subx + poptLin[1]
    ax.plot(subx, fitY, '--', color='red')
    ax.axvspan(temp[x][indmin], temp[x][indmax], color='grey', alpha=0.25)
    slopes.append(round(poptLin[0], 3))
    fig.canvas.draw()

  slopes = []
  for i, row in summary.iterrows():
    plt.close('all')
    val =row.uniqueID
    print(val)
    temp = sweep_file.query("uniqueID == @val & sweep == @row.sweep").reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(temp[x], temp[y])
    ax.set_xlabel('Work(fJ)')
    ax.set_ylabel('Current (A)')
    
    span = SpanSelector(ax, fitselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))
    plt.draw()
    while not plt.waitforbuttonpress():
      pass
    new_column = pd.DataFrame({'slope': slopes})
  summary = summary.merge(new_column, left_index=True, right_index=True)
  print(slopes)
  return(slopes)
