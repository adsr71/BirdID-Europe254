####################################################################################################
###############################################  Config  ###########################################
####################################################################################################

# Model specific params (DO NOT CHANGE)

speciesPath = 'species.csv'



nClasses = 254 # Europe Birds

samplerate = 22050

fftSizeInSamples = 1536
fftHopSizeInSamples = 360

segmentDuration = 5.0

# Mel Params
nMelBands = 128
melStartFreq = 20.0
melEndFreq = 10300.0
nLowFreqsInPixelToCutMax = 4
nHighFreqsInPixelToCutMax = 6

imageMean = [0.5, 0.4, 0.3]
imageStd = [0.5, 0.3, 0.1]


#imageHeight = 224
#resizeFactor = (imageHeight/(nMelBands-nLowFreqsInPixelToCutMax/2.0-nHighFreqsInPixelToCutMax/2.0))
#imageWidth = int(resizeFactor * segmentDuration * samplerate / fftHopSizeInSamples)
#imageSize = (imageWidth, imageHeight)
#print('resizeFactor', resizeFactor, 'imageSize', imageSize) # resizeFactor 1.8211382113821137 imageSize (557, 224)
imageSize = (557, 224)


####################################################################################################

debug = False # True False

modelSize = 'medium' # Model size small: EffNetB0, meduim: EffNetB2, large: EffNetV2

nCpuWorkers = 4 #16 #-2    # Number of cpu worker for data loader, if negative: subtract from all cpus available
batchSizeInference = 8 #16 #32 # NNN
batchSizeFiles = 16 #32 #16 #8 #4

segmentOverlapInPerc = 60 #80 #20 #60 # With 5s: 90%=0.5s, 80%=1s, 60%=2s, 20%=4s,  #10=4.5
segmentDurationMasked = 5.0 #2.0
segmentPooling = 'max' # mean max meanexp

# ToAdd maybe: channelAggregation/Pooling = None|Max|Mean


printFreq = 100
dataLoaderTimeOut = 0 #20




gpuMode = True


# Mime types supported by soundfile (wav)
mimeTypesSoundfile = ['audio/wav', 'audio/wave', 'audio/x-wav', 'audio/vnd.wave', 'audio/x-pn-wav']
# Mime types supported by soundfile (ogg, FLAC)
mimeTypesSoundfile += ['audio/ogg', 'audio/x-ogg', 'audio/flac', 'audio/x-flac']
#mimeTypesSoundfile += ['audio/x-ogg-flac', 'audio/x-ogg-pcm', 'audio/x-oggflac', 'audio/x-oggpcm']


inputPath = 'example/'
outputDir = 'example/'
errorLogPath = 'example/error-log.txt'
speciesSelectionPath = speciesPath # 'example/'

fileOutputFormatsValid = ['raw_pkl', 'raw_excel', 'raw_csv', 'labels_excel', 'labels_csv', 'labels_audacity', 'labels_raven']
fileOutputFormats = ['raw_pkl', 'raw_excel', 'labels_excel']
useFloat16InPkl = False # To reduce pkl file size predictions are casted from np.float32 to np.float16 
sortSpecies = False # Sort order of species columns by max value in text output files (e.g. in raw_excel, raw_csv)
outputPrecision = 5 # Number of decimal places for results in output files

# NNN
#terminalOutputFormat = 'summery' # summery, summeryJson, naturblick2022, ammodJson, resultDictJson
terminalOutputFormat = 'summary' # summary, summaryJson, resultDictJson, naturblick2022, ammodJson, none

csvDelimiter = ';'

startTime = 0.0
endTime = -1.0 #None
channels = None
mono = False

minConfidence = 0.75
channelPooling = 'max' # max mean (todo: None)
mergeLabels = False

nameType = 'sci' # sci en de (ix id)



# To init later
speciesData = None
clearPrevErrorLog = True
modelTorchScriptPath = 'ModelTorchScript_medium.pt'


minConfidences = None
classIxsSelection = None



# Get and set config

def getConfig():
    
    return {
        'speciesPath': speciesPath,
        'speciesSelectionPath': speciesSelectionPath,
        'nClasses': nClasses,
        'samplerate': samplerate,
        'fftSizeInSamples': fftSizeInSamples,
        'fftHopSizeInSamples': fftHopSizeInSamples,
        'segmentDuration': segmentDuration,
        'nMelBands': nMelBands,
        'melStartFreq': melStartFreq,
        'melEndFreq': melEndFreq,
        'nLowFreqsInPixelToCutMax': nLowFreqsInPixelToCutMax,
        'nHighFreqsInPixelToCutMax': nHighFreqsInPixelToCutMax,
        'imageMean': imageMean,
        'imageStd': imageStd,
        'imageSize': imageSize,
        'debug': debug,
        'segmentDurationMasked': segmentDurationMasked,
        'segmentOverlapInPerc': segmentOverlapInPerc,
        'segmentPooling': segmentPooling,
        'printFreq': printFreq,
        'dataLoaderTimeOut': dataLoaderTimeOut,
        'modelSize': modelSize,
        'modelTorchScriptPath': modelTorchScriptPath,
        'nCpuWorkers': nCpuWorkers,
        'batchSizeInference': batchSizeInference,
        'batchSizeFiles': batchSizeFiles,
        'gpuMode': gpuMode,
        'mimeTypesSoundfile': mimeTypesSoundfile,
        'inputPath': inputPath,
        'outputDir': outputDir,
        'fileOutputFormats': fileOutputFormats,
        'terminalOutputFormat': terminalOutputFormat,
        'csvDelimiter': csvDelimiter,
        'errorLogPath': errorLogPath,
        'clearPrevErrorLog': clearPrevErrorLog,
        'startTime': startTime,
        'endTime': endTime,
        'channels': channels,
        'mono': mono,
        'minConfidence': minConfidence,
        'channelPooling': channelPooling,
        'mergeLabels': mergeLabels,
        'nameType': nameType,
        'speciesData': speciesData,
        'minConfidences': minConfidences,
        'classIxsSelection': classIxsSelection,

    }


def setConfig(c):

    global speciesPath
    global speciesSelectionPath
    global nClasses
    global samplerate
    global fftSizeInSamples
    global fftHopSizeInSamples
    global segmentDuration
    global nMelBands
    global melStartFreq
    global melEndFreq
    global nLowFreqsInPixelToCutMax
    global nHighFreqsInPixelToCutMax
    global imageMean
    global imageStd
    global imageSize
    global debug
    global segmentDurationMasked
    global segmentOverlapInPerc
    global segmentPooling
    global printFreq
    global dataLoaderTimeOut
    global modelSize
    global modelTorchScriptPath
    global nCpuWorkers
    global batchSizeInference
    global batchSizeFiles
    global gpuMode
    global mimeTypesSoundfile
    global inputPath
    global outputDir
    global fileOutputFormats
    global terminalOutputFormat
    global csvDelimiter
    global errorLogPath
    global clearPrevErrorLog
    global startTime
    global endTime
    global channels
    global mono
    global minConfidence
    global channelPooling
    global mergeLabels
    global nameType
    global speciesData
    global minConfidences
    global classIxsSelection
    
    speciesPath = c['speciesPath']
    speciesSelectionPath = c['speciesSelectionPath']
    nClasses = c['nClasses']
    samplerate = c['samplerate']
    fftSizeInSamples = c['fftSizeInSamples']
    fftHopSizeInSamples = c['fftHopSizeInSamples']
    segmentDuration = c['segmentDuration']
    nMelBands = c['nMelBands']
    melStartFreq = c['melStartFreq']
    melEndFreq = c['melEndFreq']
    nLowFreqsInPixelToCutMax = c['nLowFreqsInPixelToCutMax']
    nHighFreqsInPixelToCutMax = c['nHighFreqsInPixelToCutMax']
    imageMean = c['imageMean']
    imageStd = c['imageStd']
    imageSize = c['imageSize']
    debug = c['debug']
    segmentDurationMasked = c['segmentDurationMasked']
    segmentOverlapInPerc = c['segmentOverlapInPerc']
    segmentPooling = c['segmentPooling']
    printFreq = c['printFreq']
    dataLoaderTimeOut = c['dataLoaderTimeOut']
    modelSize = c['modelSize']
    modelTorchScriptPath = c['modelTorchScriptPath']
    nCpuWorkers = c['nCpuWorkers']
    batchSizeInference = c['batchSizeInference']
    batchSizeFiles = c['batchSizeFiles']
    gpuMode = c['gpuMode']
    mimeTypesSoundfile = c['mimeTypesSoundfile']
    inputPath = c['inputPath']
    outputDir = c['outputDir']
    fileOutputFormats = c['fileOutputFormats']
    terminalOutputFormat = c['terminalOutputFormat']
    csvDelimiter = c['csvDelimiter']
    errorLogPath = c['errorLogPath']
    clearPrevErrorLog = c['clearPrevErrorLog']
    startTime = c['startTime']
    endTime = c['endTime']
    channels = c['channels']
    mono = c['mono']
    minConfidence = c['minConfidence']
    channelPooling = c['channelPooling']
    mergeLabels = c['mergeLabels']
    nameType = c['nameType']
    speciesData = c['speciesData']
    minConfidences = c['minConfidences']
    classIxsSelection = c['classIxsSelection']
    