import os
import traceback
import time
import numpy as np
import json
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
from PIL import Image

# To get MIME type
import magic
mime = magic.Magic(mime=True)

# NNN
resampleModule = 'librosa' # 'librosa' or 'torchaudio'
if resampleModule == 'torchaudio':
    import torchaudio.transforms as T


from collections import OrderedDict
import csv
from multiprocessing import cpu_count, freeze_support, Pool
import ffmpeg
from scipy.signal import butter, sosfilt
import pickle
import pandas as pd

import warnings
import argparse
import config as cfg


def addAndParseConfigParams(parser):

    parser.add_argument('-d', '--debug', action='store_true', default=cfg.debug, help='Show some debug infos.')

    parser.add_argument('-i', '--inputPath', type=str, default=cfg.inputPath, metavar='', help='Path to input audio file or folder. Defaults to ' + str(cfg.inputPath) + '.')

    parser.add_argument('-s', '--startTime', type=float, default=0.0, metavar='', help='Start time of audio segment to analyze in seconds. Defaults to 0.')
    parser.add_argument('-e', '--endTime', type=float, default=-1.0, metavar='', help='End time of audio segment to analyze in seconds. Defaults to duration of audio file.')

    parser.add_argument('--mono', action='store_true', default=cfg.mono, help='Mix audio to mono before analysis.')
    parser.add_argument('--channels', nargs='*', default=cfg.channels, type=int, metavar='', help='Audio channels to analyze. List of values in [1, 2, ..., #channels]. Defaults to None (using all channels).')

    parser.add_argument('-sd', '--segmentDuration', type=float, default=cfg.segmentDurationMasked, metavar='', help='Duration of audio chunks to analyze in seconds. Value between 1 and 5. Defaults to ' + str(cfg.segmentDurationMasked) + '.')
    parser.add_argument('-ov', '--overlapInPerc', type=float, default=cfg.segmentOverlapInPerc, metavar='', help='Overlap of audio chunks to analyze in percent. Value between 0 and 80. Defaults to ' + str(cfg.segmentOverlapInPerc) + '.')

    parser.add_argument('-m', '--modelSize', type=str, default=cfg.modelSize, metavar='', help='Model size. Value in [small, medium, large]. Defaults to medium.')
    
    parser.add_argument('-f', '--batchSizeFiles', type=int, default=cfg.batchSizeFiles, metavar='', help='Number of files to preprocess in parallel (if input path is directory). Defaults to ' + str(cfg.batchSizeFiles) + '.')
    parser.add_argument('-b', '--batchSizeInference', type=int, default=cfg.batchSizeInference, metavar='', help='Number of segments to process in parallel (by GPU). Defaults to ' + str(cfg.batchSizeInference) + '.')
    parser.add_argument('-c', '--nCpuWorkers', type=int, default=cfg.nCpuWorkers, metavar='', help='Number of CPU workers to prepare segments for inference. Defaults to ' + str(cfg.nCpuWorkers) + '.')

    
    parser.add_argument('-o', '--outputDir', type=str, default=cfg.outputDir, metavar='', help='Directory for result output file(s). Defaults to directory of input path.')
    parser.add_argument('--fileOutputFormats', nargs='*', default=cfg.fileOutputFormats, type=str, metavar='', help='Format of output file(s). List of values in [raw_pkl, raw_excel, raw_csv, labels_excel, labels_csv, labels_audacity, labels_raven]. Defaults to raw_pkl raw_excel labels_excel.')

    parser.add_argument('--minConfidence', type=float, default=cfg.minConfidence, metavar='', help='Minimum confidence threshold. Value between 0.01 and 0.99. Defaults to ' + str(cfg.minConfidence) + '.')
    parser.add_argument('--channelPooling', type=str, default=cfg.channelPooling, metavar='', help='Pooling method to aggregate predictions from different channels. Value in [max, mean]. Defaults to max.')
    parser.add_argument('--mergeLabels', action='store_true', default=cfg.mergeLabels, help='Merge overlapping/adjacent species labels.')
    parser.add_argument('--nameType', type=str, default=cfg.nameType, metavar='', help='Type or language for species names. Value in [sci, en, de, ...]. Defaults to ' + str(cfg.nameType) + '.')

    
    parser.add_argument('--useFloat16InPkl', action='store_true', default=cfg.useFloat16InPkl, help='Reduce pkl file size by casting prediction values to float16. ')
    parser.add_argument('--outputPrecision', type=int, default=cfg.outputPrecision, metavar='', help='Number of decimal places for values in text output files. Defaults to ' + str(cfg.outputPrecision) + '.')
    parser.add_argument('--sortSpecies', action='store_true', default=cfg.sortSpecies, help='Sort order of species columns in raw data files or rows in label files by max value.')
    parser.add_argument('--includeFilePathInOutputFiles', action='store_true', default=cfg.includeFilePathInOutputFiles, help='Include file path in output files.')
    parser.add_argument('--csvDelimiter', type=str, default=cfg.csvDelimiter, metavar='', help='Delimiter used in CSV files. Defaults to ' + str(cfg.csvDelimiter) + '.')
    

    parser.add_argument('--speciesPath', type=str, default=cfg.speciesSelectionPath, metavar='', help='Path to custom species metadata file or folder. If a folder is provided, the file needs to be named "species.csv". Defaults to ' + str(cfg.speciesSelectionPath) + '.')

    parser.add_argument('--errorLogPath', type=str, default=cfg.errorLogPath, metavar='', help='Path to error log file. Defaults to error-log.txt in outputDir.')

    parser.add_argument('--terminalOutputFormat', type=str, default=cfg.terminalOutputFormat, metavar='', help='Format of terminal output. Value in [summary, summaryJson, naturblick2022, resultDictJson, none]. Defaults to ' + str(cfg.terminalOutputFormat) + '.')

    # ToAddMaybe:
    # -sp, --species
    # --overlapInSec

    args = parser.parse_args()

    cfg.inputPath = args.inputPath

    # If outputDir not passed (has default value) set it to inputPath
    if args.outputDir == 'example/':
        if os.path.isfile(cfg.inputPath):
            cfg.outputDir = os.path.dirname(cfg.inputPath) + os.sep
        else: cfg.outputDir = cfg.inputPath
    else: cfg.outputDir = args.outputDir

    # Add '/' to outputDir
    if cfg.outputDir[-1] != os.sep:
        cfg.outputDir += os.sep

    # If errorLogPath not passed (has default value) use outputDir
    if args.errorLogPath == cfg.errorLogPath:
        cfg.errorLogPath = cfg.outputDir + 'error-log.txt'

    cfg.segmentDurationMasked = args.segmentDuration
    cfg.segmentOverlapInPerc = args.overlapInPerc

    

    cfg.modelSize = args.modelSize
    # NNN
    cfg.batchSizeFiles = args.batchSizeFiles


    cfg.batchSizeInference = args.batchSizeInference
    cfg.nCpuWorkers = args.nCpuWorkers
    cfg.debug = args.debug
    
    cfg.fileOutputFormats = args.fileOutputFormats
    cfg.useFloat16InPkl = args.useFloat16InPkl
    cfg.sortSpecies = args.sortSpecies
    cfg.outputPrecision = args.outputPrecision
    cfg.csvDelimiter = args.csvDelimiter
    cfg.includeFilePathInOutputFiles = args.includeFilePathInOutputFiles
    
    cfg.terminalOutputFormat = args.terminalOutputFormat
    
    cfg.startTime = args.startTime
    cfg.endTime = args.endTime

    # Sanity check and correction of start/end time
    if cfg.startTime < 0: cfg.startTime = 0.0
    if cfg.endTime < cfg.startTime: cfg.endTime = None

    cfg.channels = args.channels
    cfg.mono = args.mono

    cfg.minConfidence = args.minConfidence
    cfg.channelPooling = args.channelPooling
    cfg.mergeLabels = args.mergeLabels

    cfg.nameType = args.nameType
    cfg.speciesSelectionPath = args.speciesPath
    
    
    
    #print('cfg.channels', cfg.channels, flush=True)


    return parser


# Init more config parameters 
def init():

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Sanity checks and corrections
    if cfg.segmentDurationMasked < 1.0:
        cfg.segmentDurationMasked = 1.0
        print('Warning: segment duration < 1.0 not valid, set to 1.0', flush=True)
    if cfg.segmentDurationMasked > 5.0:
        cfg.segmentDurationMasked = 5.0
        print('Warning: segment duration > 5.0 not valid, set to 5.0', flush=True)

    if cfg.segmentOverlapInPerc < 0:
        cfg.segmentOverlapInPerc = 0
        print('Warning: segment overlap < 0 % not valid, set to 0 %', flush=True)
    if cfg.segmentOverlapInPerc > 80:
        cfg.segmentOverlapInPerc = 80
        print('Warning: segment overlap > 80 % not valid, set to 80 %', flush=True)


    if cfg.modelSize == 'small':
        cfg.modelTorchScriptPath = 'ModelTorchScript_small.pt'
    if cfg.modelSize == 'medium':
        cfg.modelTorchScriptPath = 'ModelTorchScript_medium.pt'
    if cfg.modelSize == 'large':
        cfg.modelTorchScriptPath = 'ModelTorchScript_large.pt'

    if cfg.minConfidence < 0.01:
        cfg.minConfidence = 0.01
        print('Warning: minConfidence < 0.01 not valid, set to 0.01', flush=True)
    if cfg.minConfidence > 0.99:
        cfg.minConfidence = 0.99
        print('Warning: minConfidence > 0.99 not valid, set to 0.99', flush=True)

    if cfg.channelPooling not in ['max', 'min']:
        cfg.channelPooling = 'max'
        print('Warning: channelPooling not in [max, min], set to max', flush=True)

    if cfg.nameType not in ['sci', 'en', 'de', 'ix', 'id']:
        cfg.nameType = 'sci'
        print('Warning: nameType not in [sci, en, de, ix, id], set to sci', flush=True)


    # Set number of cpu workers for data loader (relative to num of cpus available, if negative)
    nCpusAvailable = cpu_count()
    if cfg.nCpuWorkers > nCpusAvailable: 
        cfg.nCpuWorkers = nCpusAvailable
    if cfg.nCpuWorkers <= 0:
        cfg.nCpuWorkers = nCpusAvailable + cfg.nCpuWorkers

    # Check if gpus available
    if cfg.gpuMode:
        cfg.gpuMode = torch.cuda.is_available()

    # Get species metadata
    cfg.speciesData = pd.read_csv(cfg.speciesPath, sep=';', encoding='utf-8')
    if len(cfg.speciesData.index) != cfg.nClasses:
        print('Error: Wrong number of species. Please use original species.csv', flush=True)

    # Init species selection
    cfg.classIxsSelection = list(range(cfg.nClasses))
    
    # Use global minConfidence for all species
    cfg.minConfidences = [cfg.minConfidence] * cfg.nClasses

    # Get custom species selection and min confidence
    if os.path.isdir(cfg.speciesSelectionPath):
        cfg.speciesSelectionPath = os.path.join(cfg.speciesSelectionPath, 'species.csv')
        print('cfg.speciesSelectionPath', cfg.speciesSelectionPath, flush=True)

    if os.path.isfile(cfg.speciesSelectionPath):
        cfg.speciesDataSelection = pd.read_csv(cfg.speciesSelectionPath, sep=';', encoding='utf-8')
        cfg.classIxsSelection = cfg.speciesDataSelection['ix'].tolist()
        # Get and set custom minConfidence
        dfIxsWithCustomMinConfidence = cfg.speciesDataSelection[cfg.speciesDataSelection['minConfidence'].notnull()].index.tolist()
        for ix in dfIxsWithCustomMinConfidence:
            classIx = cfg.speciesDataSelection.at[ix, 'ix']
            cfg.minConfidences[classIx] = cfg.speciesDataSelection.at[ix, 'minConfidence']
    else:
        print('Warning: Custom species metadata file not found. Using all species.', flush=True)
    
    if cfg.debug: 
        print('nCpusAvailable', nCpusAvailable, flush=True)
        print('nCpuWorkers', cfg.nCpuWorkers, flush=True)
        print('gpuMode', cfg.gpuMode, flush=True)
        print('batchSizeInference', cfg.batchSizeInference, flush=True)
        print('batchSizeFiles', cfg.batchSizeFiles, flush=True) # NNN

    else:
        # Suppress ”SourceChangeWarning” (Needed?)
        warnings.filterwarnings("ignore")

    if cfg.fileOutputFormats and not os.path.exists(cfg.outputDir): 
        os.makedirs(cfg.outputDir)


    if cfg.clearPrevErrorLog:
        clearErrorLog()

    cudnn.benchmark = True # Not working otherwise ?

def loadModel():

    # Load Model (Checkpoint)

    model = torch.jit.load(cfg.modelTorchScriptPath)
    

    n_gpus = 0
    if cfg.gpuMode:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            model = torch.nn.DataParallel(model).cuda()
            #print('DataParallel', flush=True)
        else:
            model.cuda()


    if cfg.debug:
        print('Model loaded', flush=True)
        print('n_gpus', n_gpus, flush=True)

    #print('model', model)
    
    return model


def getAudioFilesInDir(path, allowed_filetypes=['wav', 'flac', 'mp3', 'ogg', 'm4a', 'mp4']):

    # Add backslash to path if not present
    if not path.endswith(os.sep):
        path += os.sep

    # Get all files in directory and subdirectories
    paths = []
    fileIds = [] # To check for duplicate filenames (without extension)
    for root, dirs, flist in os.walk(path):
        for f in flist:
            if len(f.rsplit('.', 1)) > 1 and f.rsplit('.', 1)[1].lower() in allowed_filetypes:
                path = os.path.join(root, f)
                paths.append(path)
                # Check for duplicate filenames
                fileId = os.path.splitext(os.path.basename(f))[0]
                if fileId not in fileIds:
                    fileIds.append(fileId)
                else:
                    print('Warning duplicate filename:', fileId, flush=True)
                    print('Output files might only be writen for:', path, flush=True)

    print('Found {} audio files to analyze'.format(len(paths)), flush=True)

    return sorted(paths)

def clearErrorLog():
    if os.path.isfile(cfg.errorLogPath):
        os.remove(cfg.errorLogPath)

def writeErrorLog(msg):
    with open(cfg.errorLogPath, 'a') as f:
        f.write(msg + '\n')

def round_floats(o, precision):
    if isinstance(o, float): return round(o, precision)
    if isinstance(o, dict): return {k: round_floats(v, precision) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x, precision) for x in o]
    return o


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AudioDataset(Dataset):

    def __init__(self, audioDicts, transform=None):

        self.audioDicts = audioDicts
        self.transform = transform

        self.segmentDurationInSamples = int(cfg.segmentDuration*cfg.samplerate)
        
        # Create list of segments
        # segment has attr.: fileIx, channelIx, startSample
        self.segments = []

        segmentDurationMaskedInSamples = int(cfg.segmentDurationMasked*cfg.samplerate)
        nFiles = len(audioDicts)

        # Remember segment offset index for each file
        self.segmentOffsetIxAtFileIx = []
        nSegments = 0
        
        for fileIx in range(nFiles):
            self.segmentOffsetIxAtFileIx.append(nSegments)
            audioDict = audioDicts[fileIx]
            nChannels = audioDict['nChannels']
            startTimes = audioDict['startTimes']
            for channelIx in range(nChannels):
                for startTime in startTimes:
                    segment = {}
                    segment['fileIx'] = fileIx
                    segment['channelIx'] = channelIx
                    segment['startSample'] = int(startTime*cfg.samplerate)
                    segment['endSample'] = segment['startSample'] + segmentDurationMaskedInSamples
                    
                    self.segments.append(segment)
                    nSegments += 1

    def __len__(self):
        return len(self.segments)


    def __getitem__(self, segmentIx):

        # Important: don't use any config params in here that have changed during runtime because windows doesn' support fork()

        segment = self.segments[segmentIx]
        
        fileIx = segment['fileIx']
        channelIx = segment['channelIx']
        startSample = segment['startSample']
        endSample = segment['endSample']

        #print(segmentIx, fileIx, channelIx, startSample, flush=True)
        
        specImage = getSpecSegment(self.audioDicts, fileIx, channelIx, startSample, endSample, self.segmentDurationInSamples)

        # Convert ...
        h,w = specImage.shape
        c=3
        ThreeChannelSpec = np.empty((h,w,c), dtype=np.uint8)
        ThreeChannelSpec[:,:,0] = specImage
        ThreeChannelSpec[:,:,1] = specImage
        ThreeChannelSpec[:,:,2] = specImage

        specImage = Image.fromarray(ThreeChannelSpec)

        if self.transform: specImage = self.transform(specImage)
        
        return specImage


def get_start_and_end_times_of_overlapping_segments(global_end_time, global_start_time=0.0, segment_duration=5.0, step_duration=1.0, fit_last_segment_to_global_end_time=True):

    start_times = []
    end_times = []

    start_time = global_start_time
    end_time = start_time + segment_duration

    while end_time < global_end_time:

        start_times.append(start_time)
        end_times.append(end_time)
        
        start_time += step_duration
        end_time = start_time + segment_duration

    if fit_last_segment_to_global_end_time:
        end_time = global_end_time
        start_time = end_time - segment_duration
        if start_time < global_start_time:
            start_time = global_start_time
        start_times.append(start_time)
        end_times.append(end_time)

    return start_times, end_times


def apply_high_pass_filter(input, sample_rate):


    # # Normalize (to prevent filter errors)
    # if np.max(input):
    #     input = input / np.max(input)
    #     input = input * 0.5

    order = 2
    cutoff_frequency = 2000

    sos = butter(order, cutoff_frequency, btype="highpass", output="sos", fs=sample_rate)
    output = sosfilt(sos, input, axis=0)

    # If anything went wrong (nan in array or max > 1.0 or min < -1.0) --> return original input
    #if np.isnan(output).any() or np.max(output) > 1.0 or np.min(output) < -1.0:
    if np.isnan(output).any():
        print("Warning filter instability", flush=True)
        output = input


    # print(type, order, np.min(input), np.max(input), np.min(output), np.max(output), flush=True)

    # # Normalize to -3dB
    # if np.max(output):
    #     output = output / np.max(output)
    #     output = output * 0.71

    return output

def readAndPreprocessAudioFile(audioPath, startTime=0.0, endTime=None, channels=None, mono=False):

    # Read audio file via soundfile if format is supported, else use ffmgeg

    mimeType = mime.from_file(audioPath)
    
    if mimeType in cfg.mimeTypesSoundfile:

        # Get some audio file infos
        with sf.SoundFile(audioPath, 'r') as f:
            sample_rate_src = f.samplerate
            n_channels = f.channels
            nSamples = len(f)
            duration = nSamples/sample_rate_src
            #print('duration', duration, flush=True)

        # Sanity check start/endTime
        if startTime and startTime >= duration:
            startTime = 0.0
            if cfg.debug:
                print('Warning: startTime >= duration, reset startTime=0.0', flush=True)
        if endTime and endTime >= duration:
            endTime = None
            if cfg.debug:
                print('Warning: endTime >= duration, reset endTime=None', flush=True)


        input_kwargs = {}
        input_kwargs['dtype'] = 'float32'
        input_kwargs['always_2d'] = True
        if startTime:
            input_kwargs['start'] = int(startTime*sample_rate_src)
        if endTime:
            input_kwargs['stop'] = int(endTime*sample_rate_src)
        
        #audioData, sample_rate_src = sf.read(audioPath, start=startSample, stop=endSample, always_2d=True)
        audioData = sf.read(audioPath, **input_kwargs)[0]

        if cfg.debug:
            print('Read audio file via sf', sample_rate_src, audioData.shape, audioData.dtype, flush=True)

    else:

        # Read via ffmpeg (all audio formats not supported by soundfile)

        # Get some metadata (sample_rate, n_channels)
        metadata = ffmpeg.probe(audioPath)['streams'][0]
        #print(metadata, flush=True)
        n_channels = metadata['channels']
        # Unfortunately duration from metadata is not very precise
        duration = float(metadata['duration'])
        #duration = float(metadata['duration'])-float(metadata['start_time'])
        sample_rate_src = int(metadata['sample_rate'])
        #print('duration', duration, flush=True)

        # Sanity check start/endTime
        if startTime and startTime >= duration-0.5:
            startTime = 0.0
            if cfg.debug:
                print('Warning: startTime >= duration-0.5, reset startTime=0.0', flush=True)
        if endTime and endTime >= duration-0.5:
            endTime = None
            if cfg.debug:
                print('Warning: endTime >= duration-0.5, reset endTime=None', flush=True)

        
        input_kwargs = {}
        output_kwargs = {}
        input_kwargs['ss'] = startTime
        if endTime:
            duration_to_read = endTime - startTime
            input_kwargs['t'] = duration_to_read
        #input_kwargs['loglevel'] = 'error'
        input_kwargs['v'] = 'error'

        #output_kwargs['ar'] = cfg.samplerate
        out, err = (ffmpeg
            .input(audioPath, **input_kwargs)
            #.output('pipe:', format='f32le', acodec='pcm_f32le', **output_kwargs)
            .output('pipe:', format='f32le', acodec='pcm_f32le')
            .global_args('-nostdin', '-hide_banner', '-nostats', '-loglevel', 'error') # use '-nostdin' to not break terminal 
            .run(capture_stdout=True)
        )
        # Convert to numpy
        out_np = np.frombuffer(out, np.float32)
        # Reshape to soundfile style (n_frames, n_channels)
        audioData = np.reshape(out_np, (-1, n_channels))

        if cfg.debug:
            print('Read audio file via ffmpeg', sample_rate_src, audioData.shape, audioData.dtype, flush=True)

    #print('audioData.shape', audioData.shape, flush=True)
    
    # Mix to mono
    if mono and audioData.shape[1] > 1:
        audioData = np.mean(audioData, axis=1, keepdims=True)
        if cfg.debug:
            print('Applying mono mix', audioData.shape, audioData.dtype, flush=True)

    # Filter channels
    if channels:
        # Remove channels out of bounds
        channels = [x for x in channels if x > 0 and x <= audioData.shape[1]]
        # Add first channel if all out of bounds
        if not channels:
            channels = [1]
        # channel no. to index: 1,2,3 --> 0,1,2
        channel_ixs = [x - 1 for x in channels]
        # Filter channels
        audioData = audioData[:, channel_ixs]
        # Reassign n_channels
        n_channels = audioData.shape[1]

    else:
        # Add all channels 1...
        channels = list(range(1, audioData.shape[1]+1))

    
    
    
    # Apply HP filter
    if cfg.debug:
        print('Applying high pass filter', flush=True)
    audioData = apply_high_pass_filter(audioData, sample_rate_src) # dtype is float64 after filter
    
    
    
    # Resample if needed
    if sample_rate_src != cfg.samplerate:

        if cfg.debug:
            print('Resample', sample_rate_src, '>', cfg.samplerate, 'using', resampleModule, flush=True)

        if resampleModule == 'torchaudio':

            # Reshape and convert to torch tensor
            audioDataTorch = np.transpose(audioData)            # [n_frames x n_channels] --> [n_channels x n_frames]
            audioDataTorch = torch.from_numpy(audioDataTorch)

            ## Simulating librosa resampling

            # # kaiser_best
            # lowpass_filter_width = 64
            # rolloff = 0.9475937167399596
            # resampling_method = "sinc_interp_kaiser"
            # beta = 14.769656459379492

            # kaiser_fast
            lowpass_filter_width = 16
            rolloff = 0.85
            resampling_method = "sinc_interp_kaiser"
            beta = 8.555504641634386

            resampler = T.Resample(
                sample_rate_src,
                cfg.samplerate,
                lowpass_filter_width=lowpass_filter_width,
                rolloff=rolloff,
                resampling_method=resampling_method,
                beta=beta,
                dtype=audioDataTorch.dtype,
            )

            audioDataTorchResampled = resampler(audioDataTorch)

            # Convert back to numpy
            audioData = audioDataTorchResampled.numpy()
            # Resahpe to sound file format [n_channels x n_frames] --> [n_frames x n_channels]
            audioData = np.transpose(audioData)

        else:

            # Make sure audio_data is in correct format for librosa processing
            if n_channels > 1:
                audioData = np.transpose(audioData)     # [n_frames x n_channels] --> [n_channels x n_frames]
            else:
                audioData = audioData[:,0]              # [n_frames x 1] --> (n_frames,)

            # # Normalize to -3dB
            # audioData /= np.max(audioData)
            # audioData *= 0.71

            # librosa needs Fortran-contiguous audio buffer (maybe not needed anymore)
            #if not audioData.flags["F_CONTIGUOUS"]: audioData = np.asfortranarray(audioData)

            audioData = librosa.resample(audioData, orig_sr=sample_rate_src, target_sr=cfg.samplerate, res_type='kaiser_fast')
            #audioData = librosa.resample(audioData, orig_sr=sample_rate_src, target_sr=cfg.samplerate, res_type='soxr_hq')
            

            # Reshape to sound file format [n_frames x n_channels]
            if n_channels > 1:
                audioData = np.transpose(audioData)                 # [n_channels x n_frames] --> [n_frames x n_channels]
            else:
                audioData = np.reshape(audioData, (-1, n_channels)) # (n_frames,) --> [n_frames x 1]
    

    # # Normalize to -3dB
    # audioData /= np.max(audioData)
    # audioData *= 0.71

    # if cfg.debug:
    #     print('sample_rate_src', sample_rate_src, 'n_channels', n_channels, 'audioData.shape', audioData.shape, flush=True)

    # Return audio data and (maybe modified) startTime, endTime, channels
    return audioData, startTime, endTime, channels

def resizeSpecImage(specImage, imageSize):

    specImagePil = Image.fromarray(specImage.astype(np.uint8))
    #specImagePil = specImagePil.resize(imageSize, Image.LANCZOS) # deprecated
    specImagePil = specImagePil.resize(imageSize, Image.Resampling.LANCZOS) 
    # Cast to int8
    specImageResized = np.array(specImagePil, dtype=np.uint8)
    #print('SpecImageResized.shape', SpecImageResized.shape, flush=True)
    return specImageResized


def getMelSpec(sampleVec):

    # Librosa mel-spectrum
    
    melSpec = librosa.feature.melspectrogram(y=sampleVec, sr=cfg.samplerate, n_fft=cfg.fftSizeInSamples, hop_length=cfg.fftHopSizeInSamples, n_mels=cfg.nMelBands, fmin=cfg.melStartFreq, fmax=cfg.melEndFreq, power=2.0)

    melSpec = librosa.power_to_db(melSpec, ref=np.max, top_db=100)
    
    nLowFreqsInPixelToCut = int(cfg.nLowFreqsInPixelToCutMax/2.0)
    nHighFreqsInPixelToCut = int(cfg.nHighFreqsInPixelToCutMax/2.0)

    if nHighFreqsInPixelToCut:
        melSpec = melSpec[nLowFreqsInPixelToCut:-nHighFreqsInPixelToCut]
    else:
        melSpec = melSpec[nLowFreqsInPixelToCut:]

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    melSpec = melSpec[::-1, ...]

    # Normalize values between 0 and 1
    melSpec -= melSpec.min()
    melSpec /= melSpec.max()

    maxVal = 255.9
    melSpec *= maxVal
    melSpec = maxVal-melSpec

    
    return melSpec


def getSpecSegment(audioDicts, fileIx, channelIx, startSample, endSample, segmentDurationInSamples):

    audioData = audioDicts[fileIx]['audioData']
    sampleVec = audioData[startSample:endSample, channelIx].copy()

    # Needed if segmentDurationMasked < segmentDuration or audio file is shorter than segmentDuration
    sampleVec.resize(segmentDurationInSamples) 

    specSegment = getMelSpec(sampleVec)
    specSegment = resizeSpecImage(specSegment, cfg.imageSize)

    return specSegment


def predictSegmentsInParallel(loader, model):

    model.eval() # switch to evaluate mode

    if cfg.debug:
        batch_time = AverageMeter()
        end = time.time()


    # Create output matrix tensor of size nSegments, nClasses
    outputs = torch.empty((len(loader.dataset), cfg.nClasses), dtype=torch.float32)

    n_gpus = 0
    if cfg.gpuMode:
        n_gpus = torch.cuda.device_count()

    with torch.no_grad():
        for i, sample_batched in enumerate(loader):
            
            input = sample_batched

            if cfg.gpuMode and n_gpus < 2:
                input = input.cuda(non_blocking=True)

            output = model(input) # torch.Size([bs, nClasses]) torch.float32

            #outputNp = output.cpu().data.numpy()

            # Insert output into outputs
            segmentIx = i*cfg.batchSizeInference
            outputs[segmentIx:segmentIx+output.shape[0]] = output

            if cfg.debug:
                batch_time.update(time.time() - end) # measure elapsed time
                end = time.time()
                if i % cfg.printFreq == 0:
                    logStr = time.strftime("%y%m%d-%H%M%S") + '\t'
                    logStr += ('TEST [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} '.format(
                           i, len(loader), batch_time=batch_time))
                    print(logStr, flush=True)


    return outputs.cpu().data.numpy()



def summarizePredictionsFileBased(predictions, poolingMethod='mean'):

    #nChannels, nTimeIntervals, nClasses = predictions.shape
    #print(predictions.shape, predictions.dtype, flush=True)

    #predictionsReshaped = predictions.reshape(nChannels*nTimeIntervals, nClasses)
    predictionsReshaped = predictions.reshape(-1, predictions.shape[-1])
    #print(predictionsReshaped.shape, predictionsReshaped.dtype, flush=True)
    
    # #EmphasizeExponent = 2 # 1,2,3,4
    #predictionsFileBased = np.mean(predictionsReshaped ** EmphasizeExponent, axis=0)

    if poolingMethod == 'mean':
        predictionsFileBased = np.mean(predictionsReshaped, axis=0)
    
    if poolingMethod == 'meanexp':
        predictionsFileBased = np.mean(predictionsReshaped ** 2, axis=0)
    
    if poolingMethod == 'max':
        predictionsFileBased = np.max(predictionsReshaped, axis=0)
    
    return predictionsFileBased


def getOutputInSummaryStyle(resultDict):

    # fileId (nChannels, nTimeIntervals, nClasses) --> species1 prob1, species2 prob2, species3 prob3
    summary = resultDict['summary']
    output = resultDict['fileId'] + ' ' + str(resultDict['probs'].shape) + ' --> '
    numOfPredictionsToShow = 3
    for i in range(numOfPredictionsToShow):
        classIx = summary[i]['ix']
        score = "%.1f"%(summary[i]['prob']*100.0)
        output += str(cfg.speciesData.at[classIx, cfg.nameType]) + ' ' + score + '%, '
    # Remove last ,
    output = output[:-2]

    return output

def getOutputInSummaryJsonStyle(resultDict):
    numOfPredictionsToShow = 3
    outputDict = {
        'fileId': resultDict['fileId'],
        'summary': resultDict['summary'][:numOfPredictionsToShow]
    }
    return json.dumps(outputDict, ensure_ascii=False, indent=2)


def getOutputInNaturblick2022Style(resultDict):

    # Naturblick 2022 json format:
    # {"result": [{"score": 95.0, "classId": "TurMer0", "nameDt": "Amsel", "nameLat": "Turdus merula"}, {"score": 94.9, "classId": "AcrSci0", "nameDt": "Teichrohrsänger", "nameLat": "Acrocephalus scirpaceus"}, {"score": 91.7, "classId": "FriCoe0", "nameDt": "Buchfink", "nameLat": "Fringilla coelebs"}]}
    
    # Use and convert first 3 summary results from resultDict
    # [{"ix": 3, "name": "Turdus merula", "prob": 0.9504496455192566}, ... --> {"result": [{"score": 95.0, "classId": "TurMer0", "nameDt": "Amsel", "nameLat": "Turdus merula"}, ...
    summary = resultDict['summary']
    outputDict = {}
    outputDict['result'] = []
    numOfPredictionsToShow = 3
    for i in range(numOfPredictionsToShow):
        classIx = summary[i]['ix']
        score = float("%.1f"%(summary[i]['prob']*100.0))
        outputDict['result'].append({
            'score': score,
            'classId': cfg.speciesData.at[classIx, 'id'],
            'nameDt': cfg.speciesData.at[classIx, 'de'],
            'nameLat': cfg.speciesData.at[classIx, 'sci']
        })

    return json.dumps(outputDict, ensure_ascii=False)



def mergeOverlappingLabels(df):

    # Input: dataframe with startTime [s], endTime [s], species, confidence (sorted by startTime)
    # Output: dataframe with same structure but overlapping/adjacent intervals of same species are merged

    # Algo idea from https://www.geeksforgeeks.org/merging-intervals/?ref=lbp
    # Time complexity: O(N*log(N))

    # Get distinct species
    speciesList = df['species'].unique()
    #print('speciesList', speciesList, flush=True)

    labels_new = []

    for species in speciesList:
        # Filter by species
        df_species = df[df['species']==species]
        labels = df_species.to_dict('records')

        stack = []
        # insert first label into stack
        stack.append(labels[0])
        for label in labels[1:]:
            # Check for overlapping label,
            # if label overlap
            if stack[-1]['startTime [s]'] <= label['startTime [s]'] <= stack[-1]['endTime [s]']:
                stack[-1]['endTime [s]'] = max(stack[-1]['endTime [s]'], label['endTime [s]'])
                # Use max value for merged label
                stack[-1]['confidence'] = max(stack[-1]['confidence'], label['confidence'])
            else:
                stack.append(label)

        labels_new += stack
        
    df_new = pd.DataFrame.from_dict(labels_new)

    # Sort by startTime (asc), endTime [s] (asc) and confidence (desc)
    if cfg.sortSpecies:
        df_new = df_new.sort_values(by=['startTime [s]', 'endTime [s]', 'confidence'], ascending=[True, True, False]).reset_index(drop=True)
    else:
        df_new = df_new.sort_values(by=['startTime [s]', 'endTime [s]'], ascending=[True, True]).reset_index(drop=True)

    return df_new





def writeResultToFile(resultDict, outputDir, fileOutputFormats):

    # Collect names of writen files to reference for e.g. download in browser
    outputFiles = []

    # Check if format is valid
    for fileOutputFormat in fileOutputFormats:
        if fileOutputFormat not in cfg.fileOutputFormatsValid:
            print('Warning, invalid file output format:', fileOutputFormat, flush=True)


    # ToDo: Add start/end time info to filename ?
    fileId = resultDict['fileId']

    float_format = '%.' + str(cfg.outputPrecision) + 'f'
    
    
    # Write raw data as dict to pkl files
    if 'raw_pkl' in fileOutputFormats:
        filename = fileId + '.pkl'
        path = outputDir + filename
        with open(path, 'wb') as f:
            if cfg.useFloat16InPkl:
                resultDictFloat16 = resultDict.copy()
                resultDictFloat16['probs'] = resultDictFloat16['probs'].astype(np.float16)
                pickle.dump(resultDictFloat16, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                pickle.dump(resultDict, f, protocol=pickle.HIGHEST_PROTOCOL)
            outputFiles.append(filename)


    if 'raw_csv' in fileOutputFormats or 'raw_excel' in fileOutputFormats:

        classNames = cfg.speciesData[cfg.nameType].tolist()
        nChannels = resultDict['probs'].shape[0]
        dfs = []
        for channelIx in range(nChannels):
            df = pd.DataFrame(resultDict['probs'][channelIx], columns=classNames)
            # Sort species cols by max value
            if cfg.sortSpecies:
                colMax = np.max(resultDict['probs'][channelIx], axis=0)
                colMaxIxs = np.argsort(colMax)[::-1]
                #classNamesSorted = list(np.array(classNames)[colMaxIxs])
                # Reorder cols via index
                df = df.iloc[:,colMaxIxs]

            # NNN    
            if cfg.includeFilePathInOutputFiles:
                # Add filePath, startTimes, endTimes at first, second and third column
                df.insert(0, 'filePath', resultDict['filePath'])
                df.insert(1, 'startTime [s]', resultDict['startTimes'])
                df.insert(2, 'endTime [s]', resultDict['endTimes'])
            else:
                # Add startTimes, endTimes at first and second column
                df.insert(0, 'startTime [s]', resultDict['startTimes'])
                df.insert(1, 'endTime [s]', resultDict['endTimes'])       
            

            # Format start & end times to ms precision but remove trailing zeros
            #df['startTime [s]'] = df['startTime [s]'].map(lambda x: ('%.3f' % x).rstrip('0').rstrip('.'))
            #df['endTime [s]'] = df['endTime [s]'].map(lambda x: ('%.3f' % x).rstrip('0').rstrip('.'))

            #print('df.dtypes', df.dtypes, flush=True)
            dfs.append(df)

            # Write to csv (one csv per channel)
            if 'raw_csv' in fileOutputFormats:
                filename = fileId + '_c' + str(resultDict['channels'][channelIx])
                if cfg.sortSpecies:
                    filename += '_sorted.csv'
                else:
                    filename += '.csv'
                
                path = outputDir + filename
                df.to_csv(path, index=False, sep=cfg.csvDelimiter, float_format=float_format)
                outputFiles.append(filename)

        # Write to excel file (one sheet per channel)
        if 'raw_excel' in fileOutputFormats:
            if cfg.sortSpecies:
                filename = fileId + '_sorted.xlsx'
            else:
                filename = fileId + '.xlsx'
            path = outputDir + filename
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                for channelIx in range(nChannels):
                    sheet_name = 'channel ' + str(resultDict['channels'][channelIx])
                    dfs[channelIx].to_excel(writer, index=False, sheet_name=sheet_name, float_format=float_format)
                outputFiles.append(filename)

    # Write label files (intervals of species predictions exceeding minConfidence)
    if 'labels' in ' '.join(fileOutputFormats):
        
        nSegments = resultDict['probs'].shape[1]
        preds = resultDict['probs']
        
        # ToDo: Implement channelPooling=None
        if cfg.channelPooling:
            # nChannels, nIntervals, nSpecies --> nIntervals, nSpecies
            if cfg.channelPooling == 'mean':
                preds = np.mean(preds, axis=0)
            if cfg.channelPooling == 'max':
                preds = np.max(preds, axis=0)

            df_dict = {}
            df_dict['startTime [s]'] = []
            df_dict['endTime [s]'] = []
            df_dict['species'] = []
            df_dict['confidence'] = []
            
            for segmIx in range(nSegments):
                startTime = resultDict['startTimes'][segmIx]
                endTime = resultDict['endTimes'][segmIx]
                predRow = preds[segmIx]
                
                # Get classIxs with values above threshold
                #classIxs = np.argwhere(predRow >= cfg.minConfidence).reshape(-1).tolist()
                classIxs = np.argwhere(predRow >= cfg.minConfidences).reshape(-1).tolist()

                for classIx in classIxs:
                    # Filter by species selection
                    if classIx in cfg.classIxsSelection:
                        confidence = predRow[classIx]
                        className = cfg.speciesData.at[classIx, cfg.nameType]
                        
                        df_dict['startTime [s]'].append(startTime)
                        df_dict['endTime [s]'].append(endTime)
                        df_dict['species'].append(className)
                        df_dict['confidence'].append(confidence)
                        #print(segmIx, startTime, endTime, classIx, className, confidence, flush=True)

            df = pd.DataFrame.from_dict(df_dict)

            # Sort by startTime (asc) and confidence (desc)
            if cfg.sortSpecies:
                df = df.sort_values(by=['startTime [s]','confidence'], ascending=[True, False]).reset_index(drop=True)
            
            if cfg.mergeLabels:
                df = mergeOverlappingLabels(df)

            # NNN
            if cfg.includeFilePathInOutputFiles:
                # Add filePath at last column
                df['filePath'] = resultDict['filePath']

            #print(df, flush=True)

            if 'labels_excel' in fileOutputFormats:
                filename = fileId + '_labels.xlsx'
                path = outputDir + filename
                df.to_excel(path, float_format=float_format, index=False, engine='openpyxl')
                outputFiles.append(filename)

            if 'labels_csv' in fileOutputFormats:
                filename = fileId + '_labels.csv'
                path = outputDir + filename
                df.to_csv(path, index=False, sep=cfg.csvDelimiter, float_format=float_format)
                outputFiles.append(filename)

            if 'labels_audacity' in fileOutputFormats:
                df_audacity = df.copy()

                # NNN
                if cfg.includeFilePathInOutputFiles:
                    # Remove col filePath
                    df_audacity = df_audacity.drop(columns=['filePath'])

                # Check if there are any labels (predictions above threshold)
                if len(df_audacity.index):
                    # Format to string (0.9123 --> [91.2%])
                    df_audacity['confidence'] = df_audacity['confidence'].apply(lambda x: '[' + '{:.1f}'.format(x*100.0) + '%]')
                    # Join species and confidence (species [91.2%])
                    df_audacity['label'] = df_audacity['species'] + ' ' + df_audacity['confidence']
                    df_audacity = df_audacity.drop(columns=['species', 'confidence'])
                
                filename = fileId + '_labels_audacity.txt'
                path = outputDir + filename
                df_audacity.to_csv(path, sep ='\t', header=False, index=False)
                outputFiles.append(filename)

            if 'labels_raven' in fileOutputFormats:

                df_raven = df.copy()

                df_raven.insert(loc=0, column='Selection', value=list(range(1,len(df_raven.index)+1)))
                df_raven.insert(loc=1, column='View', value='Spectrogram 1')
                df_raven.insert(loc=2, column='Channel', value=1)
                df_raven.rename(columns={"startTime [s]": "Begin Time (s)", "endTime [s]": "End Time (s)"}, inplace=True)
                df_raven.insert(loc=5, column='Low Freq (Hz)', value=cfg.melStartFreq)
                df_raven.insert(loc=6, column='High Freq (Hz)', value=cfg.melEndFreq)
                df_raven.rename(columns={"species": "Species", "confidence": "Confidence"}, inplace=True)

                filename = fileId + '.BirdID.selections.txt'
                path = outputDir + filename
                df_raven.to_csv(path, sep ='\t', index=False, float_format=float_format)
                outputFiles.append(filename)


    return outputFiles


def createTerminalOutput(resultDict, terminalOutputFormat='summary'):

    output = None

    if terminalOutputFormat == 'summary':
        output = getOutputInSummaryStyle(resultDict)

    if terminalOutputFormat == 'summaryJson':
        output = getOutputInSummaryJsonStyle(resultDict)

    if terminalOutputFormat == 'naturblick2022':
        output = getOutputInNaturblick2022Style(resultDict)

    if terminalOutputFormat == 'resultDictJson':
        resultDictJsonReady = resultDict.copy()
        # Numpy ndarray is not JSON serializable --> convert to list of lists
        resultDictJsonReady['probs'] = resultDictJsonReady['probs'].tolist()
        # Round floats to output precision
        resultDictJsonReady['probs'] = round_floats(resultDictJsonReady['probs'], cfg.outputPrecision)
        output = json.dumps(resultDictJsonReady, ensure_ascii=False, indent=2)

    if terminalOutputFormat == 'none':
        output = None


    return output


def prepareAudioDict(paramDict):

    # Create audioDict with values for path, fileId, startTime, endTime, channels, startTimes, endTimes, ..., audioData

    # Get audio path and restore config
    audioPath = paramDict['audioPath']
    cfg.setConfig(paramDict['configDict'])

    # Check if file exists
    if not os.path.isfile(audioPath):
        msg = 'Error: Cannot find file {}'.format(audioPath)
        print(msg, flush=True)
        writeErrorLog(msg)
        return None

    try:
        audioData, startTime, endTime, channels = readAndPreprocessAudioFile(
            audioPath, 
            startTime=cfg.startTime, 
            endTime=cfg.endTime,
            channels=cfg.channels,
            mono=cfg.mono
            )

        nFrames, nChannels = audioData.shape
        duration = nFrames/cfg.samplerate

        hopLength = cfg.segmentDurationMasked - cfg.segmentDurationMasked * cfg.segmentOverlapInPerc*0.01

        startTimes, endTimes = get_start_and_end_times_of_overlapping_segments(duration, segment_duration=cfg.segmentDurationMasked, step_duration=hopLength, fit_last_segment_to_global_end_time=True)
        
        audioDict = {}
        audioDict['path'] = audioPath
        audioDict['fileId'] = os.path.splitext(os.path.basename(audioPath))[0]
        audioDict['startTime'] = startTime
        #print('startTime', startTime, flush=True)
        audioDict['endTime'] = endTime
        audioDict['channels'] = channels
        audioDict['nFrames'] = nFrames
        audioDict['nChannels'] = nChannels
        audioDict['duration'] = duration
        
        audioDict['startTimes'] = startTimes
        audioDict['endTimes'] = endTimes
        audioDict['nTimeIntervals'] = len(startTimes)
        audioDict['nSegments'] = len(startTimes)*nChannels
        
        audioDict['audioData'] = audioData

        return audioDict
    
    except:
        # Print traceback
        msg = 'Error: Cannot read or preprocess file {}\n{}'.format(audioPath, traceback.format_exc())
        print(msg, flush=True)
        writeErrorLog(msg)
        return None     


def getResultDict(paramDict):

    # Create result dictionary and write output file(s)

    # Get audioDict and restore config
    audioDict = paramDict['audioDict']
    cfg.setConfig(paramDict['configDict'])

    try:

        # Adjust startTimes relative to startTime (add startTime as offset)
        startTime = audioDict['startTime']
        startTimes = audioDict['startTimes']
        endTimes = audioDict['endTimes']

        if startTime:
            startTimes = [x+startTime for x in startTimes]
            endTimes = [x+startTime for x in endTimes]


        # Create result dict
        resultDict = {}

        resultDict['modelId'] = 'birdid-europe254-2103'

        resultDict['fileId'] = audioDict['fileId']
        resultDict['filePath'] = audioDict['path']
        resultDict['startTime'] = audioDict['startTime']
        resultDict['endTime'] = audioDict['endTime']

        resultDict['channels'] = audioDict['channels']

        resultDict['segmentDuration'] = cfg.segmentDurationMasked
        
        resultDict['classIds'] = cfg.speciesData['id'].tolist()
        resultDict['classNamesScientific'] = cfg.speciesData['sci'].tolist()
        resultDict['classNamesGerman'] = cfg.speciesData['de'].tolist()
        #resultDict['classNamesEnglish'] = cfg.speciesData['en'].tolist()

        #resultDict['nChannels'] = predictions.shape[0]
        #resultDict['nSegments'] = predictions.shape[1]
        #resultDict['nClasses'] = predictions.shape[2]

        resultDict['startTimes'] = startTimes
        resultDict['endTimes'] = endTimes

        
        # Prediction Matrix: nChannels x nSegments x nClasses
        resultDict['probs'] = audioDict['probs']
        #resultDict['logits']
        #resultDict['targets']

        #import sys
        #print('args')
        #print(sys.argv)

        # Add summary
        predictionsFileBased = summarizePredictionsFileBased(resultDict['probs'], poolingMethod=cfg.segmentPooling)
        ixsBest = np.argsort(predictionsFileBased)[::-1]
        resultDict['summary'] = []
        numOfPredictionsInSummary = 5 # 8
        for i in range(numOfPredictionsInSummary):
            classIx = ixsBest[i]
            classProbability = predictionsFileBased[classIx]
            resultDict['summary'].append({
                'ix': int(classIx),
                #'id': cfg.speciesData.at[classIx, 'id'],
                #'sci': cfg.speciesData.at[classIx, 'sci'],
                #'en': cfg.speciesData.at[classIx, 'en'],
                #'de': cfg.speciesData.at[classIx, 'de'],
                'name': cfg.speciesData.at[classIx, cfg.nameType],
                #'prob': float(classProbability)
                'prob': round(float(classProbability), cfg.outputPrecision)
                })

        
        # Add infos on output files
        resultDict['outputDir'] = cfg.outputDir

        # Write results to file
        if cfg.terminalOutputFormat != 'naturblick2022':
            resultDict['outputFiles'] = writeResultToFile(resultDict, cfg.outputDir, cfg.fileOutputFormats)
        
        # Create terminal output
        terminalOutput = createTerminalOutput(resultDict, terminalOutputFormat=cfg.terminalOutputFormat)
        
        if terminalOutput:
            print(terminalOutput, flush=True)

        return resultDict
    
    except:
        # Print traceback
        msg = 'Error: Cannot create resultDict for file {}\n{}'.format(audioDict['path'], traceback.format_exc())
        print(msg, flush=True)
        writeErrorLog(msg)
        return None     

def processFiles(model, audioPath):

    if cfg.debug:
        startTimeProcessFile = time.time()


    # audioPath is file
    if os.path.isfile(audioPath):
        paths = [audioPath]
    # audioPath is directory
    else:
        paths = getAudioFilesInDir(audioPath)
    
    # Collect resultDict per file
    resultDicts = []

    # Get batches of file paths with size cfg.batchSizeFiles to read and preprocess in parallel
    for ix in range(0, len(paths), cfg.batchSizeFiles):
        paths_batch = paths[ix:ix+cfg.batchSizeFiles]
        
        # Create list of paramDicts with path and config
        # Windows doesn't support fork(), so each process needs own config
        #paramDicts = {'path': [], 'configDict': []}
        paramDicts = []
        for path in paths_batch:
            paramDicts.append({'audioPath': path, 'configDict': cfg.getConfig()})


        # Use multiprocessing if more than one file
        if len(paramDicts) > 1:
            with Pool(cfg.batchSizeFiles) as p:
                audioDicts = p.map(prepareAudioDict, paramDicts)
        else:
            audioDicts = []
            audioDicts.append(prepareAudioDict(paramDicts[0]))

        #print(audioDicts)

        # Remove items from list with errors (AudioDict=None)
        if None in audioDicts:
            audioDicts = [x for x in audioDicts if x != None]

        
        # Prepare audio dataset for inference
        normalize = transforms.Normalize(mean=cfg.imageMean, std=cfg.imageStd)
        audioDataSetObj = AudioDataset(audioDicts, transform=transforms.Compose([transforms.ToTensor(), normalize]))
        nSegmentsPerBatchOfFiles = len(audioDataSetObj)

        if cfg.debug:
            print('nSegmentsPerBatchOfFiles', nSegmentsPerBatchOfFiles, flush=True)
            print('ElapsedTime Preprocessing', time.time()-startTimeProcessFile, flush=True)


        # If inference fails (happens in rare cases), try again
        inferenceSuccess = False
        inferenceAttempts = 0
        while not inferenceSuccess and inferenceAttempts < cfg.inferenceAttemptsMax:
            try:
                # Create data loader
                loader = torch.utils.data.DataLoader(audioDataSetObj, batch_size=cfg.batchSizeInference, shuffle=False, num_workers=cfg.nCpuWorkers, pin_memory=True, timeout=cfg.dataLoaderTimeOut)
                if cfg.debug:
                    print('DataLoader created.', flush=True)
                # Get predictions (outputs of size nSegments, nClasses)
                outputs = predictSegmentsInParallel(loader, model)
                if cfg.debug:
                    print('Inference done.', flush=True)
                inferenceSuccess = True
            except:
                msg = 'Error: Inference failed at attempt {}\n{}'.format(inferenceAttempts+1, traceback.format_exc())
                print(msg, flush=True)
                writeErrorLog(msg)
                inferenceSuccess = False

            inferenceAttempts += 1
        
        

        # Iterate over files in batch and collect predictions per file
        paramDicts = []
        nFilesPerBatch = len(audioDicts)
        for filesIx in range(nFilesPerBatch):
            # Get outputs per files
            segmentOffsetIx = audioDataSetObj.segmentOffsetIxAtFileIx[filesIx]
            nSegments = audioDicts[filesIx]['nSegments']
            #print(filesIx, segmentOffsetIx, nSegmentsPerFile, flush=True)
            outputsPerFile = outputs[segmentOffsetIx:segmentOffsetIx+nSegments]
            # Reshape output of size (nSegments, nClasses) to (nChannels, nTimeIntervals, nClasses)
            predictions = outputsPerFile.reshape(audioDicts[filesIx]['nChannels'], audioDicts[filesIx]['nTimeIntervals'], -1)
            #print(filesIx, audioDicts[filesIx]['fileId'], predictions.shape, flush=True)
            # Add predictions (probs) to audioDict
            audioDicts[filesIx]['probs'] = predictions
            # Collect audioDicts and congigDicts for multiprocessing
            paramDicts.append({'audioDict': audioDicts[filesIx], 'configDict': cfg.getConfig()})

        
        # Use multiprocessing (if more than one file) to get resultDicts
        if len(paramDicts) > 1:
            with Pool(cfg.batchSizeFiles) as p:
                resultDicts = p.map(getResultDict, paramDicts)
        else:
            resultDicts = []
            resultDicts.append(getResultDict(paramDicts[0]))


    
    if cfg.debug:
        print('ElapsedTime', time.time()-startTimeProcessFile, flush=True)


    return resultDicts   









#############################################################################
#############################################################################
#############################################################################




if __name__ == "__main__":


    # On Windows calling this function is necessary.
    # On Linux/OSX it does nothing.
    freeze_support()

    #torch.multiprocessing.set_start_method('spawn')


    # Parse arguments
    parser = argparse.ArgumentParser(description='Identify birds in audio files')

    #parser.add_argument('path', type=str, metavar='', help='Path to input audio file.')
    #parser.add_argument('-p', '--path', default='', metavar='', help='path to input audio file')
    

    # Add more inference args
    parser = addAndParseConfigParams(parser)

    #args = parser.parse_args()


    #audioPath = args.path
    audioPath = cfg.inputPath




    init()
    model = loadModel()
    #processFile(model, audioPath, startTime=startTime, endTime=endTime)
    resultDicts = processFiles(model, audioPath)



