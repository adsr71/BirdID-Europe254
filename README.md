<h1 align="center">BirdID-Europe254</h1>
<p align="center">Identification of European bird species in audio recordings.</p>

![CC BY-NC-SA 4.0][license-badge]
![Supported OS][os-badge]
![Number of species][species-badge]

[license-badge]: https://badgen.net/badge/License/CC-BY-NC-SA%204.0/green
[os-badge]: https://badgen.net/badge/OS/Linux%2C%20Windows/blue
[species-badge]: https://badgen.net/badge/Species/254/blue

## Introduction
This repository contains models and scripts to identify European bird species in audio files. The models are trained on large amounts of audio data mainly from the [MfN Animal Sound Archive Berlin](https://www.museumfuernaturkunde.berlin/en/science/animal-sound-archive), the [Reference System of Bioacoustic Data](https://www.tierstimmenarchiv.de/RefSys/Preview.php?CurLa=en) and [Xeno-canto](https://xeno-canto.org/).

## About
The models are developed at the [Museum für Naturkunde Berlin](https://www.museumfuernaturkunde.berlin/en). They are based on training deep neural networks (deep learning). Previous versions where successfully implemented and evaluated e.g. in the LifeCLEF Bird Identification Tasks [2018](https://www.imageclef.org/node/230) and [2019](https://www.imageclef.org/BirdCLEF2019). Further details on training setup and methods can be found in the corresponding CLEF papers:

[Lasseck M (2018) Audio-based Bird Species Identification with Deep Convolutional Neural Networks. In: CEUR Workshop Proceedings.](http://ceur-ws.org/Vol-2125/paper_140.pdf)

[Lasseck M (2019) Bird Species Identification in Soundscapes. In: CEUR Workshop Proceedings.](http://ceur-ws.org/Vol-2380/paper_86.pdf)

[Lasseck M (2023) Bird Species Recognition using Convolutional Neural Networks with Attention on Frequency Bands. In: CEUR Workshop Proceedings.](https://ceur-ws.org/Vol-3497/paper-175.pdf)

Current models included in this repo are advanced versions specifically designed to work well for audio files and soundscape recordings containing European bird species. They are already used in practice in different biodiversity monitoring projects like e.g. [AMMOD](https://ammod.de/) and [DeViSe](https://www.idmt.fraunhofer.de/de/institute/projects-products/projects/devise.html).

Parts of the scripts and usage scenarios are inspired by [BirdNET](https://github.com/kahst/BirdNET-Analyzer), a similar application to identify birds in audio data. BirdNET is a very popular and already well established tool for bird identification and monitoring in various projects and institutions. In contrast to our application BirdNET is based on TensorFlow instead of PyTorch, it can identify more species (including non European birds) and optionally uses additional geografic information to filter species by location. Advantages of the application in this repo over BirdNET include better GPU support (faster inference), better identification performance for some species and scenarios and more flexible input/output options (e.g. multi-channel inference, variable segment duration, selecting start/end times or channels for inference, etc.)


This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

## Content
[Setup (Ubuntu)](#setup-ubuntu)  
[Setup (Windows)](#setup-windows)  
[Usage](#usage)  
[Usage (Docker)](#usage-docker)  
[Usage (Server)](#usage-server)  


## Setup (Ubuntu)
### Without (Anaconda) environment

Install Python 3:
```
sudo apt-get update
sudo apt-get install python3-dev python3-pip
sudo pip3 install --upgrade pip
```

Install PyTorch:
```
sudo pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Install additional packages and dependencies:

```
sudo pip3 install pysoundfile librosa ffmpeg-python python-magic
sudo pip3 install flask pandas openpyxl
sudo apt-get install ffmpeg
```


### Anaconda (recommended)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install librosa ffmpeg-python timm python-magic -c conda-forge
conda install flask pandas openpyxl -c anaconda
```

Download or clone the repository

```
git clone https://github.com/adsr71/BirdID-Europe254.git
cd BirdID-Europe254
```

## Setup (Windows)
You can use Anaconda and install the requirements listed above (see Anaconda (recommended)) or download and use the already packaged version from here:

[BirdID-Europe254 for Windows](https://www.tierstimmenarchiv.de/download/birdid-europe254.zip)

1. Download zip file
2. Unpack zip file (needs approx. 5.5 GB of free disk space)
3. Navigate to extracted folder named "birdid-europe254"
4. Analyze audio files via command prompt (or Windows PowerShell) with ```inference.exe -i "path\to\folder" ...``` (see Usage section for more details)

## Usage

Run `inference.py` to analyze audio files. To set the input path, either select  a file or a folder with several audio files. If set to a folder, audio files in subfolders are also analyzed. Optionally, an output path to save result files may be set. If no output path is set, result files are saved to the folder of the input path. 

Simple example:
```
python inference.py -i /path/to/audio/folder/or/file -o /path/to/output/folder
```

Audio analysis can be customized in different ways. In most cases default parameters work across a wide range of application scenarios. Default parameters for inference can be set in config.py or changed via command line arguments. To see a list of all arguments use:

```
python inference.py -h
```

Complete list of command line arguments:

```
  -h, --help            Show this help message and exit
  -d, --debug           Show some debug infos.
  -i , --inputPath      Path to input audio file or folder. Defaults to example/.
  -s , --startTime      Start time of audio segment to analyze in seconds. Defaults to 0.
  -e , --endTime        End time of audio segment to analyze in seconds. Defaults to duration of audio file.
  --mono                Mix audio to mono before inference.
  --channels [ ...]     Audio channels to process. List of values in [1, 2, ..., #channels]. Defaults to None (using all channels).
  -sd , --segmentDuration 
                        Duration of audio chunks to analyze in seconds. Value between 1 and 5. Defaults to 5.0.
  -ov , --overlapInPerc 
                        Overlap of audio chunks to analyze in percent. Value between 0 and 80. Defaults to 60.
  -m , --modelSize      Model size. Value in [small, medium, large]. Defaults to medium.
  -f , --batchSizeFiles 
                        Number of files to preprocess in parallel (if input path is directory). Defaults to 16.
  -b , --batchSizeInference 
                        Number of segments to process in parallel (by GPU). Defaults to 8.
  -c , --nCpuWorkers    Number of CPU workers to prepare segments for inference. Defaults to 4.
  -o , --outputDir      Directory for result output file(s). Defaults to directory of input path.
  --fileOutputFormats [ ...]
                        Format of output file(s). List of values in [raw_pkl, raw_excel, raw_csv, labels_excel, labels_csv, labels_audacity, labels_raven]. Defaults to raw_pkl raw_excel labels_excel].
  --minConfidence       Minimum confidence threshold. Value between 0.01 and 0.99. Defaults to 0.75.
  --channelPooling      Pooling method to aggregate predictions from different channels. Value in [max, mean]. Defaults to max.
  --mergeLabels         Merge overlapping/adjacent species labels.
  --nameType            Type or language for species names. Value in [sci, en, de, ...]. Defaults to de.
  --useFloat16InPkl     Reduce pkl file size by casting prediction values to float16
  --outputPrecision     Number of decimal places for values in text output files. Defaults to 5.
  --sortSpecies         Sort order of species columns in raw data files or rows in label files by max value.
  --includeFilePathInOutputFiles 
                        Include file path in output files.
  --csvDelimiter        Delimiter used in CSV files. Defaults to ;.
  --speciesPath         Path to custom species metadata file or folder. If a folder is provided, the file needs to be named "species.csv". Defaults to species.csv.
  --errorLogPath        Path to error log file. Defaults to error-log.txt in outputDir.
  --terminalOutputFormat 
                        Format of terminal output. Value in [summary, summaryJson, ammodJson, naturblick2022, resultDictJson]. Defaults to summary.
```

### Examples for changing inference behavior with command line arguments


Set start and end time in seconds by passing `-s` and `-e` (or `--startTime` and `--endTime`) to select a certain part of the recording for inference, e.g. first 10 seconds:
```
python inference.py -i example/ -o example/ -s 0.0 -e 10.0
```

To mix a stereo or multi-channel audio file to mono before analyzing it, pass `--mono`. You can also pass a list of channels, so inference is performed only on the selected channels. For instance, to select only the first and last channel of a 4-channel recording use `--channels`:
```
python inference.py -i example/ -o example/ --channels 1 4
```

Usually, inference is successively done on 5 second intervals because audio segments of this duration were originally used for training. You can set segment duration to smaller values (between 1 and 5 seconds). This leads to a higher time resolution of output results and usually to more accurate onset/offset times of detected sound events. Smaller segment durations can also increase identification performance for some species, especially in soundscapes with many different birds calling at the same time. However, in some cases, performance might decrease because certain birds and song types need longer intervals for reliable identification. For example to set segment duration to 3 seconds (default duration in BirdNET) use argument `-sd` or `--segmentDuration`:
```
python inference.py -i example/ -o example/ -sd 3
```

Overlap of analyzed segments can be set in percent via `-ov` or `--overlapInPerc`. For instance to analyze 5-second segments with a step size of one second use an overlap of 80%:
```
python inference.py -i example/ -o example/ -ov 80
```

The repo includes three models of different sizes. All models are trained on the same data but differ in the number of layers and parameters. Larger models usually give better identification results but need more computing resources and time for inference. If run on small devices like Raspberry Pi or in real-time, the small model might be the better or even only option. Results of different models can be ensembled in a post-processing step (late fusion) to further improve identification performance. The small model uses an EfficientNet B0, the medium model an EfficientNet B2 and the large model an EfficientNet V2 backbone. To switch model size use `-m` or `--modelSize`:
```
python inference.py -i example/ -o example/ -m small
```

If input is a folder with several audio files, analysis can be accelerated by preprocessing multiple files in parallel. Use `-f` or `--batchSizeFile` to specify how many files to read and preprocess at the same time.
If you use one or multiple GPUs to accelerate inference you can also pass the number of batches to be processed in parallel by the GPUs via `-b` or `--batchSizeInference`. The maximum number of batches depends on the selected model size and the amount of RAM on your GPUs.
With `-c` or `--nCpuWorkers` you can choose the number of CPU threads to prepare the audio segments in parallel for inference.

For single short audio files, small values for `batchSizeInference` and `nCpuWorkers` should be chosen (`batchSizeFile` will be set to 1 by default if only a single file is passed). If you pass files with large durations or a folder with many files, batch sizes and number of CPU workers can be set as high as you computing resources allow to increase processing speed.

Example of fast inference for a small file:
```
python inference.py -i small.wav -b 4 -c 2
```
Example of fast inference for a large file or folder with many files:
```
python inference.py -i folder/with/many/files -f 16 -b 32 -c 16
```

Analysis results can be customized in various ways. You can select different output and file formats. Output files have the same name as the original audio file (but with different extensions and/or file types depending on output type and format). A list of desired output files can be passed via `--fileOutputFormats`. Following formats can be selected:

`raw_pkl`

Raw results can be saved in a dictionary as a binary (pickle) file for further post-processing in python. The result dictionary holds information and data accessible via keys, e.g.: `modelId`, `fileId`, `filePath`, `startTime`, `endTime`, `channels`, `classIds`, `classNamesScientific`, `classNamesGerman`, `startTimes`, `endTimes` and `probs`. With `startTimes` and `endTimes` you can access the list of start and end times for each audio segment that was analyzed. With `probs` you get access to a three-dimensional numpy array that holds prediction probabilities for all channels, audio segments and species. It therefore always has the shape: [number of channels, number of time intervals, number of species].

`raw_excel` / `raw_csv`

Raw results can also be saved as Excel and/or CSV files. In Excel files, results for each channel are saved in separate sheets. For CSV, results for each channel are saved in separate files with the channel information added to the filename (e.g. filename_c1.csv for first channel results). Output files consist of a header line and rows for each time interval. Each row has two columns for start and end time of the audio segment and additional 254 columns for each species holding the prediction probability for the particular species and time interval.

`labels_excel` / `labels_csv` / `labels_audacity` / `labels_raven`

Besides saving raw data, results can also be post-processed and aggregated to allow user-friendly access to the more relevant information on what species was identified at what time within the audio recording. So instead of outputting probabilities for all species and time intervals, labels are created only for those species and times where the model's prediction probability exceeds a minimum confidence threshold. Resulting label files can be saved in the following formats: Excel, CSV, Audacity label track and Raven selection table.

Example for saving results as raw data and aggregated labels in Excel format:
```
python inference.py -i example/ -o example/ --fileOutputFormats raw_excel labels_excel
```

The minimum confidence value (prediction probability threshold) necessary to decide if a species was identified and will be labeled can be set by passing a value between 0.01 and 0.99 to `--minConfidence`. Classifications below the threshold are not shown in the output label file. Higher confidence values lead to better precision, lower values to better recall rates.

For label files predictions of multi-channel audio files are pooled or aggregated by taking either the mean or maximum value of each channel. The pooling method can be selected by passing `mean` or `max` to `--channelPooling`.

Species labels are given for each analysis time interval. With `--mergeLabels` adjacent or overlapping time intervals with labels of the same species are merged together.

How species are named can be customized by passing a name type via `--nameType`. Possible types/languages are: Scientific (`sci`), English (`en`), German (`de`), short Identifier (`id`) or index number (`ix`).

Output in result files can be further customized by passing additional arguments. To save storage space, the size of binary pickel files can be reduced by passing `--useFloat16InPkl` to store 16 Bit instead of 32 Bit floats. For float values in text output files the number of decimal places can be changed via `--outputPrecision`. The columns in raw data text files and the label rows in label files within the same time interval can be sorted in descending order regarding species prediction confidence by passing `--sortSpecies`. With `--csvDelimiter` you can select the delimiter used in CSV files. With `--includeFilePathInOutputFiles` file paths of input files are included in result files.

By default, all 254 species are predicted. The species are listed in the file `species.csv` (including scientific and common names in different languages). A custom species list can be created to filter output results by modifying the original CSV file. This can be useful e.g. if you know what species to expect in the location of your recording area or if you are interested in identifying only certain species in your audio files. Just make a copy of the original file and remove all rows with species you don't want to be predicted in the output label files. You only have to make sure, to not change the data in the remaining rows and that there are no empty lines between rows. The path to the modified species file can then be specified via  `--speciesPath /path/to/folder/or/file`. If you pass a folder, the custom CSV file needs to be placed in that folder and has to be named `species.csv`.

The custom species CSV file can also be used to assign individual minimum confidence thresholds to certain species. For this, add custom threshold values (between 0.01 and 0.99) at the end of rows for those species that should have a threshold different to the global minimum confidence threshold. By applying individual thresholds, you can control precision/recall trade-off separately for each species. 

The following table gives an example of a custom species CSV file. Only four species are identified and labeled. For Turdus philomelos (Song Thrush) a minimum confidence threshold of 0.85 is assigned. For the other three species the global minimum confidence threshold is used (either default value from config.py or the value passed via `--minConfidence`).

ix|id|sci|de|en|minConfidence
---|---|---|---|---|---
0|ParMaj0|Parus major|Kohlmeise|Great Tit|
1|FriCoe0|Fringilla coelebs|Buchfink|Common Chaffinch|
3|TurMer0|Turdus merula|Amsel|Common Blackbird|
10|TurPhi0|Turdus philomelos|Singdrossel|Song Thrush|0.85

Errors during analysis are saved to an error log file. You can specify its path via `--errorLogPath`. If no path is assigned, the file is named `error-log.txt` and saved to the output directory.

With `--terminalOutputFormat` you can control the terminal output. By default a summary is printed for each input file listing the Top 3 species with the highest confidence scores identified in the entire recording.


To accomplish inference behaviour and output results similar to BirdNET default settings, use e.g.:
```
python inference.py -i example/ -o example/ -sd 3 -ov 0 --mono --fileOutputFormats labels_raven
```

## Usage (Docker)

Install docker for Ubuntu:
```
sudo apt install docker.io
```

Build Docker container:
```
sudo DOCKER_BUILDKIT=1 docker build -f Dockerfile -t birdid-europe254 .
```

<b>NOTE</b>: You need to run docker build again whenever you make changes to the scripts.

In order to pass a directory that contains your audio files to the docker file, you need to mount it inside the docker container with -v /my/path:/mount/path before you run the container.

You can run the container for the provided example files with:
```
sudo docker run -v $PWD/example:/audio birdid-europe254 python inference.py -i /audio -o /audio
```

You can adjust the directory that contains your recordings by providing an absolute path:
```
sudo docker run -v /path/to/audio/files:/audio birdid-europe254 python inference.py -i /audio -o /audio
```

You can also mount more than one folder, e.g., if input and output folder should be different:
```
sudo docker run -v /path/to/audio/files:/input -v /path/to/output/folder:/output birdid-europe254 python inference.py -i /input -o /output
```

For larger files or folders with many audio files you can accelerate analysis by using GPUs. If inference should be run on GPUs you need to pass `--gpus all`. To assign a specific GPU to the docker container (in case of multiple GPUs available in your machine) pass the device, e.g. `--gpus device=0`. PyTorch uses shared memory to share data between processes. To increase shared memory size you might also want to pass `--ipc=host`.

<b>NOTE</b>: To run docker container with GPUs you need to install the `nvidia-container-toolkit` package as per [official documentation at Github](https://github.com/NVIDIA/nvidia-docker).

Example of analysis with GPU support (first device) and increased shared memory:
```
sudo docker run -v /path/to/audio/files:/audio --gpus device=0 --ipc=host birdid-europe254 python inference.py -i /audio -o /audio
```

See Usage section above for more command line arguments, all of them will also work with Docker version.

<b>NOTE</b>: If you like to specify a custom species list (which will be used as post-filter), you need to put it into a folder that also has to be mounted.


## Usage (Server)

You can host your own analysis service by launching `server.py`. With this you can send files to the service, analyze them and send identification results back to a client. This could be a local service, running on a desktop PC or a remote server. The service can be accessed locally or remotely through a browser or via REST API e.g. using cURL.

When starting the service, you can specify a host name or IP address and a port number. The following example runs a local service on port 4000:
```
python server.py --host localhost --port 4000
```

Additionaly you can pass all command line arguments offered for `inference.py` (see Usage section). With `python server.py -h` you get the complete list of parameter options.

If you enter server IP and port number in a web browser (e.g. http://127.0.0.1:4000/) you can upload and analyze audio files with a simple web interface. Alternatively, you can analyze audio files via REST API with e.g. cURL commands. 

To upload and analyze a file on client side use:
```
curl -F "file=@/path/to/file.mp3" "http://127.0.0.1:4000/inference"
```
If the file is located on the server or a folder of audio files is mounted where the service runs, you don't need to upload the file(s). Instead just pass the file or a folder location per GET request:
```
curl "http://127.0.0.1:4000/inference?path=/path/to/audio/folder/or/file"
```
It is also possible to control input/output behavior for each client request individually by using additional URL parameters. Following query strings can be used (see Usage section for explanation): `startTime`, `endTime`, `nCpuWorkers`, `batchSize`, `outputDir`, `fileOutputFormats`, `serverOutputFormat`.

The service responds with a json string containing analysis results per audio file. The output format can be set via `serverOutputFormat` either globally or individually for each request. Valid output formats are e.g. `summaryJson` (default) listing the Top 3 species with their identification probabilities and `resultDictJson` for complete analysis results per file.
