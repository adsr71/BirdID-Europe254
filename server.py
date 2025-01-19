'''
# Info/Help

#####  HOW TO  #####

# Inference in browser via file upload
http://localhost:4000/

# Inference by posting file
curl -F "file=@/path/to/file.mp3" "http://localhost:4000/inference"

# Inference by passing path to file in mounted volume
curl "http://localhost:4000/inference?path=/mnt/audiofile.wav"



# Get some info / metadata
http://localhost:4000/classIds
http://localhost:4000/metadata
http://localhost:4000/birds/german
http://localhost:4000/birds/english
http://localhost:4000/birds/scientific
http://localhost:4000/birds/number

'''

from flask import Flask, jsonify, flash, request, redirect, url_for, send_from_directory, make_response
from werkzeug.utils import secure_filename
import json
import os
from multiprocessing import freeze_support
import numpy as np

import argparse
import config as cfg
import inference

########  Config  ########

debug_mode = False #False #True
host = '0.0.0.0'
port = 4000
serverOutputFormat = 'summaryJson'
indent = None

serverOutputFormatsValid = ['summaryJson', 'resultDictJson', 'naturblick2022']

# To temporary upload files passed via curl or browser
uploadDirTemp = '_UploadDirTemp/'
if not os.path.exists(uploadDirTemp): os.makedirs(uploadDirTemp)


refSysLinkPrefix = 'http://www.tierstimmenarchiv.de/RefSys/SelectPreviewSpeciesViaUrl.php?Species='



##########################


app = Flask(__name__)


def uploadPostedFile(request):

    # Check if post request has file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        audioPath = os.path.join(uploadDirTemp, filename)
        file.save(audioPath)

        return audioPath

def cleanUploadDirTemp(audioPath):
    filename = os.path.basename(audioPath)
    for f in os.listdir(uploadDirTemp):
        if f.startswith(filename[:-4]):
            os.remove(os.path.join(uploadDirTemp, f))


@app.route('/inference', methods=['GET', 'POST'])
def processRequest():


    # Reload config from app start
    cfg.setConfig(cfgStart)


    # Get parameters from url

    cfg.startTime = request.args.get('startTime', default=cfg.startTime, type=float)
    cfg.endTime = request.args.get('endTime', default=cfg.endTime, type=float)
    if cfg.startTime < 0: cfg.startTime = 0.0
    if cfg.endTime is not None and cfg.endTime < cfg.startTime: cfg.endTime = None

    cfg.nCpuWorkers = request.args.get('nCpuWorkers', default=cfg.nCpuWorkers, type=int)
    cfg.batchSizeInference = request.args.get('batchSize', default=cfg.batchSizeInference, type=int)
    
    debug = request.args.get('debug', default='0', type=str)
    if debug  == '1' or debug  == 'True' or debug  == 'true':
        cfg.debug = True


    # Identify by passing path to file in mounted volume, e.g.
    # curl "http://localhost:4000/inference?path=/mnt/TestFiles/FriCoe00001.wav"
    
    #cfg.inputPath = request.args.get('path', default='', type=str)
    cfg.inputPath = request.args.get('path', default=cfg.inputPath, type=str)
    

    cfg.outputDir = request.args.get('outputDir', default=cfg.outputDir, type=str)

    # Use inputPath if outputDir not passed on init or here
    if cfg.outputDir == 'example/':
        if os.path.isfile(cfg.inputPath):
            cfg.outputDir = os.path.dirname(cfg.inputPath) + os.sep
        else: cfg.outputDir = cfg.inputPath
    
    # Add '/' to outputDir
    if cfg.outputDir[-1] != os.sep:
        cfg.outputDir += os.sep
    

    # To check
    cfg.errorLogPath = cfg.outputDir + 'error-log.txt'


    # Format of output file(s). Values in: raw_pkl,raw_excel,raw_csv,labels_excel,labels_csv,labels_audacity,labels_raven
    fileOutputFormatsStr = request.args.get('fileOutputFormats', default=cfg.fileOutputFormats, type=str)
    if isinstance(fileOutputFormatsStr, list):
        cfg.fileOutputFormats = fileOutputFormatsStr
    else:
        if fileOutputFormatsStr:
            cfg.fileOutputFormats = fileOutputFormatsStr.split(',')
        else:
            cfg.fileOutputFormats = []
    
    if cfg.fileOutputFormats and not os.path.exists(cfg.outputDir): 
        os.makedirs(cfg.outputDir)


    # Format of terminal output. Values: summary, summaryJson, naturblick2022
    #cfg.terminalOutputFormat = request.args.get('terminalOutputFormat', default=cfg.terminalOutputFormat, type=str)
    serverOutputFormatRequested = request.args.get('serverOutputFormat', default=serverOutputFormat, type=str)
    if serverOutputFormatRequested not in serverOutputFormatsValid:
        serverOutputFormatRequested = 'summaryJson'
        #print('Warning: serverOutputFormat not valid, set to summaryJson', flush=True)


    # Identify by posting file (to pass url params pass post request first)
    # curl -F "file=@/path/to/file.mp3" "http://localhost:4000/inference"
    if request.method == 'POST':
        cleanUploadDirTemp(uploadDirTemp) # Remove (previous) files in uploadDirTemp
        cfg.outputDir = uploadDirTemp
        cfg.inputPath = uploadPostedFile(request)
        #cfg.terminalOutputFormat = 'naturblick2022'

    
    audioPath = cfg.inputPath
    resultDicts = inference.processFiles(model, audioPath)



    if serverOutputFormatRequested == 'summaryJson':
        numOfPredictionsToShow = 3
        outputDict = {}
        outputDict['results'] = []
        for resultDict in resultDicts:
            outputDict['results'].append({
                'fileId': resultDict['fileId'],
                'summary': resultDict['summary'][:numOfPredictionsToShow]
            })
        output = json.dumps(outputDict, ensure_ascii=False, indent=indent)

 
    if serverOutputFormatRequested == 'resultDictJson':
        # Make JSON serializable
        resultDictsJsonReady = {}
        resultDictsJsonReady['results'] = []
        for resultDict in resultDicts:
            resultDictJsonReady = resultDict.copy()
            # Numpy ndarray is not JSON serializable --> convert to list of lists
            resultDictJsonReady['probs'] = resultDictJsonReady['probs'].tolist()
            # Round floats to output precision
            resultDictJsonReady['probs'] = inference.round_floats(resultDictJsonReady['probs'], cfg.outputPrecision)
            resultDictsJsonReady['results'].append(resultDictJsonReady)

        output = json.dumps(resultDictsJsonReady, ensure_ascii=False, indent=indent)

    if serverOutputFormatRequested == 'naturblick2022':
        output = inference.getOutputInNaturblick2022Style(resultDicts[0])


    return output


# Run in browser to upload audio file via form: http://localhost:4000/
@app.route('/', methods=['GET', 'POST'])
def upload_file_browser():

    if request.method == 'POST':

        cleanUploadDirTemp(uploadDirTemp) # Remove (previous) files in uploadDirTemp

        audioPath = uploadPostedFile(request)
        cfg.outputDir = uploadDirTemp
        cfg.terminalOutputFormat = 'summary'
        
        resultDicts = inference.processFiles(model, audioPath)
        summary = resultDicts[0]['summary']
        
        
        htmlStr = '<!doctype html>'
        htmlStr += '<title>Bird Identification</title>'
        htmlStr += '<h2>Identify Birds in Audio File</h2>'
        htmlStr += '<br>'
        htmlStr += '<form method=post enctype=multipart/form-data>'
        htmlStr += '<input type=file name=file>'
        htmlStr += '<input type=submit value=Upload>'
        htmlStr += '</form>'

        htmlStr += '<br>'

        htmlStr += '<h3>Result Summary</h3>'
        htmlStr += '<ol>'

        numOfPredictionsToShow = 3
        for rankIx in range(numOfPredictionsToShow):
            classIx = summary[rankIx]['ix']
            name = summary[rankIx]['name']
            nameSci =  cfg.speciesData.at[classIx, 'sci']
            score = float("%.1f"%(summary[rankIx]['prob']*100.0))
            
            htmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + refSysLinkPrefix + nameSci.replace(' ', '%20') + '">' + name + '</a>'
            # Add scientific name in brackets if not chosen anyway
            if name != nameSci: htmlStr += '&nbsp;(' + nameSci + ')'
            htmlStr += '&nbsp;&nbsp;[' + str(score) + '%]</li>'
        

        htmlStr += '</ol>'
        htmlStr += '<br>'
        htmlStr += '<h3>Result Files</h3>'
        
        for filename in resultDicts[0]['outputFiles']:
            htmlStr += '<a href="' + url_for('download', filename=filename) + '">' + filename + '</a><br>'

        
        return htmlStr
        
    return '''
    <!doctype html>
    <title>Bird Identification</title>
    <h2>Identify Birds in Audio File</h2>
    <br>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):

    # Offer file in uploadDirTemp for download
    print('Download', uploadDirTemp, filename, flush=True)
    return send_from_directory(directory=uploadDirTemp, path=filename, as_attachment=True)



@app.route("/classIds")
@app.route("/ClassIds")
def getClassIds():
    return toJson(cfg.speciesData['id'].tolist())


@app.route("/birds")
@app.route("/metadata")
def getBirds():
    return toJson(cfg.speciesData.to_dict('records'))


@app.route("/birds/german")
def getBirdsGerman():
    return toJson(cfg.speciesData['de'].tolist())

@app.route("/birds/english")
def getBirdsEnglish():
    return toJson(cfg.speciesData['en'].tolist())

@app.route("/birds/scientific")
def getBirdsLat():
    return toJson(cfg.speciesData['sci'].tolist())


@app.route("/birds/number")
def getBirdsNumber():
    return toJson(cfg.nClasses)


if __name__ == "__main__":

    # On Windows calling this function is necessary.
    # On Linux/OSX it does nothing.
    freeze_support()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Identify birds in audio files')

    # Flask args
    parser.add_argument('-H', '--host', default=host, metavar='', help='Host name or IP address of API endpoint server. Defaults to \'' + host + '\'')
    parser.add_argument('-p', '--port', type=int, default=port, metavar='', help='Port of API endpoint server. Defaults to ' + str(port))
    parser.add_argument('--serverOutputFormat', type=str, default=serverOutputFormat, metavar='', help='Format of server output. Value in [summaryJson, resultDictJson, naturblick2022]. Defaults to ' + serverOutputFormat + '.')

    # Inference args
    parser = inference.addAndParseConfigParams(parser)


    args = parser.parse_args()

    port = args.port
    host = args.host
    serverOutputFormat = args.serverOutputFormat

    if serverOutputFormat not in serverOutputFormatsValid:
        serverOutputFormat = 'summaryJson'
        print('Warning: serverOutputFormat not valid, set to summaryJson', flush=True)





    # Init config
    inference.init()
    
    # Load model on start
    model = inference.loadModel()

    #cfg_start = copy.deepcopy(cfg)
    cfgStart = cfg.getConfig()
    
    app.run(host=host, port=port, debug=debug_mode)

