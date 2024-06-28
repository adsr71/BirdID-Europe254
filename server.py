#!/usr/bin/python
# -*- coding: utf-8 -*-

# sources: 
    # https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask
    # http://flask.pocoo.org/docs/1.0/patterns/fileuploads/

'''
# Info/Help

# Run docker
docker run -it -p 4000:4000 --ipc=host --name birdid --rm birdid-europe254-2103-flask-v05

# Run docker with gpu support and mounted volume
DATADIR=/path/to/audiodir
docker run -it -p 4000:4000 --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -v $DATADIR:/mnt --ipc=host --name birdid --rm birdid-europe254-2103-flask-v05



#####  HOW TO  #####

# Identify in browser via file upload
http://localhost:4000/

# Identify by posting file
curl -F "file=@/path/to/file.mp3" "http://localhost:4000/identify"

# Identify by passing path to file in mounted volume
curl "http://localhost:4000/identify?path=/mnt/audiofile.wav"



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

import argparse
import config as cfg
import inference

########  Config  ########

debug_mode = False #False #True
host = '0.0.0.0'
port = 4000


# To temporary upload files passed via curl
#uploadDirTemp = '_UploadDirTemp/'
uploadDirTemp = '_UploadDirTemp/'
if not os.path.exists(uploadDirTemp): os.makedirs(uploadDirTemp)


refSysLinkPrefix = 'http://www.tierstimmenarchiv.de/RefSys/SelectPreviewSpeciesViaUrl.php?Species='



##########################


app = Flask(__name__)



if debug_mode:
    app.config['JSON_AS_ASCII'] = False # "Drosselrohrs\u00e4nger" --> "Drosselrohrsänger" (possible security risc!)




def toJson(obj):
    if debug_mode:
        return jsonify(obj)
    else:
        return json.dumps(obj, ensure_ascii=False) # no need for app.config['JSON_AS_ASCII'] = False

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


@app.route('/identify', methods=['GET', 'POST'])
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
    # curl "http://localhost:4000/identify?path=/mnt/TestFiles/FriCoe00001.wav"
    #audioPath = request.args.get('path', default='', type=str)
    cfg.inputPath = request.args.get('path', default='', type=str)
    

    cfg.outputDir = request.args.get('outputDir', default=cfg.outputDir, type=str)

    # Use inputPath if outputDir not passed on init or here
    if cfg.outputDir == 'example/':
        if os.path.isfile(cfg.inputPath):
            cfg.outputDir = os.path.dirname(cfg.inputPath) + os.sep
        else: cfg.outputDir = cfg.inputPath

    # Add '/' to outputDir
    if cfg.outputDir[-1] != os.sep:
        cfg.outputDir += os.sep
    
    if not os.path.exists(cfg.outputDir): os.makedirs(cfg.outputDir)

    # To check
    cfg.errorLogPath = cfg.outputDir + 'error-log.txt'


    # Format of output file(s). Values in: pkl,csv,csv_sorted
    fileOutputFormatsStr = request.args.get('fileOutputFormats', default=cfg.fileOutputFormats, type=str)
    if isinstance(fileOutputFormatsStr, list):
        cfg.fileOutputFormats = fileOutputFormatsStr
    else:
        if fileOutputFormatsStr:
            cfg.fileOutputFormats = fileOutputFormatsStr.split(',')
        else:
            cfg.fileOutputFormats = []
    



    # Format of terminal output. Values: summery, summeryJson, naturblick2022, ammodJson
    cfg.terminalOutputFormat = request.args.get('terminalOutputFormat', default=cfg.terminalOutputFormat, type=str)


    # Identify by posting file (to pass url params pass post request first)
    # curl -F "file=@/path/to/file.mp3" "http://localhost:4000/identify"
    if request.method == 'POST':
        cfg.inputPath = uploadPostedFile(request)

    
    audioPath = cfg.inputPath
    terminalOutputs, outputPaths = inference.processFiles(model, audioPath)

    

    if request.method == 'POST' and os.path.isfile(audioPath):
        cleanUploadDirTemp(audioPath) # Remove files in uploadDirTemp


    # ToDo: handle (naturblick2022) output (if audioPath is dir) --> terminalOutputs is now list
    # Deal with audioPath is file --> list has only 1 item vs. dir
    # Deal with terminalOutput types

    if cfg.terminalOutputFormat == 'naturblick2022':
        output = terminalOutputs[0]
    else:
        output = json.dumps(terminalOutputs, ensure_ascii=False, indent=2)

    return output


# Run in browser to upload audio file via form: http://localhost:4000/
@app.route('/', methods=['GET', 'POST'])
def upload_file_browser():

    if request.method == 'POST':

        cleanUploadDirTemp(uploadDirTemp) # Remove (previous) files in uploadDirTemp

        audioPath = uploadPostedFile(request)
        cfg.outputDir = uploadDirTemp
        cfg.terminalOutputFormat = 'summeryJson'
        terminalOutputs, outputPaths = inference.processFiles(model, audioPath)
        
        # ToDo better
        output = json.loads(terminalOutputs[0])


        htmlStr = '<!doctype html>'
        htmlStr += '<title>Bird Identification</title>'
        htmlStr += '<h1>Upload Audio File</h1>'
        htmlStr += '<form method=post enctype=multipart/form-data>'
        htmlStr += '<input type=file name=file>'
        htmlStr += '<input type=submit value=Upload>'
        htmlStr += '</form>'

        htmlStr += '<h3>Results</h3>'
        htmlStr += '<ol>'
        htmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + refSysLinkPrefix + output['result'][0]['nameLat'].replace(' ', '%20') + '">' + output['result'][0]['nameDt'] + '</a>&nbsp;(' + output['result'][0]['nameLat'] + ')&nbsp;&nbsp;[' + str(output['result'][0]['score']) + '%]</li>'
        htmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + refSysLinkPrefix + output['result'][1]['nameLat'].replace(' ', '%20') + '">' + output['result'][1]['nameDt'] + '</a>&nbsp;(' + output['result'][1]['nameLat'] + ')&nbsp;&nbsp;[' + str(output['result'][1]['score']) + '%]</li>'
        htmlStr += '<li><a target="_blank" rel="noopener noreferrer" href="' + refSysLinkPrefix + output['result'][2]['nameLat'].replace(' ', '%20') + '">' + output['result'][2]['nameDt'] + '</a>&nbsp;(' + output['result'][2]['nameLat'] + ')&nbsp;&nbsp;[' + str(output['result'][2]['score']) + '%]</li>'
        htmlStr += '</ol>'

        htmlStr += '<br>'

        for outputPath in outputPaths:
            filename = os.path.basename(outputPath)
            htmlStr += '<a href="' + url_for('download', filename=filename) + '">' + filename + '</a><br>'

        return htmlStr
        
    return '''
    <!doctype html>
    <title>Bird Identification</title>
    <h1>Upload Audio File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):

    # Offer file in uploadDirTemp for download
    print('Download', uploadDirTemp, filename)
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
    parser = argparse.ArgumentParser(description='Identify birds in audio file')

    # Flask args
    parser.add_argument('-H', '--host', default=host, metavar='', help='Host name or IP address of API endpoint server. Defaults to \'' + host + '\'')
    parser.add_argument('-p', '--port', type=int, default=port, metavar='', help='Port of API endpoint server. Defaults to ' + str(port))


    # Inference args
    parser = inference.addAndParseConfigParams(parser)


    args = parser.parse_args()

    port = args.port
    host = args.host





    # Init config
    inference.init()
    
    # Load model on start
    model = inference.loadModel()

    #cfg_start = copy.deepcopy(cfg)
    cfgStart = cfg.getConfig()
    
    app.run(host=host, port=port, debug=debug_mode)

