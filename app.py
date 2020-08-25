from flask import Flask,url_for,request,render_template,jsonify,send_file,flash,redirect
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import json
from whatsappstats import *

import re
import regex
import pandas as pd
import numpy as np

# NLP Pkgs
import spacy
from textblob import TextBlob 
nlp = spacy.load('en')

# WordCloud & Matplotlib Packages
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from io import BytesIO
import random
import time
import os


from transformers import T5ForConditionalGeneration, T5Tokenizer


# Initialize App
app = Flask(__name__)
Bootstrap(app)

#Configuration file
app.config.from_pyfile('config.cfg')

@app.route('/')
def about():
	return render_template('about.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	# Receives the input query from form
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		# Analysis
		docx = nlp(rawtext)
		# Tokens
		custom_tokens = [token.text for token in docx ]
		# Word Info
		custom_wordinfo = [(token.text,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
		custom_postagging = [(word.text,word.tag_,word.pos_,word.dep_) for word in docx]
		# NER
		custom_namedentities = [(entity.text,entity.label_)for entity in docx.ents]
		blob = TextBlob(rawtext)
		blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
		# allData = ['Token:{},Tag:{},POS:{},Dependency:{},Lemma:{},Shape:{},Alpha:{},IsStopword:{}'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
		allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","Lemma":"{}","Shape":"{}","Alpha":"{}","IsStopword":"{}"'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop)) for token in docx ]

		result_json = json.dumps(allData, sort_keys = False, indent = 2)

		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,custom_tokens=custom_tokens,custom_postagging=custom_postagging,custom_namedentities=custom_namedentities,custom_wordinfo=custom_wordinfo,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,final_time=final_time,result_json=result_json)

# API ROUTES
@app.route('/api')
def basic_api():
	return render_template('restfulapidocs.html')

# API FOR TOKENS
@app.route('/api/tokens/<string:mytext>',methods=['GET'])
def api_tokens(mytext):
	# Analysis
	docx = nlp(mytext)
	# Tokens
	mytokens = [token.text for token in docx ]
	return jsonify(mytext,mytokens)

# API FOR LEMMA
@app.route('/api/lemma/<string:mytext>',methods=['GET'])
def api_lemma(mytext):
	# Analysis
	docx = nlp(mytext.strip())
	# Tokens & Lemma
	mylemma = [('Token:{},Lemma:{}'.format(token.text,token.lemma_))for token in docx ]
	return jsonify(mytext,mylemma)

# API FOR NAMED ENTITY
@app.route('/api/ner/<string:mytext>',methods=['GET'])
def api_ner(mytext):
	# Analysis
	docx = nlp(mytext)
	# Tokens
	mynamedentities = [(entity.text,entity.label_)for entity in docx.ents]
	return jsonify(mytext,mynamedentities)

# API FOR NAMED ENTITY
@app.route('/api/entities/<string:mytext>',methods=['GET'])
def api_entities(mytext):
	# Analysis
	docx = nlp(mytext)
	# Tokens
	mynamedentities = [(entity.text,entity.label_)for entity in docx.ents]
	return jsonify(mytext,mynamedentities)


# API FOR SENTIMENT ANALYSIS
@app.route('/api/sentiment/<string:mytext>',methods=['GET'])
def api_sentiment(mytext):
	# Analysis
	blob = TextBlob(mytext)
	mysentiment = [ mytext,blob.words,blob.sentiment ]
	return jsonify(mysentiment)

# API FOR MORE WORD ANALYSIS
@app.route('/api/nlpiffy/<string:mytext>',methods=['GET'])
def nlpifyapi(mytext):

	docx = nlp(mytext.strip())
	allData = ['Token:{},Tag:{},POS:{},Dependency:{},Lemma:{},Shape:{},Alpha:{},IsStopword:{}'.format(token.text,token.tag_,token.pos_,token.dep_,token.lemma_,token.shape_,token.is_alpha,token.is_stop) for token in docx ]
	
	return jsonify(mytext,allData)
	

@app.route('/images/<mytext>')
def images(mytext):
    return render_template("index.html", title=mytext)

@app.route('/fig/<string:mytext>')
def fig(mytext):
    plt.figure(figsize=(20,10))
    wordcloud = WordCloud(background_color='white', mode = "RGB", width = 2000, height = 1000).generate(mytext)
    plt.imshow(wordcloud)
    plt.axis("off")
    img = BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')


@app.route('/text')
def index():
	return render_template('index.html')

@app.route('/whatsapp')
def whatsapp():
	return render_template('whatsapp.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSION']


@app.route('/whatsapp', methods=['GET', 'POST'])
def upload_file():
	
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', category='error')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file', category='error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            openfile(filepath,filename)
            whats_stats(filepath, filename) 
            return render_template('results.html')
    return render_template('whatsapp.html')

def startsWithDateAndTime(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -' 
    result = re.match(pattern, s)
    if result:
        return True
    return False

def FindContact(s):
    s=s.split(":")
    if len(s)==2:
        return True
    else:
        return False

def getDataPoint(line):   
    splitLine = line.split(' - ') 
    dateTime = splitLine[0]
    date, time = dateTime.split(', ') 
    message = ' '.join(splitLine[1:])
    if FindContact(message): 
        splitMessage = message.split(': ') 
        contact = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        contact = None
    return date, time, contact, message

parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe

def openfile(filepath, filename):
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), encoding="utf-8") as fp:
        fp.readline() # Skipping first line of the file because contains information related to something about end-to-end encryption
        messageBuffer = [] 
        date, time, contact = None, None, None
        while True:
            line = fp.readline() 
            if not line: 
                break
            line = line.strip() 
            if startsWithDateAndTime(line): 
                if len(messageBuffer) > 0: 
                    parsedData.append([date, time, contact, ' '.join(messageBuffer)]) 
                messageBuffer.clear() 
                date, time, contact, message = getDataPoint(line) 
                messageBuffer.append(message) 
            else:
                messageBuffer.append(line)

def textData(filepath,filename):
    
    df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Contact', 'Message']) # Initialising a pandas Dataframe.
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_csv(os.path.join('csvs',filename),index=False)
    return df


@app.route('/results',methods=['GET'])
def whats_stats(filepath,filename):
    # TXT to CSV
    
    textData(os.path.join(app.config['UPLOAD_FOLDER'],filename),filename)
    
    # Read CSV
    df = pd.read_csv(os.path.join('csvs',filename))
    
    # functions to get the facts/information about the data
    contacts = df.Contact.unique()
    most(filename)
    word(filename)
    week(filename)
    msgs = number_of_msgs(filename)
    member = number_of_unique_members(filename)
    sdate = start_date(filename)
    edate = end_date(filename)
    avg = average_length_msg(filename)[:4]
    maxl , name = max_length_msg(filename)
    month = month_busy(filename)
    day =  weekday_busy(filename)

    return render_template('results.html',name=filename, source=whats_stats)

# Text Summarisation

model_size = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(model_size)
tokenizer = T5Tokenizer.from_pretrained(model_size)

def summarise(text):
    tokens = tokenizer.encode(text, 
                            max_length=256,
                            return_tensors='pt',
                            pad_to_max_length=True)

    gen_tokens = model.generate(tokens, max_length = 256)
    return tokenizer.decode(gen_tokens.tolist()[0])

@app.route('/summary', methods=['GET'])
def summary():
    return render_template("summary.html")


@app.route('/api/v1/summarize', methods=['POST','GET'])
def api_summarize():
    if request.method == 'POST':
        text = request.form.get('text')
        return summarise(text)

if __name__ == '__main__':
	app.run(debug=True)


