#!/usr/bin/env python
# coding: utf-8

# # Visualizing WhatsApp Chats

# ### Importing Modules:
# * re - for using regex expressions
# * pandas - for exploiting dataframes
# * matplotlib - for visualization (inline for ploting dataframes)

# In[2]:


import re
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Exported chats
# Export WhatsApp chats as follows:
# * Navigate to contact/group you want to export chat of
# * Tap on the menu (3dots in the right corner)
# * Tap on more and then tap on Export chat
# * Choose WITHOUT MEDIA from the prompt
# * Wait for a moment on the Initializing screen
# * Share through mail or any other means
# * Gather the exported text file
# 
# The text file should have a name as: WhatsApp Chat with ... .txt. Use the same below in the filename variable

# In[3]:


filename = 'Exported Chat.txt'
file = open(filename, 'r', encoding="utf8")


# In[4]:


raw_data = file.read().splitlines()
# print(raw_data)
file.close()


# #### Regex
# Regex for catching the header expression:
# 
# 12/14/18, 1:28 PM - Person 1: Hello!
# 
# (date), (12-hour-time) - (sender-name): (Message)

# In[5]:


header_regex = re.compile(r'(\d+\/\d+\/\d+)(,)(\s)(\d+:\d+)(\s)(\w+)(\s)(-)(\s[^:]+)*(:)(\s)')


# Breaking the captured regex, throughout the file and storing it in a dictionary

# In[6]:


data_dict = {'id': [], 'date': [], 'time': [], 'sender': [], 'msg': []}

i = 0
for m in raw_data:
    mo = header_regex.search(m[:50]) # shorten the search string for faster search
    if mo != None:
        data_dict['id'].append(i)
        i += 1
        date = mo.group(1)
        time = mo.group(4) + mo.group(5) + mo.group(6) # including Meridiem
        sender = mo.group(9)[1:] # ignoring the initial space
        msg = m[len(mo.group(0)):]
        data_dict['date'].append(date)
        data_dict['time'].append(time)
        data_dict['sender'].append(sender)
        data_dict['msg'].append(msg)
    else: # continued msg for a sender
        if(len(data_dict['msg']) != 0):
            data_dict['msg'][-1] += m
            
# print(data_dict)


# Converting the dictionary into the main Pandas DataFrame

# In[7]:


df = pd.DataFrame(data=data_dict)
# df.head()


# ## Sender List

# In[8]:


senders = df['sender'].unique().tolist()
print(senders)


# ## Sender Message Count

# In[9]:


sender_msg_count = df['sender'].value_counts()
print(sender_msg_count)


# ## Bar Chart showing messages sent by each sender

# In[10]:


sender_msg_count.plot(kind = 'bar', figsize = (15, 10))q


# ## Pie Chart showing messages sent by each sender

# In[11]:


sender_msg_count.plot(kind = 'pie', figsize = (10, 10))


# Calculating the total number of messages datewise, storing it in another Pandas DataFrame

# In[43]:


dates = df['date'].unique().tolist() # data list
# print(dates)

temp_dict = {'date': dates, 'msg': []} # no. of msg by date
for d in dates:
    c = 0
    for f in df['date']:
        if d == f:
            c += 1
    temp_dict['msg'].append(c)
    
# print(temp_dict)

datewise_msg_count = pd.DataFrame(data=temp_dict)
# datewise_msg_count.head()


# ## Area Chart showing the total number of messages sent by dates
# Days with zero messages are ommited

# In[44]:


datewise_msg_count.plot(x = 'date', kind = 'area', figsize = (15, 10), grid = True)


# ## Date on which maximum number of messages were sent

# In[13]:


datewise_msg_count.loc[datewise_msg_count['msg'].idxmax()]


# ## Date on which minimum number of messages were sent
# Days with zero messages are ommited

# In[14]:


datewise_msg_count.loc[datewise_msg_count['msg'].idxmin()]


# Calculating total number of messages sent by each sender on all the dates (senders with zero message in a day are included too)

# In[45]:


temp_dict_2 = {'date': dates}
for s in senders:
    temp_dict_2[s] = [] # creating lists for each sender

for s in senders:
    for d in dates:
        date_filter = df['date'] == d # date mask
        sender_filter = df['sender'] == s # sender mask
        c = df['msg'][date_filter & sender_filter].count() # counting occurences of the masked data
        temp_dict_2[s].append(c)
        
# print(temp_dict_2)
senderwise_datewise_msg_count = pd.DataFrame(data=temp_dict_2)
# senderwise_datewise_msg_count.head()


# ## Line Chart showing messaging frequency of each sender, datewise

# In[16]:


senderwise_datewise_msg_count.plot(x='date', kind = 'line', figsize = (15, 10), grid = True)


# Grouping time by 1 hour intervals throughout the day, over all dates

# In[47]:


times = [] # hour list(filters only hour from each time)
time_regex = re.compile(r'(\d+):(\d+)(\s)(\w+)')
for t in df['time']:
    mo = time_regex.search(t)
    if mo.group(4) == 'PM' and mo.group(1) == '12':
        times.append(12)
    elif mo.group(4) == 'PM':
        times.append(int(mo.group(1)) + 12)
    elif mo.group(4) == 'AM' and mo.group(1) == '12':
        times.append(0)
    else:
        times.append(int(mo.group(1)))
        
time_dict = {'time': times, 'msg': list(df['msg'])}
time_df = pd.DataFrame(data=time_dict) # contains hour and msg
# print(time_df)

time_count_dict = {'hour': [], 'msg': []} # hour and count of msgs
for i in range(0, 24):
    match = (str(i) + ':')
    time_filter = time_df['time'] == i # hour mask
    time_count_dict['hour'].append(i)
    time_count_dict['msg'].append(time_df['msg'][time_filter].count())

# print(time_count_dict)

time_count_df = pd.DataFrame(data=time_count_dict)
# time_count_df


# ## Bar Chart showing hourly messaging frequency

# In[18]:


time_count_df.plot(x = 'hour', kind = 'bar', figsize = (15, 10))


# ## Sentiment Analysis
# (Sentiment analysis only work on English words and sentences. Other languages or phonetics are ignored or thown as neutral statements)

# In[49]:


import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer() # initializing the object


# In[50]:


def senti_counter(sentences):
    '''This function returns the no. of negative, neutral and postive emotions, respectively'''
    # emotions
    neg = 0 # negative
    neu = 0 # neutral
    pos = 0 # positive

    for s in sentences:
        if s == '<Media omitted>': # skipping this
            continue

        scores = sid.polarity_scores(s) # calculating the scores
        scores.pop('compound') # removing compound emotion

        # find max of the 3 emotions
        max_senti = 'neu'
        max_val = scores[max_senti]
        for key, items in scores.items():
            if scores[key] >= max_val:
                max_senti = key
                max_val = scores[max_senti]

        # updating emotion counter
        if max_senti == 'neg':
            neg += 1
        elif max_senti == 'neu':
            neu += 1
        elif max_senti == 'pos':
            pos += 1
            
    return (neg, neu, pos) # returning them


# In[53]:


sentences = df['msg'].tolist()

neg, neu, pos = senti_counter(sentences)

total = neg + neu + pos
# print(neg, neu, pos, total)


# ### Pie Chart showing Overall Sentiment

# In[54]:


plt.pie([neg, neu, pos], labels = ['Negative', 'Neutral', 'Positive'], radius = 2)

mul = 100 / total
print('Negative: {:.2f}%, Neutral:{:.2f}%, Positive: {:.2f}%'.format(neg * mul, neu * mul, pos * mul))


# ### Pie Chart showing Senderwise Sentiment

# In[61]:


for s in senders:
    sender_filter = df['sender'] == s # sender mask
    sender_sentences = df['msg'][sender_filter].tolist()         
#     print(sender_sentences)
    
    neg, neu, pos = senti_counter(sender_sentences)
    total = neg + neu + pos
#     print(neg, neu, pos, total)
    
    mul = 100 / total
    plt.pie([neg, neu, pos], labels = ['Negative', 'Neutral', 'Positive'], radius = 1)
    plt.title('Sentiment Analysis for: ' + s)
    plt.show()
    print('Negative: {:.2f}%, Neutral:{:.2f}%, Positive: {:.2f}%\n\n\n'.format(neg * mul, neu * mul, pos * mul))


# ## Word Cloud

# In[63]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS 

whole_text = ''
for s in sentences:
    if s == '<Media omitted>':
        continue   
    whole_text += s

wordcloud = WordCloud(width = 800, height = 800, background_color ='white', stopwords = STOPWORDS, min_font_size = 10).generate(whole_text)

# plt.figure(figsize = (10, 10), facecolor = None) 
# plt.axis("off") 
# plt.tight_layout(pad = 0) 

# plt.imshow(wordcloud)  
# plt.show() 


# In[ ]:




