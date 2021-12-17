from __future__ import print_function, division
import sys
import pandas as pd
import numpy as np
import math

def compute_lh_fscore_unigram(dat,alpha):
   # Calculate the log probabilities of the test data for each of the unigram language models (sexist and non-sexist), 
   # smoothed by alpha, classify tweets, and calculate precision, recall and fscore.                                                                                                                      
   prob_test=pd.DataFrame(columns=['judgment','tweet','logprobsex','logprobnsex','predicted'])
   comprob_sex_test=list()
   comprob_nsex_test=list()
   tweets_test=list()
   keys_sex=list(sounds_sex)
   keys_nsex=list(sounds_nsex)

   for tweet in dat['tweet']:
      b=1
      h=1
      # Calculate the log probability of each character in the tweet, and add that to the total so that we get
      # the compound probability of the whole tweet.
      for letter in tweet:
           if letter in keys_sex:
               sounds_sex_freqs[letter]=(sounds_sex[letter]+alpha)/(total_no_sounds_sex+(alpha*len(sounds_sex)))
               b=b+math.log(sounds_sex_freqs[letter])
           else:
               sounds_sex_freqs[letter]=(sounds_sex['unk']+alpha)/(total_no_sounds_sex+(alpha*len(sounds_sex)))
               b=b+math.log(sounds_sex_freqs['unk'])
           if letter in keys_nsex:
               sounds_nsex_freqs[letter]=(sounds_nsex[letter]+alpha)/(total_no_sounds_nsex+(alpha*len(sounds_nsex)))
               h=h+math.log(sounds_nsex_freqs[letter])
           else:
               sounds_nsex_freqs[letter]=(sounds_nsex['unk']+alpha)/(total_no_sounds_nsex+(alpha*len(sounds_nsex)))
               h=h+math.log(sounds_nsex_freqs['unk'])

      sounds_sex_freqs['unk']=sounds_sex['unk']+alpha/(total_no_sounds_sex + (alpha*len(sounds_sex_freqs)))
      sounds_nsex_freqs['unk']=(sounds_nsex['unk']+alpha)/(total_no_sounds_nsex+(alpha*len(sounds_nsex_freqs)))

      comprob_sex_test.append(b)
      comprob_nsex_test.append(h)
      tweets_test.append(tweet)

   # Calculate the prior log probabilities of a tweet being sexist / not sexist in the test data.       
   prior_sexist=math.log(len(testdat[testdat['judgment']=='sexism'])/len(testdat))
   prior_nonsexist=math.log(len(testdat[testdat['judgment']!='sexism'])/len(testdat))

   # Next, calculate the prob of EACH TWEET being sexist and  nonsexist by multiplying the log probabilities of
   # the sexist 
   # and nonsexist data from the language models by the prior probabilities (frequencies of sexist and 
   # nonsexist data in the  test data set). 
                                               
   # First make sure logprobsex and logprobnsex are initialized to 0.               
   prob_test['logprobsex']=0
   prob_test['logprobnsex']=0

   # Put the compound probabilities in the dataframe.
   prob_test['logprobsex']=comprob_sex_test
   prob_test['logprobnsex']=comprob_nsex_test

   #Add two log probs (equivalent of multiplication in regular prob space).               
   prob_test['logprobsex']=prob_test['logprobsex'].apply(lambda x: x  +  prior_sexist)
   prob_test['logprobnsex']= prob_test['logprobnsex'].apply(lambda x:x +  prior_nonsexist)

   prob_test['tweet']=tweets_test
   prob_test['judgment']=dat['judgment']
   prob_test['predicted']=np.where(prob_test['logprobsex']>prob_test['logprobnsex'],'sexism','neither')

   correct_preds_sex = len(prob_test[((prob_test['judgment']=='sexism')& (prob_test['predicted']=='sexism'))])
   all_pred_sexist=len(prob_test[prob_test['predicted']=='sexism'])
   all_judg_sexist=len(prob_test[prob_test['judgment']=='sexism'])

   precision_sexist=correct_preds_sex/all_pred_sexist
   recall_sexist=all_pred_sexist/all_judg_sexist
   fscore_sexist=(2*precision_sexist*recall_sexist)/(precision_sexist+recall_sexist)

   return (precision_sexist,recall_sexist,fscore_sexist)

def compute_lh_fscore_bigram(dat2,beta):
   # Calculate the log probabilities of the test data for each of the bigram language models (sexist and 
   # non-sexist), smoothed by beta (dirichlet smoothing), classify tweets, and calculate precision, 
   # recall and fscore.  
   
   prob_test_bg=pd.DataFrame(columns=['judgment','tweet','logprobsex','logprobnsex','predicted','form_nsex'])
   comprob_sex_test_bg=list()
   comprob_nsex_test_bg=list()
   tweets_bg=list()
   judgments_bg=list()
   total_bigrams_a=0
   total_bigrams_a2=0
   form_nsex=[]
   formulas_nsex_bg=list()
   del tweets_bg[:]

   for i in np.arange(0,len(dat2)):
       y=1 # prob of sex tweets
       z=1 # prob of nonsexist tweets
       tweet_bg = dat2['tweet'][i]
       judgment_bg=dat2['judgment'][i]
       for ltr in range(len(tweet_bg)-1):
         a=tweet_bg[ltr]
         b=tweet_bg[ltr+1]
         if (a in bigrams_sex) and (b in bigrams_sex[a]):
             bigrams_sex_freqs[a]={}
             vals=list(bigrams_sex[a].values())
             vals=vals[:-1]
             total_bigrams_a=sum(vals)
             if b in sounds_sex:
                  bigrams_sex_freqs[a][b]=(bigrams_sex[a][b]+(beta*sounds_sex_freqs[b]))/((total_bigrams_a) + (beta*total_types_bigrams_sex))
             else:
                 bigrams_sex_freqs[a][b]=(bigrams_sex[a][b]+(beta*sounds_sex_freqs['unk']\
))/((total_bigrams_a) + (beta*total_types_bigrams_sex))   
             y=y+math.log(bigrams_sex_freqs[a][b])
         elif a not in bigrams_sex:
             bigrams_sex[a]={}
             bigrams_sex[a][b]=1
             bigrams_sex['unk']={}
             bigrams_sex['unk']['unk']=[]
             bigrams_sex['unk']['unk']=1
             vals=list(bigrams_sex[a].values())
             vals=vals[:-1]
             total_bigrams_a=sum(vals)
             bigrams_sex_freqs[a]={}
             bigrams_sex_freqs[a][b]=(bigrams_sex['unk']['unk']+(beta*sounds_sex_freqs['unk']))/((total_bigrams_a) + (beta*total_types_bigrams_sex))
             y=y+math.log(bigrams_sex_freqs[a][b])
         elif ((b not in bigrams_sex[a]) and (a in bigrams_sex)):
             bigrams_sex_freqs[a]={}
             bigrams_sex_freqs[a]['unk']={}
             bigrams_sex[a]['unk']=1
             vals=list(bigrams_sex[a].values())
             vals=vals[:-1]
             total_bigrams_a=sum(vals)
             if b in sounds_sex:
                bigrams_sex_freqs[a][b]=(bigrams_sex[a]['unk']+(beta*sounds_sex_freqs[b]))/((total_bigrams_a) + (beta*total_types_bigrams_sex))
             else:
                bigrams_sex_freqs[a][b]=(bigrams_sex[a]['unk']+(beta*sounds_sex_freqs['unk']))/((total_bigrams_a)+(beta*total_types_bigrams_sex))
             y=y+math.log(bigrams_sex_freqs[a][b])
         
         if a in bigrams_nsex and b in bigrams_nsex[a]:
               bigrams_nsex_freqs[a]={}
               form_nsex="formula #1"
               vals=list(bigrams_nsex[a].values())
               vals=vals[:-1]
               total_bigrams_a2=sum(vals)
               bigrams_nsex_freqs[a][b]=(bigrams_nsex[a][b]+(beta*sounds_nsex_freqs[b]))/((total_bigrams_a2)+(beta*total_types_bigrams_nsex))
               z=z+math.log(bigrams_nsex_freqs[a][b])
         elif a not in bigrams_nsex:
               bigrams_nsex[a]={}
               bigrams_nsex['unk']={}
               bigrams_nsex['unk']['unk']=1
               vals=list(bigrams_sex[a].values())
               vals=vals[:-1]
               total_bigrams_a2=sum(vals)
               bigrams_nsex_freqs[a]={}
               form_nsex="formula #2"
               bigrams_nsex_freqs[a][b]=(bigrams_nsex['unk']['unk']+(beta*sounds_nsex_freqs['unk']))/((total_bigrams_a2)+(beta*total_types_bigrams_nsex))
               z=z+math.log(bigrams_nsex_freqs[a][b])
         elif ((b not in bigrams_nsex[a]) and (a in bigrams_nsex)):
               bigrams_nsex_freqs[a]={}
               bigrams_nsex_freqs[a]['unk']={}
               bigrams_nsex[a]['unk']=1
               vals=list(bigrams_nsex[a].values())
               vals=vals[:-1]
               total_bigrams_a=sum(vals)
               if b in sounds_nsex:
                  form_nsex="formula #3"
                  bigrams_nsex_freqs[a][b]=(bigrams_nsex[a]['unk']+(beta*sounds_nsex_freqs[b]))/((total_bigrams_a2) + (beta*total_types_bigrams_nsex))
               else:
                  form_nsex="formula #4"
                  bigrams_nsex_freqs[a][b]=(bigrams_nsex[a]['unk']+(beta*sounds_nsex_freqs['unk']))/((total_bigrams_a2)+(beta*total_types_bigrams_nsex))
               z=z+math.log(bigrams_nsex_freqs[a][b])
       comprob_sex_test_bg.append(y)
       comprob_nsex_test_bg.append(z)
       tweets_bg.append(tweet_bg)
       judgments_bg.append(judgment_bg)
       formulas_nsex_bg.append(form_nsex)

   # Calculate the dot product of all of the log probs to get a single value.                                                                                                
   print('The sum of all log probs, which is the dot product of them is ', np.sum(comprob_sex_test_bg))

   # Put the compound probabilities in the dataframe.
   prob_test_bg['logprobsex']=comprob_sex_test_bg
   prob_test_bg['logprobnsex']=comprob_nsex_test_bg
   prob_test_bg['form_nsex']=formulas_nsex_bg

   # Calculate the prior log probabilities of a tweet being sexist / not sexist in the data.
   prior_sexist_bg=math.log(len(dat2[dat2['judgment']=='sexism'])/len(dat2))
   prior_nonsexist_bg=math.log(len(dat2[dat2['judgment']!='sexism'])/len(dat2))

   #Add two log probs (equivalent of multiplication in regular prob space).                                                
   prob_test_bg['logprobsex']=prob_test_bg['logprobsex'].apply(lambda x: x  +  prior_sexist_bg)
   prob_test_bg['logprobnsex']=prob_test_bg['logprobnsex'].apply(lambda x: x + prior_nonsexist_bg)
 
   prob_test_bg['tweet']=tweets_bg
   prob_test_bg['judgment']=judgments_bg

   prob_test_bg['predicted']=np.where(prob_test_bg['logprobsex']>prob_test_bg['logprobnsex'],'sexism','neither')

   #print(prob_test_bg[['tweet','judgment','predicted']])
   print(prob_test_bg['tweet'][23],prob_test_bg['judgment'][23],prob_test_bg['predicted'][23])
   print(prob_test_bg['tweet'][21],prob_test_bg['judgment'][21],prob_test_bg['predicted'][21])
   print(prob_test_bg['tweet'][6],prob_test_bg['judgment'][6],prob_test_bg['predicted'][6])

   # This gets the # of each type. 
   correct_preds_sex_bg = len(prob_test_bg[((prob_test_bg['judgment']=='sexism')& (prob_test_bg['predicted']=='sexism'))])
   all_pred_sexist_bg=len(prob_test_bg[prob_test_bg['predicted']=='sexism'])
   all_judg_sexist_bg=len(prob_test_bg[prob_test_bg['judgment']=='sexism'])

   precision_bg_beta1=correct_preds_sex_bg/all_pred_sexist_bg
   recall_bg_beta1=all_pred_sexist_bg/all_judg_sexist_bg
   fscore_bg_beta1=(2*precision_bg_beta1*recall_bg_beta1)/(precision_bg_beta1+recall_bg_beta1)

   return(precision_bg_beta1,recall_bg_beta1,fscore_bg_beta1)

# This script creates a pair of character based unigram, bigram and trigram models that are trained
# on sexist tweets, and non-sexist tweets using a set of training data that are labeled based
# on crowd-sourced labeling of sexist vs. non-sexist tweets.

# It first calculates the log probs of the test data with alpha=1, then it uses the tuning data to choose the optimal 
# smoothing value for alpha), and then recalculates the accuracy of the  test set.

# First build a model for the sexist data and calculate the probabilitie (relative frequenciess) for each 
# of the unigrams, bigrams and trigrams.

# Read the training data, tuning data, and test data into separate andas dataframes..
trndat=pd.read_table("amateur-aggregated-train.txt",names=("judgment","tweet"))
tunedat=pd.read_table("amateur-aggregated-tune.txt",names=("judgment","tweet"))
testdat=pd.read_table("amateur-aggregated-test.txt",names=("judgment","tweet"))

# Separate out the sexist and nonsexist training data into separate dataframes.
sexistdata = trndat[trndat['judgment']=="sexism"]
nonsexistdata=trndat[trndat['judgment']!="sexism"]

# Initialize dictionaries for unigrams, bigrams and trigrams for the sexist and non-sexist language models.
sounds_sex={}
sounds_nsex={}

bigrams_sex={}
bigrams_nsex={}

# Initialize dictionaries for the relative frequencies of the different sounds.
sounds_sex_freqs={}
sounds_nsex_freqs={}
bigrams_sex_freqs={}
bigrams_nsex_freqs={}

# Set the initial alpha value for smoothing to 1.
alpha=1

# Create data frames that will hold the log probabilities for each sexist and non-sexist tweets.
prob_sex=pd.DataFrame(columns=['tweet','logprob'])
prob_nsex=pd.DataFrame(columns=['tweet','logprob'])

# Create the sets that will hold the compound probabilities for each tweet, and all of the tweets.
comprob=list()
tweets=list()

# Count the raw frequencies of each character found in all the sexist tweets in 
# the train data for the unigram model.
for tweet in sexistdata['tweet']:
    for letter in tweet:
        if letter in sounds_sex:
           sounds_sex[letter]+=1
        else:
           sounds_sex[letter]=1

# Count the raw frequencies of all of the characters in the nonsexist data.
# Initialize the count of unknown characters for the sexist and unsexist data to 0.
sounds_nsex['unk']=0
sounds_sex['unk']=0

# Now do the same thing for the non-sexist data.
for tweet in nonsexistdata['tweet']:
   prob_nsex['tweet']=tweet

   for letter in tweet:
       if letter in sounds_nsex:
           sounds_nsex[letter]+=1
       else:
           sounds_nsex[letter]=1

# Add all of the characters that are in the nonsexist data, but are not in the sexist data to an 'unk' key.
for tweet in nonsexistdata['tweet']:
   for letter in tweet:
       if letter not in sounds_sex and sounds_sex['unk']==0:
           sounds_sex['unk']=1
       elif letter not in sounds_sex and sounds_sex['unk']>0:
           sounds_sex['unk']+=1

# Add all of the characters that are in the sexist data that are not in the nonsexist data
# to an 'unknown' character in the nonsexist data frame (model).
for tweet in sexistdata['tweet']:
    for letter in tweet:
        if letter not in sounds_nsex and sounds_nsex['unk']==0:
            sounds_nsex['unk']=1
        elif letter not in sounds_nsex and sounds_nsex['unk']>0:
            sounds_nsex['unk']+=1
           
comprob_ns=list()
tweets_ns=list()
unk=0

total_no_sounds_sex=0
total_no_sounds_nsex=0

# Calculate the total number of characters (types * tokens of each type) in the sexist and nonsexist tweet sets, so 
# that I can use this  as the denominator to calculate the relative frequencies of each sound.
for letter in sounds_sex:
    total_no_sounds_sex+=sounds_sex[letter]

for letter in sounds_nsex:
     total_no_sounds_nsex+= sounds_nsex[letter]

#sounds_sex_freqs['unk']=sounds_sex['unk']+alpha/(total_no_sounds_sex + (alpha*len(sounds_sex_freqs)))
#sounds_nsex_freqs['unk']=(sounds_nsex['unk']+alpha)/(total_no_sounds_nsex+(alpha*len(sounds_nsex_freqs)))

precision_prior,recall_prior,fscore_prior=compute_lh_fscore_unigram(testdat,alpha)

print ('The precision prior to tuning is',precision_prior)
print('The recall prior to tuning is',recall_prior)
print('The fscore prior to tuning is',fscore_prior)

best_alpha=0.005

# Choose the optimal alpha value to maximimze the likelihood of the tuning data.
precision_tune,recall_tune,fscore_tune=compute_lh_fscore_unigram(tunedat,best_alpha)

print('The precision of the tuning data is ',precision_tune)
print ('The recall of the training data is ',recall_tune)
print('The fscore of the training data is ',fscore_tune)

precision_post,recall_post,fscore_post =compute_lh_fscore_unigram(testdat,best_alpha)

print('Precision of test data after turning is ',precision_post)
print('Rrecall of tesst data after tuning is ,',recall_post)
print('fscore after tuning is ', fscore_post)

# Now calculate bigram raw frequencies in the sexist training data (trndat).

bigrams_sex['unk']={}
bigrams_nsex['unk']={}

# This calculates the raw frequencies for each bigram found in the sexist data.
for tweet in sexistdata['tweet']:
   for ltr in range(len(tweet)-1):
       a=tweet[ltr]
       b=tweet[ltr+1]
       if a in bigrams_sex:
         if b in bigrams_sex[a]:
             bigrams_sex[a][b]+=1
         else:
            bigrams_sex[a][b]={}
            bigrams_sex[a][b]=1
       else:
         bigrams_sex[a]={}
         bigrams_sex[a][b]=[]
         bigrams_sex[a][b]=1

# This next part calculates the raw frequencies of each bigram found in the nonsexist data. 
for tweet in nonsexistdata['tweet']:
  for char in range(len(tweet)-1):
     c=tweet[char]
     d=tweet[char+1]
     if c in bigrams_nsex:
        if d in bigrams_nsex[c]:
           bigrams_nsex[c][d]+=1
        else:
           bigrams_nsex[c][d]={}
           bigrams_nsex[c][d]=1
     else:
        bigrams_nsex[c]={}
        bigrams_nsex[c][d]=[]
        bigrams_nsex[c][d]=1

# Add 'unk' values for bigrams in nonsexistdata that are not in sexistdata, and vice versa.
# Add all of the characters that are in the nonsexist data, but are not in the sexist data to an 'unk' key.
for tweet in nonsexistdata['tweet']:
   for char in range(len(tweet)-1):
       f=tweet[char]
       g=tweet[char+1]
       if f not in bigrams_sex:
          bigrams_sex[f]={}
       bigrams_sex[f]['unk']=0
       #if g not in bigrams_sex[f] and bigrams_sex[f]['unk']==0:
       if g not in bigrams_sex[f] and 'unk' not in bigrams_sex[f]: 
          bigrams_sex[f]['unk']=1
       elif g not in bigrams_sex[f] and bigrams_sex[f]['unk']>0:
           bigrams_sex['unk']+=1
       # Need to add an 'unk' to the second layer of the dictionary'
       if f not in bigrams_sex:
           bigrams_sex['unk']=1

for tweet in sexistdata['tweet']:
   for char in range(len(tweet)-1):
      j=tweet[char]
      k=tweet[char+1]
      if j not in bigrams_nsex:
         bigrams_nsex[j]={}
      bigrams_nsex[j]['unk']=0
      if k not in bigrams_nsex[j] and 'unk' not in bigrams_sex[j]:
         bigrams_sex[j]['unk']=1
      elif k not in bigrams_nsex[j] and bigrams_nsex[j]['unk']>0:
        bigrams_nsex['unk']+=1
      if j not in bigrams_nsex[j]:
         bigrams_nsex['unk']=1                                                                        

#start here
#for tweet in nonsexistdata['tweet']:
#   for char in range(len(tweet)-1):
#      j=tweet[char]
#      k=tweet[char+1]
#     
#      if j not in bigrams_sex:
#          bigrams_sex[j]={}
#          if k not in bigrams_sex[j]:
#             bigrams_sex[j][k]=bigrams_sex['unk']
#      else:
#          if k not in bigrams_sex[j]:
#              bigrams_sex[j]['unk']=bigrams_sex['unk']
             

total_tokens_bigrams_sex=0
total_tokens_bigrams_nsex=0

total_types_bigrams_sex=0
total_types_bigrams_nsex=0

total_tokens_bigrams_sex=(sum(v) for v in bigrams_sex.values())
#print(total_tokens_bigrams_sex,'is total sex  tokens')
total_tokens_bigrams_nsex=(sum(v) for v in bigrams_nsex.values())
#print(total_tokens_bigrams_nsex,'is total nsex tokens.')
total_types_bigrams_sex=sum(len(v) for v in bigrams_sex.values())
#total_types_bigrams_nsex=sum(len(p) for p in bigrams_nsex.values())

for e in bigrams_nsex:
     if type(bigrams_nsex[e]) is dict:
          total_types_bigrams_nsex += len(bigrams_nsex[e].values())
     else:
          total_types_bigrams_nsex += 1

print('total types bigrams sex are ',total_types_bigrams_sex)
print('total types bigrams nsex are ',total_types_bigrams_nsex)

# Step 4 of assignment - try classification using bigram model with dirichlet smoothing using beta=1

beta=1
precision_bg_beta1,recall_bg_beta1,fscore_bg_beta1=compute_lh_fscore_bigram(testdat,beta)

print('precision of bigram model prior to tuning is',precision_bg_beta1)
print('recall of bigram model prior to tuning is', recall_bg_beta1)
print('fscore of bigram model prior to tuning is', fscore_bg_beta1)
 
best_beta=0.001

# Choose the optimal beta value to maximimze the likelihood of the tuning data.                                                                                             
precision_bg_tune,recall_bg_tune,fscore_bg_tune=compute_lh_fscore_bigram(tunedat,best_beta)

print('precision of bigram model tuning data is ',precision_bg_tune)
print('recall of bigram model tuning data is ',recall_bg_tune)
print('fscore of bigram model tuning data is ',fscore_bg_tune)

# rerun the test data with the optimal beta value.
precision_bg_bestbeta,recall_bg_bestbeta,fscore_bg_bestbeta=compute_lh_fscore_bigram(testdat,best_beta)

print('precision of bigram model afer tuning is ',precision_bg_tune)
print('recall of bigram model after tuning is ',recall_bg_tune)
print('fscore of bigram model after tuning is ',fscore_bg_tune)







