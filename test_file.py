#aspect based sentiment analysis
import pandas as pd
from textblob import TextBlob
from textblob.taggers import PatternTagger
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = list(stopwords.words('english')) #About 150 stopwords

#1.Read Data
df=pd.read_csv("/home/pearlsoft/Downloads/Comments12.csv",nrows=150, encoding='cp1252')
df =df.head(100)
# df=df.tail(1)
# # print(df)
#2 function for finding aspects
def data_tags(x):
    text_tokens = word_tokenize(x)
    stop_word = stopwords.words('english')
    not_stopwords = {'very','not','too', 'over', 'is', 'was'} 
    final_stop_words = set([word for word in stop_word if word not in not_stopwords])
    # print(final_stop_words)
    tokens_without_sw = [word for word in text_tokens if not word in final_stop_words]
    filtered_sentence = (" ").join(tokens_without_sw)
    print(filtered_sentence)
    lst=[]
    tag_data = TextBlob(filtered_sentence).tags
    print(tag_data)
    for i in range(0,len(tag_data)):
        if tag_data[i][1] in ['NN', 'NNP','NNS', 'VB', 'VBG', 'VBP', 'VBN', 'JJ']:
        # if tag_dat/a[i][1]=='NNP' or tag_data[i][1]=='NNS' or tag_data[i][1]=='NN' or tag_data[i][1]=='JJ' or tag_data[i][1]=='VB':
            if tag_data[i-1][1] in ['RB']:
                res =  tag_data[i-1][0] + ' ' + tag_data[i][0]
                lst.append(res)
            elif tag_data[i-1][1] in ['JJ'] and tag_data[i][1] == 'JJ':
                res =  tag_data[i-1][0] + ' ' + tag_data[i][0]
                lst.append(res)
            else:
                lst.append(tag_data[i][0])
    return lst    

#3.Given aspects
# print("...............................hello.....................")
df['aspects']=df['Comments'].apply(data_tags)
aspect_list=df['aspects'].to_list()    
lst_cmmt=df['Comments']

#4.function for polarity as a result
def pol_iter(x):
    lst_pol=[]
    for word in x:
    #   for word in line:
        tag=TextBlob(word).tags
        pol=TextBlob(word).polarity
        print(word,pol)
        lst_pol.append(pol)
    return lst_pol  
df['new_pol']=df['aspects'].apply(pol_iter)
#5.function for extracted aspects
def extract_aspect(x):
    lst_aspect=[]
    for aspect in x:
        if TextBlob(aspect).polarity>0:
            lst_aspect.append(aspect)    
        elif TextBlob(aspect).polarity<0:
            lst_aspect.append(aspect)   
        else:
            pass   
    return lst_aspect

df['new_aspectss']=df['aspects'].apply(extract_aspect)    

#6.function for extract sentiment
def extract_sentiment(x):
    lst_sentiment=[]
    for aspect in x:
        aspect_sentimence = {aspect:''}
        if TextBlob(aspect).polarity>0:
            aspect_sentimence[aspect] = "Positive"
            lst_sentiment.append(aspect_sentimence)   
        elif TextBlob(aspect).polarity<0:
            aspect_sentimence[aspect] = "Negative"
            lst_sentiment.append(aspect_sentimence)    
        else:
            pass        
    return lst_sentiment

df['new_sentimentss']=df['aspects'].apply(extract_sentiment) 
df1=df.drop(['aspects', 'new_pol','Compassion', 'Competence', 'Unnamed: 3'],axis=1, errors='ignore')   
print(df['new_sentimentss'])
compassion=['helpless','helpful','help','benevolence','goodwill','mercy','Love','wellbeing','well-being','empathy','humanity','humanities','mankind','kind','kindful','kindfulness','love','care','caring','compassion','compassionate','charity',' condolences','comfort','comfortable','misery','condolence', 'regret','humaneness', 'humanity','kindheartedness','kindliness','kindness','mercy','pity','ruth','softheartedness','warmheartedness','affinity','empathy','rapport','sensitivity','understanding','altruism','benevolence','benignity','bigheartedness', 'charity','generosity','goodwill','humanitarianism','largeheartedness','largesse','magnanimity','philanthropy','bigheartedness','charity', 'commiseration','feeling','good-heartedness','heart','largeheartedness','mercy','pity','ruth','softheartedness','sympathy','feelings', 'responsiveness','sensibility','sensitivity','affection','love','regard','affinity','empathy','rapport','altruism','benevolence','benignancy', 'benignity','generosity','humaneness','humanism','humanitarianism','philanthropy','heart cheese cloud','commiserate','commiseration','compassion','compassionate','compassionately','solatory','crying','hard luck','tough cheese','shame','listener','loss luck','heart bleeds', 'pathetic','pathetically','pathospiteous','piteously','ache','adroitness','affection','affectionateness','affinity','agility','almsgiving','altruism','amiability','analysis','appetite','attention','awareness','beneficence','benevolence','benignity','big-heartedness','bounteousness','bounty','broken-heartedness','burning','cacoethes','care','caring','charitableness','charity','charm','cheerlessness','chemistry','clemency','closeness','cogitation','comfort','commiseration','commiserations','compassion','compunction','concern','condolence','condolences','considerateness','consideration','consolation','contemplation','contriteness','contrition','courteousness','craving','decency','deftness','dejection','deliberation','depression','desire','desolation','despair','despondency','dexterity','disconsolateness','disconsolation','discussion','dismalness','dolefulness','dolorous','doubts','down','downheartedness','eagerness','ease','effortlessness','elegance','empathy','encouragement','examination','fancy','feeling','feelings of guilt','fellow feeling','fellow feeling for','fellowship with','fervour','finesse','flow','flowingness','fluency','fluidity of movement','forbearance','forgiveness','free-handedness','friendliness','generosity','gentleness','gloom','gloominess','glumness','goodness','goodwill','grace','gracefulness','graciousness','greed','grief','guilt','guilty conscience','hankering','heart','heartache','heaviness of heart','heed','helpfulness','hesitancy','hesitation','hospitality','humaneness','humanitarianism','humanity','hunger','hungering','identification with','inclination','indulgence','inspection','itch','kind-heartedness','kindliness','kindness','largesse','lavishness','lenience','leniency','lenity','liberality','light-footedness','lightsomeness','like-mindedness','longing','low spirits','lust','magnanimity','magnanimousness','meditation','melancholy','mercifulness','mercy','mildness','misery','misgivings','mournfulness','mulling','munificence','musing','naturalness','neatness','need','neighbourliness','nimbleness','notice','open-handedness','pangs of conscience','twinges of conscience','patience','penitence','philanthropism','pining','pity','poetry in motion','poise','pondering','precision','public-spiritedness','qualms','quarter','rapport','reflection','regard','regret','reluctance','remorse','repentance','reservations','review','rumination','ruthlessness','sadness','scruples','scrutiny','selflessness','self-reproach','sensibility','sensitivity towards','smoothness','social conscience','softheartedness','soft-heartedness','softness','solace','solicitousness','solicitude','sorrow','strictness','stylishness','suppleness','support','sympathy','tenderheartedness','tender-heartedness','tenderness','thirst','thought','thoughtfulness','togetherness','tolerance','understanding','unease','uneasiness','unhappiness','unselfishness','urge','want','warm-heartedness','warmth','wish','woe','worries','wretchedness','yearning','yen']
communication=['honest','contact','talk','talking','telling','tell','message','information','inform','friendly','conversation','chat','comment','judgment','judgmented','judgmental','knowledge','account','accustomed','acquaint','advertise','advice','advise','advisement','announce','announcement','announcing','answer','ardent','articulate','articulation','artistic','assertion','associate with','babble','be an indication of','be close to','be in touch','be near','betray','blather','blether','briefing','broadcast','broadcasting','bulletin','bumf','buzz','chat','chatter','chew the fat','chew the rag','chunter','circulating','circulation','cogent','coherent','commune with','communicate','communicative','communion','comprehensible','confabulate','confer','connect','connection','consult each other','contact','conversation','converse','convey','conveyance','conveying','correspond','correspondence','corresponding','counsel','counselling','declaration','declare','deeply felt','deets','delivery','demonstrative','descriptive','details','direction','directions','directive','disclose','disclosing','disclosure','discourse','discover','discuss things','dispatch','disseminate','dissemination','divulgation','divulge','divulgence','drop a line','drop a note','effective','eloquent','elucidation','emotional','enlighten','enlightenment','enunciate','establish contact','evocative','excerpt','express','expression','expressive','facts','familiar','figures','fluent','friendly','full of emotion','full of feeling','gab','gen','get across','get on the horn','get through','give a call','give a ring','give notice of','give voice','go on','gossip','guidance','guidelines','handing on','have a chat','have a confab','have a talk','have confidence of','have negotiations','hear from','herald','hint','hot story','ideas','illuminating','imaginative','impart','imparting','imply','indicate','indicating emotion','indicating feeling','info','inform','information','inside story','inspired','instruction','instructions','intelligence','intelligible','intense','interact','interchange','intercommunication','intercourse','interface','intimate','keep in touch','knowledge','known','known to','language','let on','let out','link','lowdown','lucid','make known','make public','making known','material','meaningful','mention','message','missive','moving','natter','negotiate','network','news','note','notice','notifying','on friendly terms','on good terms','orate','orientation','palaver','parley','particulars','pass on','passing on','passionate','persuasive','phoned','pipe up','pipeline','poignant','powerful','prate','prattle','precis','preparation','presenting','proclaim','promulgation','pronounce','prophecy','publication','publicity','publicize','publish','raise','rapped out','rattle on','reach','reach out','reading','reception','recognized','relate','relay','reply','report','reporting','reveal','revealing','revelation','ring up','run off at the mouth','rundown','say','scoop','shoot','signalled','signify','silver-tongued','skinny','sound off','speak','speak to each other','speak up','speech','spout','spread','spreading','state','statement','statistics','stirring','striking','suggest','suggestive','summary','talk','talk up','talking','teach','teaching','tell','telling','the dope','the inside story','the latest','the low-down','the poop','tidings','touch base','transfer','translating','translation','transmission','transmit','understandable','unfold','utter','utterance','verbalize','very close','visionary','vivid','vocal','vocalize','voice out','voice over','warning','well known','witter','word','work','write','writing']
competence=['super','disinterest','disinterested','wonderful','crowd','complain','complaining','comfort','discomfort','super','efficient','knowledgeable','risk','struggle','struggled','inconvenience','enjoy','understaffing','outstanding','good','bad','amazing','fabulous','great','exceptional','nice','thanks','best','waiting','wait', 'ability','able','accomplishment','acuity','acumen','acute','acuteness','adeptness','adequacy','adroitness','agility','alertness','appropriateness','aptitude','aptness','argute','arguteness','artfulness','artistry','astute','astuteness','basics','beginnings','bent','brightness','brilliance','calculatedness','calculation','calibre','canniness','canny','capability','capable','capacity','characteristics','clever','cleverness','clumsiness','common sense','commonsense','competence','competency','craft','craftiness','cunning','cutting it','cutting the mustard','deftness','delicacy','dexterity','dexterousness','diplomacy','discernment','discretion','discrimination','effectiveness','efficacy','efficiency','effortlessness','equal','essence','essentials','experience','expert','expertise','expertness','facilities','facility','faculties','faculty','felicity','finesse','fit','fitness','flair','forte','genius','gift','giftedness','good','gullible','hacking it','handiness','heads-up','horse sense','imagination','imaginativeness','inability','inadequacy','incapacity','ingenuity','ingredients','insight','intelligence','intelligent','inventiveness','knack','know-how','knowledge','long-headed','making the grade','makings','mastery','materials','means','media-savvy','merit','might','moxie','niftiness','nimbleness','nippiness','nous','on the ball','pawky','penetration','perception','perceptive','perceptiveness','perspicacious','perspicaciousness','perspicacity','potential','potentiality','power','productiveness','proficiency','promise','propensity','prowess','qualification','qualified','qualifiedness','qualities','quick-wittedness','resourcefulness','resources','rudiments','sagacious','sagacity','sage','sageness','sapience','sapient','savoir faire','savvy','sensitivity','sharp','sharpness','sharp-witted','sharp-wittedness','shrewd','shrewdness','skilfulness','skill','sleight of hand','slickness','smart','smartness','streetwise','strong point','stuff','stupid','stupidity','suitability','suitable','talent','the goods','the right stuff','value','virtuosity','what it takes','where withal','whip-smart','wiliness','wisdom','wit','wizardry','worth']

"""
Method 1 clustering:- Individual clustering
"""

def check_category(category, data):
    l1_compt=[]
    for data in data:
        print(data)
        for aspect, sentiment in data.items():
            if set(category).intersection(aspect.split()):
                test_str ="yes"
                l1_compt += [test_str]  
                if set(category).intersection(aspect.split()):  
                    aspect_value = set(category).intersection(aspect.split())
                    val="("+str(list(aspect_value)[0])+":"+sentiment+")"
                    l1_compt.append(val)  
    return l1_compt

df1['Competence']=df1['new_sentimentss'].apply(lambda x: check_category(competence,x))           
df1['Compassion']=df1['new_sentimentss'].apply(lambda x: check_category(compassion,x))           
df1['Communication']=df1['new_sentimentss'].apply(lambda x: check_category(communication,x))           


df2=df1.drop(['new_aspectss', 'new_sentimentss'],axis=1)
df2.to_csv("Results.csv")  