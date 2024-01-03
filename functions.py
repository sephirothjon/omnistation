    
import streamlit as st
import pandas as pd
import numpy as np
import time, os


def valid_range(x,min,max):
    assert(min<max), "valid_range, min > max"
    assert(x>=min), "valid_range, x < min"
    assert(x<=max), "valid_range, x > max" 


def tictoc(func):
    def wrapper():
        t1=time.time()
        func()
        t2=time.time()-t1
        print(f'(func.__name__) ran in {t2} secs)')
    return wrapper

@tictoc
def do_this(): time.sleep(1.3)
#do_this()


AWS_BUCKET_URL = "http://streamlit-demo-data.s3-us-west-2.amazonaws.com"
AWS_BUCKET_FILE = "agri.csv.gz"

# My load data from web func
@st.cache_data
def load_data_from_web(_url,_file,_nrows=0):
    valid_range(_nrows,0,100000000)
    if _nrows==0: df = pd.read_csv(_url + "/" + _file)
    else: df = pd.read_csv(_url + "/" + _file, nrows=_nrows)
    lowercase = lambda x: str(x).lower() # all vars lowercase
    df.rename(lowercase, axis='columns', inplace=True)
    return df.reset_index()


@st.cache_data
def load_data_from_web2(_url,_nrows):
    df = pd.read_csv(_url, nrows=_nrows)
    lowercase = lambda x: str(x).lower() # all vars lowercase
    df.rename(lowercase, axis='columns', inplace=True)
    return df.reset_index()


# filename = bike_rental_stats.json, bart_stop_stats.json, bart_path_stats.json
@st.cache_data
def load_jsondata_from_web(_file):
    url = ("http://raw.githubusercontent.com/streamlit/"
        "example-data/master/hello/v1/%s" % _file)
    df = pd.read_json(url)
    lowercase = lambda x: str(x).lower() # all vars lowercase
    df.rename(lowercase, axis='columns', inplace=True)
    return df.reset_index()


#csvlist=[]
#for f in os.listdir("./csv"):
#    if os.path.isdir("./csv/"+f):
#        csvlist.append(f)
#print(type(os.listdir("./csv")))

# str(typ).lower()=="csv"

# st.set_page_config(page_title="Plotting Demo", page_icon="📈")
# st.set_page_config(page_title="DataFrame Demo", page_icon="📊")


# df = pd.read_excel('sales_data.xlsx', sheet_name='2021')


# all_sheets = pd.read_excel('sales_data.xlsx', sheet_name=None)
# type(all_sheets)
# for key, value in all_sheets.items():
#   print(key, type(value))
#   print(key)
#   display(value)


#selected_layers = [ 
#  layer
#  for layer_name, layer in ALL_LAYERS.items()
#  if st.sidebar.checkbox(layer_name, True)]


# df[DATE_COL] = pd.to_datetime(df[DATE_COL])
# df.X
# help(np.zeros)
# type(df2), len(df2), df2.shape , df2.ndim , df2.size, df2.columns
# df2.T, df2[:2], df2[1:7], df2['Price'], df2[df2['Price']>50000]
# df2['Price'].min(),max,mean,std,var,sum,cumsum,square,sqrt,abs,cumprod,median
# df2['Price'].count() # same as len(df2)
# df2['Price'].corr(df2['Quantity'])
# df2.describe() # for each numeric var, gives count,mean,std,min,25,50,75,max
# df2['Price'].diff()
# df2['Price'].drop_duplicates()
# df2=(df2.drop_duplicates(['varname']))['varname']
# df2['Price'].dropna()
# df2['Price'].fillna(999999), df2['Price'].fillna('')
# df2['Price'].head()
# df2['Price'].hist() # histogram
# df2['Price'].plot() # series plot
# df2['Price'].isna()
# df2['Price'].isnull()
# df2['Price'].rank()
# df2.loc[:, 'field1'] = value
# df.loc[df['field1'] < 0] = 0
# df.loc[df['field']=="PASS"]
# p1nv['sum']=p1nv.apply(lambda r: r['field1']+r['field2']+r['field3'],axis=1)
# df['value'] = df['value'].replace(nan, 'Null', regex=True)
# stdflg=len(((p1nv.loc[p1nv['stdflg']!=0]).drop_duplicates(['name']))['name'])
# pchisq_flg=len(((p3.loc[p3['probchisq']>0.05]).drop_duplicates(['varname']))['varname'])
# dvt=dvt.rename(columns={dvt.columns[0]:"t",dvt.columns[1]:"N_"})
# dfn=dfn.rename(columns={"bin":"woe"})
# dfc=dfc.rename(columns={"bin":"woe","cat":"range"})
# pd.concat([dfn,dfc])
# clustdf=self.p3f[['cluster','varname','probt','rsq','adjrsq']]
# clustdf=clustdf.round({'probt': 6})
# clustdf.sort_values(['cluster', 'adjrsq'],ascending=[False,True],inplace=True)
# orcldf.sort_values(by=['gini'],ascending=True,inplace=True)



#confusion_matrix(y_train_5, y_train_pred)
#precision_score(y_train_5, y_train_pred) # PS = TP / (TP + FP)
#recall_score(y_train_5, y_train_pred) # RS = TP / (TP + FN)
#f1_score(y_train_5, y_train_pred) # F1 = (2 * PS * RS) / (PS + RS)
#y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
#precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
#    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
#    plt.xlabel("Threshold"); plt.legend(loc="upper left"); 
#    plt.xlim([-60000, 60000]); plt.ylim([0, 1.01])
#plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#plt.show()
# Using chart, select a threshold to give desired precision/recall, 90% for example
#y_train_pred_90 = (y_scores > 10000)
#precision_score(y_train_5, y_train_pred_90) , recall_score(y_train_5, y_train_pred_90)



#    def logit_pvalue(model, x):
#        p = model.predict_proba(x)
#        n = len(p)
#        m = len(model.coef_[0]) + 1
#        coefs = np.concatenate([model.intercept_, model.coef_[0]])
#        x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
#        ans = np.zeros((m, m))
#        for i in range(n):
#            ans = ans + np.dot(np.transpose(x_full[i, :]), 
#                               x_full[i, :]) * p[i,1] * p[i, 0]
#        vcov = np.linalg.inv(np.matrix(ans))
#        se = np.sqrt(np.diag(vcov))
#        t =  coefs/se  
#        p = (1 - norm.cdf(abs(t))) * 2
#        wald = t**2
#        return pd.DataFrame([coefs, p, wald]).T

