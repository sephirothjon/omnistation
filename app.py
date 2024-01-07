# cd C:\temp\vscode\temp_proj
# python -m venv .venv
# .venv\Scripts\activate.bat

# pip install streamlit
# python -m streamlit hello
# python -m streamlit run app.py OR
# python -m streamlit run ./app.py
# Cntl+C to stop

# Put your app in a public GitHub repo (and make sure it has a requirements.txt!)
# Use pip freeze
# https://learnpython.com/blog/python-requirements-file/
# Sign into share.streamlit.io
# Click 'Deploy an app' and then paste in your GitHub URL

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, os
import math
import altair as alt
from vega_datasets import data
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.transform import factor_cmap
from functions import *
from other_app_defs import *
from varclushi import VarClusHi
from scipy.stats import norm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import xgboost as xgb
import lightgbm as lgb

##################################################################################
##################################################################################
##################################################################################
def load_csv_local(_path,_file):
    df = pd.read_csv(os.path.join(_path,_file))
    lowercase = lambda x: str(x).lower() # all vars lowercase
    df.rename(lowercase, axis='columns', inplace=True)
    #df.reset_index() -> adds an index column
    return df


@st.cache_data 
def generate_data():
    X, y = make_classification(
        n_samples=5000, n_classes=2, random_state=999,
        n_features=25, n_informative=10, n_redundant=5,
        flip_y=0.1, # high value to add more noise
        class_sep=0.9, # low value to add more noise
        weights=[0.85] # X% will be goods
    )
    DS_ = pd.DataFrame(X)
    DS_.columns=["X"+str(i) for i in range(1,len(DS_.columns)+1)]
    DS_['depvar'] = y
    return DS_


@st.cache_data     
def get_valid_dv(ds_):
    print("get_valid_dv")
    valid_dv_ = []
    for v in ds_.columns:
        if (ds_[v].dtype != "object" and ds_[v].isnull().sum()==0 and 
            ds_[v].nunique()==2 and ds_[v].min()==0 and ds_[v].max()==1):
            valid_dv_.append(v)
    return valid_dv_


def display_Sidebar():
    print("\n\n\n\n\n")
    print("********************")
    print("*****  Sidebar  ****")
    print("********************")
    print("Sidebar: ", st.session_state.AppState)
    c = st.sidebar.container(border=True)

    # Load Data or Random Generate
    def data_selector(): 
        st.session_state.AppState = "Initial"
        st.session_state.DV = None
    ds_loadorrandom = c.radio(
        "Load Data or Random Generate", ("Load","Random"), index=0, 
        key="loadorrandom", horizontal=True, on_change=data_selector) 
    if ds_loadorrandom == "Random": st.session_state.disable_dsselect=True
    else: st.session_state.disable_dsselect=False

    # Dataset Selection
    if 'disable_dsselect' not in st.session_state: 
        st.session_state.disable_dsselect = False
        st.session_state.ALT_MDLS=None
    def ds_selector(): 
        st.session_state.AppState = "Initial"
        st.session_state.DV = None
    ds_select = c.selectbox(
        'Dataset Selection', os.listdir("./csv"), on_change=ds_selector,
        index=None, placeholder="Select preloaded dataset...", key="selboxcsv",
        disabled=st.session_state.disable_dsselect)  
    
    # Sidebar State Logic
    if ds_select != None or ds_loadorrandom == "Random":
        if st.session_state.AppState == "Initial":
            setAppState("DS_Selected")
            if ds_loadorrandom == "Load":
                st.session_state.DS = load_csv_local("./csv", ds_select)
            else: 
                st.session_state.DS = generate_data()
            st.session_state.VALID_DV = get_valid_dv(st.session_state.DS)
        st.session_state.DV = c.selectbox(
            'DV Selection', st.session_state.VALID_DV,
            index=None, placeholder="Select DV...", key="selboxdv")            
            
        if st.session_state.DV != None: 
            if st.session_state.AppState == "DS_Selected":
                setAppState("DV_Selected")
                st.session_state.NUMVARS = len(st.session_state.DS.columns)
            st.session_state.DS_SPLITTER = c.radio(
                "Select Data Split", ("Random","Field"), 
                index=None, key="radiodevsplit", horizontal=True) 
            if st.session_state.DS_SPLITTER != None and st.session_state.AppState=="DV_Selected":
                setAppState("DS_SPLITTER_Selected")
                if st.session_state.DS_SPLITTER=="Random": 
                    print("ds_data_splitter")
                    DS = st.session_state.DS
                    DS['devootflag']=DS.apply(
                        lambda r: st.session_state.RNG.uniform(), axis=1)
                    st.session_state.DEV = (
                        DS[DS['devootflag'] <= .75]).drop(columns=['devootflag'])
                    st.session_state.OOT = (
                        DS[DS['devootflag'] > .75]).drop(columns=['devootflag'])
                    st.session_state.DS = DS.drop(columns=['devootflag'])
                elif st.session_state.DS_SPLITTER=="Field": 
                    st.write("TBA") 
                    setAppState("DV_Selected")
                    st.session_state.DS_SPLITTER = None
        else: setAppState("DS_Selected")
    else: 
        for key in st.session_state.keys(): del st.session_state[key]
        setAppState("Initial")

    # Cntl Parameters
    with st.sidebar.expander("Control Parameters", expanded=False):
        def change_slider(): setAppState("DS_SPLITTER_Selected")
        st.session_state.CP1 = st.slider(
            'misspctflg', 0.0, 1.0, .99, on_change=change_slider, key="cp1")
        st.session_state.CP2 = st.slider(
            'manygrpsflg', 2, 50, 15, on_change=change_slider, key="cp2")
        
    st.sidebar.caption(f"Current AppState: {st.session_state.AppState}")

    # Debug Toggle button
    st.session_state.DEBUGON = st.sidebar.toggle(
        "Debug", value=False, key="debugtoggle")
    
    # Reset button
    if 'clicked_reset' not in st.session_state: 
        st.session_state.clicked_reset = False
    def click_reset(x): st.session_state.clicked_reset = x
    st.sidebar.button(
        "Reset", type="primary", on_click=click_reset, args=[True], 
        key="btn_reset")
    if st.session_state.clicked_reset: 
        for key in st.session_state.keys(): del st.session_state[key]
        setAppState("Initial")
        st.session_state.selboxcsv = None
        st.rerun() 

    st.sidebar.image("./static/DSCF5048.JPG")
    st.sidebar.link_button("Go to gallery URL", "https://streamlit.io/gallery")


@tictoc
def setup_and_summary():
    print("setup_and_summary: ", st.session_state.AppState)
    DV = st.session_state.DV
    DS = st.session_state.DS
    DEV = st.session_state.DEV
    OOT = st.session_state.OOT
    VLIST = []; VLIST_N = []; VLIST_C = [] 
    N_DEV = len(DEV); N_OOT = len(OOT)
    NB_DEV = DEV[DV].sum(); NG_DEV = N_DEV - NB_DEV
    NB_OOT = OOT[DV].sum(); NG_OOT = N_OOT - NB_OOT
    for v in DS.columns:
        if v != DV: 
            VLIST.append(v)
            if DS[v].dtype=="object": VLIST_C.append(v)
            else: VLIST_N.append(v)
        if DS[v].dtype=="bool": # convert bool type into 1-shot
            DEV['temp_'] = DEV[v].apply(lambda x: 1 if x == True else 0)
            # OR DEV['temp_'] = DEV[v].map({TRUE: 1, FALSE: 0})
            DEV.drop(columns=[v], inplace=True)
            DEV.rename(columns={'temp_': v}, inplace=True)
            OOT['temp_'] = OOT[v].apply(lambda x: 1 if x == True else 0)
            OOT.drop(columns=[v], inplace=True)
            OOT.rename(columns={'temp_': v}, inplace=True)
    NUMVARS_N = len(VLIST_N)
    NUMVARS_C = len(VLIST_C)
    st.session_state.SUMMARY_N = DEV[VLIST_N].describe()
    if len(VLIST_C) != 0: st.session_state.SUMMARY_C = DEV[VLIST_C].describe()
    st.session_state.NUMVARS_N = NUMVARS_N
    st.session_state.NUMVARS_C = NUMVARS_C
    st.session_state.VLIST = VLIST
    st.session_state.VLIST_N = VLIST_N
    st.session_state.VLIST_C = VLIST_C
    st.session_state.N_DEV = N_DEV
    st.session_state.NG_DEV = NG_DEV
    st.session_state.NB_DEV = NB_DEV
    st.session_state.N_OOT = N_OOT
    st.session_state.NG_OOT = NG_OOT
    st.session_state.NB_OOT = NB_OOT


@tictoc
def setup_and_summary_layout():
    print("setup_and_summary_layout: ", st.session_state.AppState)
    DEV = st.session_state.DEV
    NUMVARS = st.session_state.NUMVARS
    NUMVARS_N = st.session_state.NUMVARS_N
    NUMVARS_C = st.session_state.NUMVARS_C
    N_DEV = st.session_state.N_DEV
    NB_DEV = st.session_state.NB_DEV
    NG_DEV = st.session_state.NG_DEV
    N_OOT = st.session_state.N_OOT
    NB_OOT = st.session_state.NB_OOT
    NG_OOT = st.session_state.NG_OOT
    overview_txt = f''':blue[**DEV Dataset**]  
            Total Records: {N_DEV}  
            [Total Bads: {NB_DEV}, Total Goods: {NG_DEV}]  
            Overall Badrate: {float("{:.2f}".format(NB_DEV/N_DEV))}'''
    overview_txt2 = f''':red[**OOT Dataset**]  
            Total Records: {N_OOT}  
            [Total Bads: {NB_OOT}, Total Goods: {NG_OOT}]  
            Overall Badrate: {float("{:.2f}".format(NB_OOT/N_OOT))}'''
    c = st.container(border=True)
    col1, col2 = c.columns([3,2])
    with col1:
        st.markdown(overview_txt)
        st.markdown(overview_txt2)
        st.markdown(f''':green[Total Columns] (including DV): {NUMVARS}  
                    [{NUMVARS_N} Numeric variables, 
                    {NUMVARS_C} Non-Numeric Variables]''')
    col2.write(DEV.dtypes.to_frame())
    # DS.describe(),DS.describe(include='all'),include=[np.number],exclude=[np.number]
    c.markdown('''Numeric Variables (including DV) [:blue[DEV] Only]:''')
    c.write(st.session_state.SUMMARY_N)
    c.markdown('''Non-Numeric Variables [:blue[DEV] Only]:''')
    if len(st.session_state.VLIST_C) != 0: 
        c.write(st.session_state.SUMMARY_C)
    else: 
        c.warning("There are no Non-Numeric Variables in the DEV data", icon="⚠️")
    with st.expander("10 Records from DEV", expanded=False):
        st.dataframe(DEV.head(10))


@tictoc
def Numer_Only_Compute():
    print("Numer_Only_Compute: ", st.session_state.AppState)
    CP1 = st.session_state.CP1
    DV = st.session_state.DV
    DEV = st.session_state.DEV
    VLIST_N = st.session_state.VLIST_N
    N_DEV = st.session_state.N_DEV
    SUMMARY_N = st.session_state.SUMMARY_N
    VLISTF_N = VLIST_N; RANKPLOTMAP_TBL = {}
    NLEVELS_TBL = pd.DataFrame(); VLIST_FAIL = pd.DataFrame()
    df = DEV[[v for v in DEV.columns if v==DV or v in (VLIST_N)]]
    k=0
    for v in df.columns: 
        if v != DV:
            # build NLEVELS_TBL
            NLEVELS_TBL.loc[k,'vname']=v
            nummiss = df[v].isnull().sum()
            NLEVELS_TBL.loc[k,'num_missings'] = nummiss
            NLEVELS_TBL.loc[k,'misspct'] = nummiss / N_DEV
            dfm = df[df[v].isnull()][[DV]]     
            if len(dfm) > 0: 
                NLEVELS_TBL.loc[k,'Nb_if_miss'] = dfm[DV].sum()
                NLEVELS_TBL.loc[k,'Ng_if_miss'] = nummiss - NLEVELS_TBL.loc[k,'Nb_if_miss']
                NLEVELS_TBL.loc[k,'br_if_miss'] = dfm[DV].sum()/len(dfm)
            else: NLEVELS_TBL.loc[k,'br_if_miss'] = np.nan
            NLEVELS_TBL.loc[k,'nlevels'] = df[v].nunique()
            k+=1
            # Append VLIST_FAIL cases
            if SUMMARY_N[v]['std'] == 0:
                temp = pd.DataFrame({"vname":[v], "reason":["stdflg"]})
                VLIST_FAIL = pd.concat([VLIST_FAIL, temp], ignore_index=True)  
                VLISTF_N.remove(v) 
            elif nummiss / N_DEV > CP1: 
                temp = pd.DataFrame({"vname":[v], "reason":["misspctflg"]})
                VLIST_FAIL = pd.concat([VLIST_FAIL, temp], ignore_index=True)
                VLISTF_N.remove(v)   
            # Apply the procRanker and build RANKPLOTMAP_TBL 
            dfv = df.sort_values(by=[v], ascending=True)[[v,DV]].dropna()
            dfv.reset_index(drop=True, inplace=True)
            dfv['pct_'] = (dfv.index+1)/N_DEV # df.index is _N_-1
            nlevels = NLEVELS_TBL.loc[NLEVELS_TBL['vname']==v]['nlevels'].item()
            maxgrps = min(15,int(nlevels)) 
            dfv = dfv.assign(**{f'grps_{g}' : 1 for g in range(2,maxgrps+1)})
            dfv['changeflag'] = dfv[v].diff()
            for i in range(2,maxgrps+1):
                global j_; j_=1
                def jinc(): global j_; j_+=1; return j_
                if i == nlevels:
                    dfv[f'grps_{i}'] = dfv.apply(
                        lambda x: jinc() if x['changeflag']>0 else j_, axis=1)
                else:
                    dfv[f'grps_{i}'] = dfv.apply(
                        lambda x: jinc() if x['changeflag']>0 and 
                        x['pct_'] > j_/i else j_, axis=1)
            RANKPLOTMAP_TBL[v]=dfv
    st.session_state.NLEVELS_TBL = NLEVELS_TBL
    st.session_state.VLIST_FAIL = VLIST_FAIL
    st.session_state.VLISTF_N = VLISTF_N
    st.session_state.RANKPLOTMAP_TBL = RANKPLOTMAP_TBL


@tictoc
def Numer_Only_Layout():
    print("Numer_Only_Layout: ", st.session_state.AppState)
    DV = st.session_state.DV
    NLEVELS_TBL = st.session_state.NLEVELS_TBL
    c = st.container(border=True)
    col1_, col2_ = c.columns([2,3])
    with col1_: st.write(' ')
    with col2_: st.markdown('''Numeric Variables''')
    c.divider()
    col1, col2 = c.columns([3,2])
    with col1:
        st.markdown('''Nlevels & Missing Info Table''')
        st.write(st.session_state.NLEVELS_TBL[[
            "vname","nlevels","num_missings","misspct","br_if_miss"]])
    with col2:
        st.markdown('''Information Value''')
        st.write(st.session_state.IV_N)
    c.divider()
    VarSelect_N = c.selectbox(
        'Variable Selection', st.session_state.VLISTF_N, index=None, 
        placeholder="Select Variable...", key="selboxvar_n") 
    if VarSelect_N != None:
        rp = st.session_state.RANKPLOTMAP_TBL[VarSelect_N]
        nlevels = NLEVELS_TBL.loc[NLEVELS_TBL['vname']==VarSelect_N]['nlevels'].item()
        maxgrps = min(15,int(nlevels))   
        binslide = 2
        if maxgrps > 2:     
            binslide = c.slider("Bin Select", 2, maxgrps, maxgrps, key="binslider")
        df = rp.groupby(f'grps_{binslide}', as_index=False).agg(
            N=(DV,'count'), Nb=(DV,'sum'), 
            min=(VarSelect_N,'min'), max=(VarSelect_N,'max'))
        df.rename(columns={f'grps_{binslide}': "grp"}, inplace=True)
        df['min'] = df['min'].apply(lambda x: round(x,4))
        df['max'] = df['max'].apply(lambda x: round(x,4))
        df['range'] = df["min"].astype(str) + " <-> " + df["max"].astype(str)
        df['Ng'] = df['N'] - df['Nb']
        df['Npct'] = df['N']/st.session_state.N_DEV
        df['br'] = df['Nb']/df['N']
        # Chart
        minbr = df["br"].min().item()
        maxbr = df["br"].max().item()
        minmaxdiff = maxbr - minbr
        p = figure(title="Pop pct & BR by groups", x_range=df["range"],
                   x_axis_label='grps', plot_width = 300, plot_height = 300)
        p.vbar(df["range"], top=df["Npct"], legend_label="Npct", width=.5, 
               color="blue", alpha=.5)
        p.extra_y_ranges['Second'] = Range1d(
            start=max(0.0,minbr-.1*minmaxdiff), 
            end=min(1.0,maxbr+.1*minmaxdiff))
        p.add_layout(LinearAxis(y_range_name='Second'), "right")
        p.line(df["range"], df["br"], legend_label="bad-rate", line_width=2, 
               y_range_name="Second", color="red")
        p.circle(df["range"], df["br"], y_range_name="Second", size=5, 
                 fill_color="black") # only for markers
        p.xaxis.major_label_orientation = math.pi/4
        c.bokeh_chart(p, use_container_width=True)
        with c.expander("Variable Summary Table", expanded=False): st.write(df)


@tictoc
def NonNumer_Only_Compute():
    print("NonNumer_Only_Compute: ", st.session_state.AppState)
    CP2 = st.session_state.CP2
    DV = st.session_state.DV
    DEV = st.session_state.DEV
    OOT = st.session_state.OOT
    N_DEV = st.session_state.N_DEV
    VLIST_C = st.session_state.VLIST_C
    VLIST_FAIL = st.session_state.VLIST_FAIL
    VLISTF_C = VLIST_C
    NLEVELS_C_TBL = pd.DataFrame(); RANKPLOT_C_TBL = pd.DataFrame()
    # adjustForNullChar
    for v in VLIST_C:
        def adjustForNullChar(x_):
            if x_==None or x_=="" or pd.isna(x_): 
                return "NULL"
            else: return x_
        DEV[v] = DEV[v].apply(lambda x: adjustForNullChar(x))
        OOT[v] = OOT[v].apply(lambda x: adjustForNullChar(x))
    # Begin processing
    df = DEV[[v for v in DEV.columns if v==DV or v in (VLIST_C)]]
    k=0 
    for v in df.columns:
        if v != DV:
            NLEVELS_C_TBL.loc[k,'vname'] = v
            NLEVELS_C_TBL.loc[k,'nlevels'] = df[v].nunique()
            dfv = pd.DataFrame()
            grouper = df[[DV,v]].groupby(v, as_index=False)
            dfv['N'] = grouper.count()[DV]
            dfv['Npct'] = dfv['N']/N_DEV
            dfv['Nb'] = grouper.sum()[DV]
            dfv['Ng'] = dfv['N'] - dfv['Nb']
            dfv['br'] = dfv['Nb']/dfv['N']
            dfv['vname'] = v
            dfv['grp'] = grouper.count()[v]
            k+=1
            # Append VLIST_FAIL cases
            if len(dfv) > CP2: 
                temp = pd.DataFrame({"vname":[v], "reason":["manygrpsflg"]})
                VLIST_FAIL = pd.concat([VLIST_FAIL, temp], ignore_index=True)
                dfv = pd.DataFrame()
                VLISTF_C.remove(v)
            elif len(dfv) == 1:
                temp = pd.DataFrame({"vname":[v], "reason":["onegrpflg"]})
                VLIST_FAIL = pd.concat([VLIST_FAIL, temp], ignore_index=True)
                dfv = pd.DataFrame()            
                VLISTF_C.remove(v) 
            if k==1: RANKPLOT_C_TBL = dfv
            else: RANKPLOT_C_TBL = pd.concat([RANKPLOT_C_TBL, dfv], ignore_index=True)
    st.session_state.VLIST_FAIL = VLIST_FAIL
    st.session_state.VLISTF_C = VLISTF_C
    st.session_state.NLEVELS_C_TBL = NLEVELS_C_TBL
    st.session_state.RANKPLOT_C_TBL = RANKPLOT_C_TBL


@tictoc
def NonNumer_Only_Layout():
    print("NonNumer_Only_Layout: ", st.session_state.AppState)
    c = st.container(border=True)
    col1_, col2_ = c.columns([2,3])
    with col1_: st.write(' ')
    with col2_: st.markdown('''Non-Numeric Variables''')
    c.divider()
    if len(st.session_state.VLISTF_C) == 0:
        c.warning("There are no Non-Numeric Variables in the DEV data", icon="⚠️")
    else:
        col1, col2 = c.columns([3,2])
        with col1:
            st.markdown('''Nlevels Table''')
            st.write(st.session_state.NLEVELS_C_TBL)
        with col2:
            st.markdown('''Information Value''')
            st.write(st.session_state.IV_C)
    c.divider()
    VarSelect_C = c.selectbox(
        'Variable Selection', st.session_state.VLISTF_C,
        index=None, placeholder="Select Variable...", key="selboxvar_c") 
    if VarSelect_C != None:
        df = st.session_state.RANKPLOT_C_TBL[(
            st.session_state.RANKPLOT_C_TBL.vname == VarSelect_C)]
        # Chart
        minbr = df["br"].min().item()
        maxbr = df["br"].max().item()
        minmaxdiff = maxbr - minbr
        p = figure(title="Pop pct & BR by groups", x_range=df["grp"],
                   x_axis_label='grps', plot_width = 300, plot_height = 300)
        p.vbar(df["grp"], top=df["Npct"], legend_label="Npct", width=.5, 
               color="blue", alpha=.5)
        p.extra_y_ranges['Second'] = Range1d(
            start=max(0.0,minbr-.1*minmaxdiff), 
            end=min(1.0,maxbr+.1*minmaxdiff))
        p.add_layout(LinearAxis(y_range_name='Second'), "right")
        p.line(df["grp"], df["br"], legend_label="bad-rate", line_width=2, 
               y_range_name="Second", color="red")
        p.circle(df["grp"], df["br"], y_range_name="Second", size=5, 
                 fill_color="black") # only for markers
        p.xaxis.major_label_orientation = math.pi/4
        c.bokeh_chart(p, use_container_width=True)
        with c.expander("Variable Summary Table", expanded=False): st.write(df)

@tictoc
def nWoE():
    print("nWoE: ", st.session_state.AppState)
    NLEVELS_TBL = st.session_state.NLEVELS_TBL
    VLISTF_N = st.session_state.VLISTF_N
    DV = st.session_state.DV
    RANKPLOTMAP_TBL = st.session_state.RANKPLOTMAP_TBL
    N_WOE={}; IV_N = pd.DataFrame()
    missdf = pd.DataFrame(); misslist=[]
    # Missing bin table and list
    missdf = NLEVELS_TBL[NLEVELS_TBL['misspct']!=0]
    if len(missdf) > 0:
        missdf.rename(
            columns={"num_missings": "N", "misspct": "Npct", "Nb_if_miss": "Nb", 
                     "Ng_if_miss": "Ng", "br_if_miss": "br"}, inplace=True)
        missdf = missdf.drop(['nlevels'], axis=1)
        missdf['range'] = "NULL"
        misslist = missdf['vname'].tolist() 
    k=1
    # Identify rank ordering bin all numvars
    for v in VLISTF_N: 
        # Get the slope of bin2
        getslope = RANKPLOTMAP_TBL[v].groupby('grps_2', as_index=False).agg(
            N=(DV,'count'), Nb=(DV,'sum'))
        getslope['Npct'] = getslope['N']/st.session_state.N_DEV
        getslope['br'] = getslope['Nb']/getslope['N']
        slope = getslope['br'].diff().tail(1).item() #st.write(v, slope)
        # Binning Algorithm
        nlevels = NLEVELS_TBL.loc[NLEVELS_TBL['vname']==v]['nlevels'].item()
        maxgrps = min(15,int(nlevels)) 
        for b in range(maxgrps, 1, -1):
            df = RANKPLOTMAP_TBL[v].groupby(f'grps_{b}', as_index=False).agg(
                N=(DV,'count'), Nb=(DV,'sum'), min=(v,'min'), max=(v,'max'))
            df['vname'] = v
            df.rename(columns={f'grps_{b}': "bin"}, inplace=True)
            df['br'] = df['Nb']/df['N']
            if slope < 0: # reverse bin order if slope is negative
                df.sort_values(by=['bin'], inplace=True, ascending=False)
            diff = df['br'].diff()
            stillbreakrank = any(r < 0 for r in diff) #st.write(v,b,stillbreakrank)
            stillbreakrank2 = any(r < .001 for r in df['Nb']/st.session_state.NB_DEV)
            if stillbreakrank == False and stillbreakrank2 == False: break
        df['range'] = df["min"].astype(str) + " <-> " + df["max"].astype(str)
        df['Ng'] = df['N'] - df['Nb']
        df['Npct'] = df['N']/st.session_state.N_DEV
        if v in misslist:
            df = pd.concat([df, missdf[missdf['vname']==v]], ignore_index=True)
        # Calc WoE & IV
        df.sort_values(by=['max'], inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        df['bin'] = df.index+1
        df['Nb_pct'] = df['Nb']/st.session_state.NB_DEV    
        df['Ng_pct'] = df['Ng']/st.session_state.NG_DEV 
        df['WOE'] = np.log(df['Ng_pct'] / df['Nb_pct'])
        df['WOE'] = df.apply(
            lambda x: 0 if x['Nb']==0 or x['Ng']==0 else x['WOE'], axis=1)
        df['IV'] = (df['Ng_pct'] - df['Nb_pct']) * df['WOE']
        N_WOE[v] = df #st.write(df)
        IV_N.loc[k,'vname'] = v
        IV_N.loc[k,'numbins'] = df['bin'].count()
        IV_N.loc[k,'IV'] = df['IV'].sum()
        k+=1
    st.session_state.N_WOE = N_WOE
    st.session_state.IV_N = IV_N


@tictoc
def cWoE():
    print("cWoE: ", st.session_state.AppState)
    VLISTF_C = st.session_state.VLISTF_C
    RANKPLOT_C_TBL = st.session_state.RANKPLOT_C_TBL
    C_WOE={}; IV_C = pd.DataFrame()
    k=1
    for v in VLISTF_C: 
        df = RANKPLOT_C_TBL[RANKPLOT_C_TBL['vname']==v]
        df.sort_values(by=['br'], inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        df['bin'] = df.index+1
        df['Nb_pct'] = df['Nb']/st.session_state.NB_DEV    
        df['Ng_pct'] = df['Ng']/st.session_state.NG_DEV 
        df['WOE'] = np.log(df['Ng_pct'] / df['Nb_pct'])
        df['WOE'] = df.apply(
            lambda x: 0 if x['Nb']==0 or x['Ng']==0 else x['WOE'], axis=1)
        df['IV'] = (df['Ng_pct'] - df['Nb_pct']) * df['WOE']
        C_WOE[v] = df #st.write(df)
        IV_C.loc[k,'vname'] = v
        IV_C.loc[k,'numbins'] = df['bin'].count()
        IV_C.loc[k,'IV'] = df['IV'].sum()
        k+=1
    st.session_state.C_WOE = C_WOE
    st.session_state.IV_C = IV_C


@tictoc
def create_wDEVOOT():
    print("create_wDEVOOT: ", st.session_state.AppState)
    DEV = st.session_state.DEV
    OOT = st.session_state.OOT
    NLEVELS_TBL = st.session_state.NLEVELS_TBL
    VLISTF_N = st.session_state.VLISTF_N
    VLISTF_C = st.session_state.VLISTF_C
    VLIST_FAIL = st.session_state.VLIST_FAIL
    N_WOE = st.session_state.N_WOE
    C_WOE = st.session_state.C_WOE
    wDEV = DEV; wOOT = OOT
    # Create WoE for Numeric
    faillist = ['empty_field_']
    if len(VLIST_FAIL) != 0: faillist = VLIST_FAIL['vname'].tolist()
    for v in VLISTF_N: 
        if v not in faillist:
            woetbl = N_WOE[v] #st.write(v, "woe", woetbl)
            cnt = woetbl['vname'].count()
            def applywoe(x_):
                if x_ == None or math.isnan(x_) or np.isnan(x_) or pd.isna(x_): 
                    return woetbl.loc[cnt-1,'WOE']
                for r in range(cnt):
                    if x_ <= woetbl.loc[r,'max']: return woetbl.loc[r,'WOE']
                return woetbl.loc[cnt-1,'WOE']
            wDEV['w'+v] = DEV[v].apply(lambda x: applywoe(x))
            wOOT['w'+v] = OOT[v].apply(lambda x: applywoe(x))
            # Replace variables with missings, with WoE-based counterparts
            if v in NLEVELS_TBL[NLEVELS_TBL['misspct']!=0]['vname'].tolist():
                wDEV.rename(columns={v: 'n'+v}, inplace=True)
                wDEV.rename(columns={'w'+v: v}, inplace=True)
                wOOT.rename(columns={v: 'n'+v}, inplace=True)
                wOOT.rename(columns={'w'+v: v}, inplace=True)
        #st.dataframe(wDEV); st.dataframe(wOOT)          
    # Create WoE for Non-Numeric
    for v in VLISTF_C: 
        if v not in faillist:    
            woetbl = C_WOE[v] #st.write(v, "woe", woetbl)
            cnt = woetbl['vname'].count()
            def applywoe_c(x_):
                for r in range(cnt):
                    if x_ == woetbl.loc[r,'grp']: return woetbl.loc[r,'WOE']
            wDEV['w'+v] = wDEV[v].apply(lambda x: applywoe_c(x))
            wOOT['w'+v] = wOOT[v].apply(lambda x: applywoe_c(x))
            # Replace variables with missings, with WoE-based counterparts
            wDEV.rename(columns={v: 'c'+v}, inplace=True)
            wDEV.rename(columns={'w'+v: v}, inplace=True)
            wOOT.rename(columns={v: 'c'+v}, inplace=True)
            wOOT.rename(columns={'w'+v: v}, inplace=True)
        #st.dataframe(wDEV); st.dataframe(wOOT) 
    st.session_state.wDEV = wDEV
    st.session_state.wOOT = wOOT


@tictoc
def PSI():    
    print("PSI: ", st.session_state.AppState)
    VLISTF_N = st.session_state.VLISTF_N
    VLISTF_C = st.session_state.VLISTF_C
    N_WOE = st.session_state.N_WOE
    C_WOE = st.session_state.C_WOE
    wOOT = st.session_state.wOOT
    PSI = pd.DataFrame()
    for v in VLISTF_N + VLISTF_C: 
        if v in VLISTF_N: woetbl = N_WOE[v]
        else: woetbl = C_WOE[v]
        df = wOOT.groupby(v, as_index=False).agg(N=(v,'count'))
        df['Npct_OOT'] = df['N']/st.session_state.N_OOT
        df = woetbl.set_index('WOE').join(
            df[[v,'Npct_OOT']].set_index(v))
        df['PSI_i'] = (df['Npct']-df['Npct_OOT'])*np.log(
            df['Npct']/df['Npct_OOT'])
        psi_sum = df['PSI_i'].sum()
        temp = pd.DataFrame({"vname":[v], "PSI":[psi_sum]})
        PSI = pd.concat([PSI, temp], ignore_index=True) #st.write(v, "PSI", df)
    st.session_state.PSI = PSI


@tictoc
def CORR():
    print("CORR: ", st.session_state.AppState)
    VLISTF_N = st.session_state.VLISTF_N
    VLISTF_C = st.session_state.VLISTF_C
    DV = st.session_state.DV
    wDEV = st.session_state.wDEV
    CORRMAT = pd.DataFrame(); CORRIV = pd.DataFrame()
    df = wDEV[[v for v in wDEV.columns 
               if v==DV or v in (VLISTF_N + VLISTF_C)]]
    CORRMAT = df.corr(method='spearman')
    CORRIV = CORRMAT[[DV]]
    CORRIV['vname'] = CORRIV.index
    CORRIV.reset_index(drop=True, inplace=True)
    CORRIV.rename(columns={DV: 'corr'}, inplace=True)
    CORRIV = CORRIV[CORRIV['vname'] != DV]
    CORRIV['abscorr'] = abs(CORRIV['corr'])
    st.session_state.CORRMAT = CORRMAT
    st.session_state.CORRIV = CORRIV


@tictoc
def Cluster():
    print("Cluster: ", st.session_state.AppState)
    VLISTF_N = st.session_state.VLISTF_N
    VLISTF_C = st.session_state.VLISTF_C
    wDEV = st.session_state.wDEV
    CLUSTER = pd.DataFrame()
    df = wDEV[[v for v in wDEV.columns if v in (VLISTF_N + VLISTF_C)]]
    varclus = VarClusHi(df,maxeigval2=.7,maxclus=20)
    varclus.varclus()
    CLUSTER = varclus.rsquare
    #varclus.clusters
    st.session_state.CLUSTER = CLUSTER


def get_params(Xmat,p_,model_):
    varnames = []; varnames.append("Intercept")
    varnames = varnames + list(Xmat.columns)
    # Design matrix: add col of 1's at the beginning of X
    X_design = np.hstack([np.ones((Xmat.shape[0], 1)), Xmat])
    # Initiate matrix of 0's, fill diagonal with each predicted obs variance
    V = np.diagflat(np.prod(p_, axis=1))
    covMat = np.linalg.pinv(X_design.T @ V @ X_design)
    SE = np.sqrt(np.diag(covMat))
    coefVector = np.insert(model_.coef_, 0, model_.intercept_)
    t =  coefVector / SE  
    pvalues = (1 - norm.cdf(abs(t))) * 2
    waldchisq = t ** 2
    results = pd.DataFrame([varnames, coefVector, pvalues, waldchisq]).T
    results.columns=["vname","coef","pval","waldchisq"]
    return results

def get_metrics(y_, p_, typ, mdl):
    auc = metrics.roc_auc_score(y_, p_[:,1]); gini = 2*auc-1
    fpr, tpr, thresholds = metrics.roc_curve(y_, p_[:,1])
    # Alternative auc using ROC curve -> auc2 = metrics.auc(fpr, tpr)
    ROC_ = pd.DataFrame([fpr, tpr, thresholds]).T
    ROC_.columns=["fpr","tpr","thresholds"] # fpr is xaxis, tpr is yaxis
    idx = np.argmax(tpr - fpr)
    ks = tpr[idx] - fpr[idx]
    p_thresh = thresholds[idx]
    newp_ = np.where(p_[:,1] > p_thresh, True, False)
    # CM Metrics
    CM = confusion_matrix(y_, newp_)
    TN, FP, FN, TP = CM.ravel() 
    Precision = round(precision_score(y_, newp_),3) # PS = TP / (TP + FP)
    Recall = round(recall_score(y_, newp_),3) # RS = TP / (TP + FN)
    F1 = round(f1_score(y_, newp_),3) # F1 = (2 * PS * RS) / (PS + RS)
    cm_results = [typ, mdl, ks, gini, auc, TP, TN, FP, FN, 
                  Precision, Recall, F1]
    CM_TBL_ = pd.DataFrame(cm_results).T
    CM_TBL_.columns=["DS", "Model", "KS", "GINI", "AUC",
                     "TP", "TN", "FP", "FN", "Precision","Recall","F1"]
    return ROC_, p_thresh, CM_TBL_
    
def Rank_p(y_, p_, yoot_, poot_):
    prank = pd.concat([pd.DataFrame(y_), pd.DataFrame(p_[:,1])], axis=1)
    prank.columns=["y", "p"]
    pootrank = pd.concat([pd.DataFrame(yoot_), pd.DataFrame(poot_[:,1])], axis=1)
    pootrank.columns=["y", "p"]
    prank.sort_values(by='p', ascending=False, inplace=True)
    prank.reset_index(drop=True, inplace=True)
    prank['pct_'] = (prank.index+1)/st.session_state.N_DEV # df.index is _N_-1
    prank['changeflag'] = prank['p'].diff()
    global j_; j_=1
    def jinc(): global j_; j_+=1; return j_
    prank['decile'] = prank.apply(
        lambda x: jinc() if x['changeflag']<0 and 
        x['pct_'] > j_/10 else j_, axis=1)
    prankg = prank.groupby('decile', as_index=False).agg(
        N_dev=('y','count'), Nb_dev=('y','sum'), min=('p','min'))
    prankg['Npct_dev'] = prankg['N_dev'] / st.session_state.N_DEV
    prankg['br_dev'] = prankg['Nb_dev'] / prankg['N_dev']
    cnt = len(prankg)
    def applyrank(x_):
        for r in range(cnt): 
            if x_ >= prankg.loc[r,'min']: return r+1
        return cnt
    pootrank['decile'] = pootrank['p'].apply(lambda x: applyrank(x))
    pootrankg = pootrank.groupby('decile', as_index=False).agg(
        N_oot=('y','count'), Nb_oot=('y','sum'))
    pootrankg['Npct_oot'] = pootrankg['N_oot'] / st.session_state.N_OOT
    pootrankg['br_oot'] = pootrankg['Nb_oot'] / pootrankg['N_oot']
    prankf = prankg.set_index('decile').join(
        pootrankg.set_index('decile'))
    prankf.reset_index(inplace=True)		
    return prankf

def Model_LOGISTIC(X_, y_, Xoot_, yoot_):
    print("Model_LOGISTIC: ", st.session_state.AppState)
    LR = None; LR_RESULTS_ = {}
    # Initial Reg
    LR = LogisticRegression(
        random_state=0, max_iter=1000, n_jobs=-1).fit(X_, y_) 
    p = LR.predict_proba(X_) # ndarray p_0 & p_1, use [:,1]
    PARAMS = get_params(X_,p,LR)
    pval_high = PARAMS[
        (PARAMS['pval']>.05) & (PARAMS['vname'] != "Intercept")]['waldchisq'].min()
    X_temp = X_.copy(); Xoot_temp = Xoot_.copy()
    # Makeshift stepwise Reg
    for i in range(1,len(X_.columns)): 
        if math.isnan(pval_high): break
        vname_out = PARAMS[PARAMS['waldchisq']==pval_high]['vname'].item()
        X_temp.drop(columns=vname_out, inplace=True)
        Xoot_temp.drop(columns=vname_out, inplace=True)
        LR = LogisticRegression(
            random_state=0, max_iter=1000, n_jobs=-1).fit(X_temp, y_) 
        p = LR.predict_proba(X_temp) # ndarray p_0 & p_1, use [:,1]
        PARAMS = get_params(X_temp,p,LR)
        pval_high = PARAMS[
            (PARAMS['pval']>.05) & (PARAMS['vname'] != "Intercept")]['waldchisq'].min()
    poot = LR.predict_proba(Xoot_temp)
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","LR")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","LR")
    LR_RESULTS_['PARAMS'] = PARAMS
    LR_RESULTS_['ROC_DEV'] = ROC_DEV; LR_RESULTS_['CM_DEV'] = CM_DEV
    LR_RESULTS_['ptdev'] = ptdev
    LR_RESULTS_['ROC_OOT'] = ROC_OOT; LR_RESULTS_['CM_OOT'] = CM_OOT
    LR_RESULTS_['ptoot'] = ptoot
    LR_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    return LR_RESULTS_

def Add_Interactions_LR(X_, y_, Xoot_):
    VLISTF_N = st.session_state.VLISTF_N
    VLISTF_C = st.session_state.VLISTF_C
    NLEVELS_TBL = st.session_state.NLEVELS_TBL
    ilist = NLEVELS_TBL[NLEVELS_TBL['misspct']==0]['vname'].tolist()
    ilist = [item for item in ilist if item in VLISTF_N]
    df = pd.DataFrame()
    Xi = X_.copy(); Xioot = Xoot_.copy()
    for i in ilist:
        for j in ilist:
            if ilist.index(i) <= ilist.index(j):
                iname = "interact_"+str(ilist.index(i))+"_"+str(ilist.index(j))
                Xi[iname] = X_[i] * X_[j]
                Xioot[iname] = Xoot_[i] * Xoot_[j]
                Xi_ = Xi[[iname]]
                iLR = LogisticRegression(
                    random_state=0, fit_intercept=False).fit(Xi_, y_) 
                ip = iLR.predict_proba(Xi_)
                iroc, iptdev, icm_dev = get_metrics(y_,ip,iname,"iLR")
                icm_dev.loc[0,'A']=i
                icm_dev.loc[0,'B']=j
                df = pd.concat([df,icm_dev[['DS','A','B','AUC']]])
    imap = df[(df['AUC']>.5)].sort_values(by='AUC', ascending=False).head(5)
    topinters = imap['DS'].tolist()
    Xi = Xi[[v for v in Xi.columns if v in (VLISTF_N + VLISTF_C + topinters)]]
    Xioot = Xioot[[v for v in Xioot.columns if v in (VLISTF_N + VLISTF_C + topinters)]]
    st.session_state.IMAP = imap
    return Xi, Xioot

def Model_DTREE(X_, y_, Xoot_, yoot_, doHyper=0):    
    DT = None; DT_RESULTS_ = {}
    minleafsize = int(0.01*st.session_state.N_DEV)
    if doHyper == 1:
        param_grid = {'criterion':["gini","entropy"],
                      'max_depth': list(range(3,len(X_.columns)))}
        dt_ = DecisionTreeClassifier(
            random_state=0, min_samples_leaf=minleafsize)
        grid_search = GridSearchCV(dt_, param_grid, cv=4)
        grid_search.fit(X_, y_)  
        DT = DecisionTreeClassifier(
            random_state=0, min_samples_leaf=minleafsize,
            criterion=grid_search.best_params_['criterion'],
            max_depth=grid_search.best_params_['max_depth']).fit(X_, y_)
    else:
        DT = DecisionTreeClassifier(
            random_state=0, min_samples_leaf=minleafsize).fit(X_, y_)
    p = DT.predict_proba(X_) # ndarray p_0 & p_1, use [:,1]
    poot = DT.predict_proba(Xoot_)
    FEATIMP = pd.DataFrame([DT.feature_names_in_, DT.feature_importances_]).T
    FEATIMP.columns=["vname", "Importance"]
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","DT")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","DT")
    PARAMS = pd.DataFrame.from_dict(DT.get_params(), orient='index')    
    DT_RESULTS_['FEATIMP'] = FEATIMP
    DT_RESULTS_['PARAMS'] = PARAMS
    DT_RESULTS_['ROC_DEV'] = ROC_DEV; DT_RESULTS_['CM_DEV'] = CM_DEV
    DT_RESULTS_['ptdev'] = ptdev
    DT_RESULTS_['ROC_OOT'] = ROC_OOT; DT_RESULTS_['CM_OOT'] = CM_OOT
    DT_RESULTS_['ptoot'] = ptoot
    DT_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    DT_RESULTS_['DT'] = DT
    return DT_RESULTS_

def Model_RFOREST(X_, y_, Xoot_, yoot_, doHyper=0): 
    RF = None; RF_RESULTS_ = {}
    minleafsize = int(0.01*st.session_state.N_DEV)
    if doHyper == 1:
        param_grid = {'n_estimators': [100, 250],
                      'max_depth': [3, 5, 8, 11, None],
                      'oob_score': [True, False]}
        rf_ = RandomForestClassifier(
            random_state=0, min_samples_leaf=minleafsize, criterion='gini',
            max_features='sqrt', n_jobs=-1)
        grid_search = GridSearchCV(rf_, param_grid, cv=4)
        grid_search.fit(X_, y_)  
        RF = RandomForestClassifier(
            random_state=0, min_samples_leaf=minleafsize, criterion='gini', 
            max_features='sqrt', n_jobs=-1,
            n_estimators=grid_search.best_params_['n_estimators'],
            max_depth=grid_search.best_params_['max_depth'],
            oob_score=grid_search.best_params_['oob_score']).fit(X_, y_)  
    else:
        RF = RandomForestClassifier(
            random_state=0, min_samples_leaf=minleafsize, n_jobs=-1,
            n_estimators=150, max_depth=10).fit(X_, y_) 
    p = RF.predict_proba(X_)
    poot = RF.predict_proba(Xoot_)
    FEATIMP = pd.DataFrame([RF.feature_names_in_, RF.feature_importances_]).T
    FEATIMP.columns=["vname", "Importance"]
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","RF")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","RF")
    PARAMS = pd.DataFrame.from_dict(RF.get_params(), orient='index')  
    RF_RESULTS_['FEATIMP'] = FEATIMP
    RF_RESULTS_['PARAMS'] = PARAMS
    RF_RESULTS_['ROC_DEV'] = ROC_DEV; RF_RESULTS_['CM_DEV'] = CM_DEV
    RF_RESULTS_['ptdev'] = ptdev
    RF_RESULTS_['ROC_OOT'] = ROC_OOT; RF_RESULTS_['CM_OOT'] = CM_OOT
    RF_RESULTS_['ptoot'] = ptoot
    RF_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    return RF_RESULTS_

def Model_GBM(X_, y_, Xoot_, yoot_, doHyper=0): 
    GBM = None; GBM_RESULTS_ = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_, test_size=.2, stratify=y_, random_state=0)
    minleafsize = int(0.01*st.session_state.N_DEV)
    if doHyper == 1:
        param_grid = {'n_estimators': [100, 250],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'max_depth': [3, 5, 8, 11, None]}
        gbm_ = GradientBoostingClassifier(
            random_state=0, min_samples_leaf=minleafsize, 
            max_features='sqrt')
        grid_search = GridSearchCV(gbm_, param_grid, cv=4)
        grid_search.fit(X_, y_)  
        GBM = GradientBoostingClassifier(
            random_state=0, min_samples_leaf=minleafsize, 
            max_features='sqrt',
            n_estimators=grid_search.best_params_['n_estimators'],
            learning_rate=grid_search.best_params_['learning_rate'],
            max_depth=grid_search.best_params_['max_depth']).fit(
                X_train, y_train)  
    else:
        GBM = GradientBoostingClassifier(
            random_state=0, min_samples_leaf=minleafsize, 
            n_estimators=150, max_depth=10, max_features='sqrt').fit(
                X_train, y_train)  
    p = GBM.predict_proba(X_)
    poot = GBM.predict_proba(Xoot_)
    FEATIMP = pd.DataFrame([GBM.feature_names_in_, GBM.feature_importances_]).T
    FEATIMP.columns=["vname", "Importance"]
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","GBM")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","GBM")
    PARAMS = pd.DataFrame.from_dict(GBM.get_params(), orient='index') 
    GBM_RESULTS_['FEATIMP'] = FEATIMP
    GBM_RESULTS_['PARAMS'] = PARAMS
    GBM_RESULTS_['ROC_DEV'] = ROC_DEV; GBM_RESULTS_['CM_DEV'] = CM_DEV
    GBM_RESULTS_['ptdev'] = ptdev
    GBM_RESULTS_['ROC_OOT'] = ROC_OOT; GBM_RESULTS_['CM_OOT'] = CM_OOT
    GBM_RESULTS_['ptoot'] = ptoot
    GBM_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    return GBM_RESULTS_

def Model_XGB(X_, y_, Xoot_, yoot_, doHyper=0): 
    XGB = None; XGB_RESULTS_ = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_, test_size=.2, stratify=y_, random_state=0)
    eval_set = [(X_test, y_test)]
    if doHyper == 1:
        param_grid = {'n_estimators': [100, 250],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'max_depth': [3, 5, 8, 11, None]}
        xgb_ = xgb.XGBClassifier(random_state=0, n_jobs=-1)
        grid_search = GridSearchCV(xgb_, param_grid, cv=4)
        grid_search.fit(X_, y_)  
        XGB = xgb.XGBClassifier(
            random_state=0, n_jobs=-1, 
            n_estimators=grid_search.best_params_['n_estimators'],
            learning_rate=grid_search.best_params_['learning_rate'],
            max_depth=grid_search.best_params_['max_depth']).fit(
                X_train, y_train, eval_set=eval_set, early_stopping_rounds=50)  
    else:
        XGB = xgb.XGBClassifier(
            random_state=0, n_jobs=-1, max_depth=10, n_estimators=150).fit(
                X_train, y_train, eval_set=eval_set, early_stopping_rounds=50)  
    p = XGB.predict_proba(X_)
    poot = XGB.predict_proba(Xoot_)
    # xgb.plot_importance(XGB) is absolute, not % based
    FEATIMP = pd.DataFrame([XGB.feature_names_in_, XGB.feature_importances_]).T
    FEATIMP.columns=["vname", "Importance"]
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","XGB")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","XGB")
    PARAMS = pd.DataFrame.from_dict(XGB.get_params(), orient='index') 
    XGB_RESULTS_['FEATIMP'] = FEATIMP
    XGB_RESULTS_['PARAMS'] = PARAMS
    XGB_RESULTS_['ROC_DEV'] = ROC_DEV; XGB_RESULTS_['CM_DEV'] = CM_DEV
    XGB_RESULTS_['ptdev'] = ptdev
    XGB_RESULTS_['ROC_OOT'] = ROC_OOT; XGB_RESULTS_['CM_OOT'] = CM_OOT
    XGB_RESULTS_['ptoot'] = ptoot
    XGB_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    return XGB_RESULTS_

def Model_MLP(X_, y_, Xoot_, yoot_, doHyper=0): 
    MLP = None; MLP_RESULTS_ = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_, test_size=.2, stratify=y_, random_state=0)
    eval_set = [(X_test, y_test)]
    if doHyper == 1:
        param_grid = {'activation': ['identity', 'relu'],
                      'max_iter': [200, 300],
                      'learning_rate': ['constant','adaptive']}
        mlp_ = MLPClassifier( #early_stopping
            random_state=0, hidden_layer_sizes=(50), activation='identity')
        grid_search = GridSearchCV(mlp_, param_grid, cv=4)
        grid_search.fit(X_, y_)  
        MLP = MLPClassifier(
            random_state=0, hidden_layer_sizes=(50), 
            activation=grid_search.best_params_['activation'],
            max_iter=grid_search.best_params_['max_iter'],
            learning_rate=grid_search.best_params_['learning_rate']).fit(
                X_train, y_train)  
    else:
        MLP = MLPClassifier(
            random_state=0, max_iter=500, hidden_layer_sizes=(100,25),
            activation='identity').fit(X_train, y_train)  
    p = MLP.predict_proba(X_)
    poot = MLP.predict_proba(Xoot_)
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","MLP")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","MLP")
    PARAMS = pd.DataFrame.from_dict(MLP.get_params(), orient='index') 
    MLP_RESULTS_['PARAMS'] = PARAMS
    MLP_RESULTS_['ROC_DEV'] = ROC_DEV; MLP_RESULTS_['CM_DEV'] = CM_DEV
    MLP_RESULTS_['ptdev'] = ptdev
    MLP_RESULTS_['ROC_OOT'] = ROC_OOT; MLP_RESULTS_['CM_OOT'] = CM_OOT
    MLP_RESULTS_['ptoot'] = ptoot
    MLP_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    return MLP_RESULTS_

def Model_LGB(X_, y_, Xoot_, yoot_, doHyper=0): 
    LGB = None; LGB_RESULTS_ = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X_, y_, test_size=.2, stratify=y_, random_state=0)
    eval_set = [(X_test, y_test)]
    minleafsize = int(0.01*st.session_state.N_DEV)
    if doHyper == 1:
        param_grid = {'n_estimators': [100, 250],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'max_depth': [3, 5, 8, 11, None]}
        lgb_ = lgb.LGBMClassifier(
            random_state=0, n_jobs=-1, importance_type='gain',
            min_child_samples=minleafsize)
        grid_search = GridSearchCV(lgb_, param_grid, cv=4)
        grid_search.fit(X_, y_)  
        LGB = lgb.LGBMClassifier(
            random_state=0, n_jobs=-1, min_child_samples=minleafsize,
            importance_type='gain', 
            n_estimators=grid_search.best_params_['n_estimators'],
            learning_rate=grid_search.best_params_['learning_rate'],
            max_depth=grid_search.best_params_['max_depth']).fit(
                X_train, y_train, eval_set=eval_set)  
    else:
        LGB = lgb.LGBMClassifier(
            random_state=0, n_jobs=-1, max_depth=10, importance_type='gain',
            min_child_samples=minleafsize, n_estimators=150).fit(
                X_train, y_train, eval_set=eval_set)  
    p = LGB.predict_proba(X_)
    poot = LGB.predict_proba(Xoot_)
    #st.pyplot(lgb.plot_importance(LGB).figure)
    FEATIMP = pd.DataFrame([LGB.feature_name_, LGB.feature_importances_]).T
    FEATIMP.columns=["vname", "Importance"]
    ROC_DEV, ptdev, CM_DEV = get_metrics(y_,p,"dev","LGB")
    ROC_OOT, ptoot, CM_OOT = get_metrics(yoot_,poot,"oot","LGB")
    PARAMS = pd.DataFrame.from_dict(LGB.get_params(), orient='index') 
    LGB_RESULTS_['FEATIMP'] = FEATIMP
    LGB_RESULTS_['PARAMS'] = PARAMS
    LGB_RESULTS_['ROC_DEV'] = ROC_DEV; LGB_RESULTS_['CM_DEV'] = CM_DEV
    LGB_RESULTS_['ptdev'] = ptdev
    LGB_RESULTS_['ROC_OOT'] = ROC_OOT; LGB_RESULTS_['CM_OOT'] = CM_OOT
    LGB_RESULTS_['ptoot'] = ptoot
    LGB_RESULTS_['Rankp'] = Rank_p(y_, p, yoot_, poot)
    return LGB_RESULTS_


@tictoc
def Oracle():
    print("Oracle: ", st.session_state.AppState)
    DV = st.session_state.DV
    VLISTF_N = st.session_state.VLISTF_N
    VLISTF_C = st.session_state.VLISTF_C
    wDEV = st.session_state.wDEV
    wOOT = st.session_state.wOOT
    CLUSTER = st.session_state.CLUSTER
    X = wDEV[[v for v in wDEV.columns if v in (VLISTF_N + VLISTF_C)]]
    Xoot = wOOT[[v for v in wOOT.columns if v in (VLISTF_N + VLISTF_C)]]
    y = wDEV[[DV]].values.ravel()
    yoot = wOOT[[DV]].values.ravel()
    if st.session_state.AppState == "Cluster_Complete":  
        LR_RESULTS = {}
        LR_RESULTS = Model_LOGISTIC(X, y, Xoot, yoot) 
        st.session_state.LR_RESULTS = LR_RESULTS
        iLR_RESULTS = {}
        Xi, Xioot = Add_Interactions_LR(X, y, Xoot)
        iLR_RESULTS = Model_LOGISTIC(Xi, y, Xioot, yoot) 
        st.session_state.iLR_RESULTS = iLR_RESULTS
    if st.session_state.ALT_MDLS != None:
        if "D-Tree" in st.session_state.ALT_MDLS:
            DT_RESULTS = {}      
            DT_RESULTS = Model_DTREE(X, y, Xoot, yoot)  
            st.session_state.DT_RESULTS = DT_RESULTS 
        if "Random Forest" in st.session_state.ALT_MDLS:
            RF_RESULTS = {}  
            RF_RESULTS = Model_RFOREST(X, y, Xoot, yoot) 
            st.session_state.RF_RESULTS = RF_RESULTS
        if "GBM" in st.session_state.ALT_MDLS:
            GBM_RESULTS = {}
            GBM_RESULTS = Model_GBM(X, y, Xoot, yoot) 
            st.session_state.GBM_RESULTS = GBM_RESULTS
        if "XGB" in st.session_state.ALT_MDLS:
            XGB_RESULTS = {}
            XGB_RESULTS = Model_XGB(X, y, Xoot, yoot) 
            st.session_state.XGB_RESULTS = XGB_RESULTS
        if "MLP-NN" in st.session_state.ALT_MDLS:
            MLP_RESULTS = {}
            MLP_RESULTS = Model_MLP(X, y, Xoot, yoot) 
            st.session_state.MLP_RESULTS = MLP_RESULTS
        if "LGB" in st.session_state.ALT_MDLS:
            LGB_RESULTS = {}
            LGB_RESULTS = Model_LGB(X, y, Xoot, yoot) 
            st.session_state.LGB_RESULTS = LGB_RESULTS

    
def Oracle_Layout_assist(DF, mdltyp):
    ROC_DEV = DF['ROC_DEV']; ROC_OOT = DF['ROC_OOT']
    ptdev = DF['ptdev']; ptoot = DF['ptoot']
    Rankp = DF['Rankp']
    col1, col2 = st.columns(2)
    if mdltyp in ("LOGISTIC REGRESSION","LOGISTIC REGRESSION - interact"):
        col1.write(DF['PARAMS'].sort_values(by='waldchisq', ascending=False))
        if mdltyp == "LOGISTIC REGRESSION - interact":
            col2.write(st.session_state.IMAP)
    else: 
        col1.write(DF['PARAMS'])
        if mdltyp != "MLP-NN":
            col2.write(DF['FEATIMP'].sort_values(by='Importance', ascending=False))
    st.write(pd.concat([DF['CM_DEV'], DF['CM_OOT']], ignore_index=True)) 
    if mdltyp == "DECISION TREE": st.graphviz_chart(export_graphviz(DF['DT']))
    # Decile chart
    deciles = map(str,Rankp['decile'].tolist())
    datasrc = ['DEV', 'OOT']
    x = [(decile, src) for decile in deciles for src in datasrc]
    fig = figure(x_range=FactorRange(*x), 
                 title='Decile Chart', x_axis_label='deciles', 
                 y_axis_label='avg bad-rate', plot_height=500, plot_width=750)
    counts = sum(zip(Rankp['br_dev'], Rankp['br_oot']), ()) 
    source = ColumnDataSource(data=dict(x=x, counts=counts))
    fig.vbar(x='x', top='counts', width=0.9, source=source, line_color="white",
             fill_color=factor_cmap(
                 'x', palette=['blue','red'], factors=datasrc, start=1, end=2))
    fig.legend.location = "bottom_right"
    st.bokeh_chart(fig, use_container_width=False) 
    # ROC chart
    fig2 = figure(title='ROC', x_axis_label='fpr', y_axis_label='tpr', 
                  plot_height=500, plot_width=500)
    fig2.line(ROC_DEV['fpr'], ROC_DEV['tpr'], line_width=2, color="blue",
              legend_label=f'ROC [dev], pthresh={round(ptdev,3)}')
    fig2.line(ROC_OOT['fpr'], ROC_OOT['tpr'], line_width=2, color="red",
              legend_label=f'ROC [oot], pthresh={round(ptoot,3)}')
    fig2.line(ROC_DEV['fpr'], ROC_DEV['fpr'], line_width=2, color="black",
              legend_label='', line_dash='dashed')
    fig2.legend.location = "bottom_right"
    st.bokeh_chart(fig2, use_container_width=False)

@tictoc
def Oracle_Layout():
    print("Oracle_Layout: ", st.session_state.AppState)
    LR_RESULTS = st.session_state.LR_RESULTS
    iLR_RESULTS = st.session_state.iLR_RESULTS
    DT_RESULTS = st.session_state.DT_RESULTS
    RF_RESULTS = st.session_state.RF_RESULTS
    GBM_RESULTS = st.session_state.GBM_RESULTS
    XGB_RESULTS = st.session_state.XGB_RESULTS   
    MLP_RESULTS = st.session_state.MLP_RESULTS 
    LGB_RESULTS = st.session_state.LGB_RESULTS 
    choices = st.session_state.ALT_MDLS
    with st.expander("LOGISTIC REGRESSION", expanded=False):
        if st.session_state.AppState == "Oracle_Complete":
            Oracle_Layout_assist(LR_RESULTS, "LOGISTIC REGRESSION")
    # LR Interaction Toggle button
    st.session_state.interact = st.toggle(
        "Interactions", value=False, key="interacttoggle")
    with st.expander("LOGISTIC REGRESSION - interact", expanded=False):
        if st.session_state.AppState == "Oracle_Complete" and st.session_state.interact == True:
            Oracle_Layout_assist(iLR_RESULTS, "LOGISTIC REGRESSION - interact")
    if choices != None:
        with st.expander("DECISION TREE", expanded=False):
            if "D-Tree" in choices and DT_RESULTS != None:
                Oracle_Layout_assist(DT_RESULTS, "DECISION TREE")
        with st.expander("RANDOM FOREST", expanded=False):
            if "Random Forest" in choices and RF_RESULTS != None:
                Oracle_Layout_assist(RF_RESULTS, "RANDOM FOREST")
        with st.expander("GRADIENT BOOST", expanded=False):
            if "GBM" in choices and GBM_RESULTS != None:
                Oracle_Layout_assist(GBM_RESULTS, "GRADIENT BOOST")
        with st.expander("XG BOOST", expanded=False):
            if "XGB" in choices and XGB_RESULTS != None:
                Oracle_Layout_assist(XGB_RESULTS, "XG BOOST")
        with st.expander("NEURAL NETWORK", expanded=False):
            if "MLP-NN" in choices and MLP_RESULTS != None:
                Oracle_Layout_assist(MLP_RESULTS, "MLP-NN")
        with st.expander("LIGHT G BOOST", expanded=False):
            if "LGB" in choices and LGB_RESULTS != None:
                Oracle_Layout_assist(LGB_RESULTS, "LGB")

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
# Define Globals
def Initialize():
    print("Initialize: ", st.session_state.AppState)
    if "RNG" not in st.session_state: st.session_state.RNG=np.random.RandomState(0)
    if "DS" not in st.session_state: st.session_state.DS=None
    if "CP1" not in st.session_state: st.session_state.CP1=None
    if "CP2" not in st.session_state: st.session_state.CP2=None
    if "VALID_DV" not in st.session_state: st.session_state.VALID_DV=None
    if "DS_SPLITTER" not in st.session_state: st.session_state.DS_SPLITTER=None
    if "DV" not in st.session_state: st.session_state.DV=None
    if "NUMVARS" not in st.session_state: st.session_state.NUMVARS=None
    if "NUMVARS_N" not in st.session_state: st.session_state.NUMVARS_N=None
    if "NUMVARS_C" not in st.session_state: st.session_state.NUMVARS_C=None
    if "VLIST" not in st.session_state: st.session_state.VLIST=None
    if "VLIST_N" not in st.session_state: st.session_state.VLIST_N=None
    if "VLIST_C" not in st.session_state: st.session_state.VLIST_C=None
    if "VLISTF_N" not in st.session_state: st.session_state.VLISTF_N=None
    if "VLISTF_C" not in st.session_state: st.session_state.VLISTF_C=None
    if "DEV" not in st.session_state: st.session_state.DEV=None
    if "OOT" not in st.session_state: st.session_state.OOT=None
    if "N_DEV" not in st.session_state: st.session_state.N_DEV=None
    if "NG_DEV" not in st.session_state: st.session_state.NG_DEV=None
    if "NB_DEV" not in st.session_state: st.session_state.NB_DEV=None
    if "N_OOT" not in st.session_state: st.session_state.N_OOT=None
    if "NG_OOT" not in st.session_state: st.session_state.NG_OOT=None
    if "NB_OOT" not in st.session_state: st.session_state.NB_OOT=None
    if "SUMMARY_N" not in st.session_state: st.session_state.SUMMARY_N=None
    if "SUMMARY_C" not in st.session_state: st.session_state.SUMMARY_C=None
    if "NLEVELS_TBL" not in st.session_state: st.session_state.NLEVELS_TBL=None
    if "NLEVELS_C_TBL" not in st.session_state: st.session_state.NLEVELS_C_TBL=None
    if "VLIST_FAIL" not in st.session_state: st.session_state.VLIST_FAIL=None
    if "RANKPLOT_C_TBL" not in st.session_state: st.session_state.RANKPLOT_C_TBL=None    
    if "RANKPLOTMAP_TBL" not in st.session_state: st.session_state.RANKPLOTMAP_TBL=None
    if "N_WOE" not in st.session_state: st.session_state.N_WOE=None
    if "IV_N" not in st.session_state: st.session_state.IV_N=None
    if "C_WOE" not in st.session_state: st.session_state.C_WOE=None
    if "IV_C" not in st.session_state: st.session_state.IV_C=None
    if "wDEV" not in st.session_state: st.session_state.wDEV=None
    if "wOOT" not in st.session_state: st.session_state.wOOT=None
    if "PSI" not in st.session_state: st.session_state.PSI=None
    if "CORRMAT" not in st.session_state: st.session_state.CORRMAT=None
    if "CORRIV" not in st.session_state: st.session_state.CORRIV=None
    if "CLUSTER" not in st.session_state: st.session_state.CLUSTER=None
    if "ALT_MDLS" not in st.session_state: st.session_state.ALT_MDLS=None
    if "LR_RESULTS" not in st.session_state: st.session_state.LR_RESULTS=None
    if "IMAP" not in st.session_state: st.session_state.IMAP=None
    if "iLR_RESULTS" not in st.session_state: st.session_state.iLR_RESULTS=None
    if "DT_RESULTS" not in st.session_state: st.session_state.DT_RESULTS=None
    if "RF_RESULTS" not in st.session_state: st.session_state.RF_RESULTS=None
    if "GBM_RESULTS" not in st.session_state: st.session_state.GBM_RESULTS=None
    if "XGB_RESULTS" not in st.session_state: st.session_state.XGB_RESULTS=None
    if "MLP_RESULTS" not in st.session_state: st.session_state.MLP_RESULTS=None
    if "LGB_RESULTS" not in st.session_state: st.session_state.LGB_RESULTS=None


    
if "AppState" not in st.session_state: st.session_state.AppState="Initial"
def setAppState(x): st.session_state.AppState = x
#if st.session_state.AppState == 0: st.button('Step 1', on_click=setAppState, args=[1])
#if st.session_state.AppState == 1: st.button('Step 2', on_click=setAppState, args=[2])
if st.session_state.AppState=="Initial": Initialize()

st.set_page_config(layout="wide")

st.header("OMNI STATION", divider='grey')
st.sidebar.header("Controller")

Apptab1, Apptab2, Apptab3, Apptab4, Apptab5, Apptab6, Apptab7, Apptab8 = st.tabs(
    ["General Summary","EDA","Corr & Cluster","Oracle","Charts","Widgts",
     "StateStatus","Debug"])

display_Sidebar()

earlyappstates = ["Initial","DS_Selected","DV_Selected"]

with Apptab1:
    if st.session_state.AppState=="DS_SPLITTER_Selected": 
        setup_and_summary()
        setAppState("Setup_Complete")
    if st.session_state.AppState not in earlyappstates: 
        setup_and_summary_layout()
        
with Apptab2:
    if st.session_state.AppState=="Setup_Complete":
        Numer_Only_Compute()
        NonNumer_Only_Compute()
        nWoE()
        cWoE()
        create_wDEVOOT()
        PSI()
        setAppState("EDA_Complete")
    if st.session_state.AppState not in earlyappstates: 
        Numer_Only_Layout()
        st.write(' ')
        NonNumer_Only_Layout()
        col1, col2 = st.columns(2)
        with col1:
            c = st.container(border=True)
            c.info('The following variables have been dropped from further analysis', 
                    icon="ℹ️")
            c.write(st.session_state.VLIST_FAIL)
        with col2:
            c = st.container(border=True)
            c.markdown('''PSI - DEV/OOT''')
            c.write(st.session_state.PSI)

with Apptab3:
    if st.session_state.AppState=="EDA_Complete":
        CORR()
        Cluster()
        setAppState("Cluster_Complete")
    if st.session_state.AppState not in earlyappstates:
        c = st.container(border=True)
        with c.expander("Correlation Matrix", expanded=False):
            st.write(st.session_state.CORRMAT)
        col1, col2 = c.columns(2)
        with col1:
            st.markdown('''Spearman Correlation''')
            st.write(st.session_state.CORRIV)
        with col2:
            st.markdown('''Cluster Table''')
            st.write(st.session_state.CLUSTER)           

with Apptab4:
    if st.session_state.AppState=="Cluster_Complete":
        Oracle()
        setAppState("Oracle_Complete")
    if st.session_state.AppState not in earlyappstates:
        update_mdls = 0
        def update_altmdls(): 
            global update_mdls; update_mdls = 1
        st.session_state.ALT_MDLS = st.multiselect(
            "Alternate Base Model Types", 
            ["D-Tree", "Random Forest", "GBM", "XGB", "MLP-NN", "LGB"], 
            default=None, key="AltMdlSelect", placeholder="Choose an option",
            on_change=update_altmdls())
        if update_mdls == 1: Oracle(); update_mdls = 0
        Oracle_Layout()

#with Apptab5:
#    display_SessionStatus()
#    display_Text_Examples()
#    display_ProgressBar_Examples()
#    display_StatusMessaging()
#    display_DF()
#    display_Charts()
#with Apptab6:
#    display_Widgets()
        
with Apptab7:
    st.write(st.session_state)

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    df = pd.DataFrame(dict(data=[2, 4, 1, 5, 9, 6, 0, 7]))
    fig, ax = plt.subplots()
    df['data'].plot(kind='bar', color='red')
    df['data'].plot(kind='line', marker='*', color='black', ms=10)
    st.pyplot(fig)

    p2 = figure(title='simple circle example', 
                x_axis_label='x', y_axis_label='y', plot_width = 400, plot_height = 400) 
    p2.circle([1, 2, 3, 4, 5], [4, 7, 1, 6, 3], size=10, color="navy", alpha=0.5) 
    st.bokeh_chart(p2, use_container_width=True)

    p3 = figure(title='simple line example', 
                x_axis_label='x', y_axis_label='y', plot_width = 400, plot_height = 400) 
    p3.line([1, 2, 3, 4, 5], [3, 1, 2, 6, 5],  legend_label='Trend',
            line_width = 2, color = "green") 
    st.bokeh_chart(p3, use_container_width=True)

    p3a = figure(title='simple scatter example', 
                x_axis_label='x', y_axis_label='y', plot_width = 400, plot_height = 400) 
    p3a.scatter([5,3,8,6,4], [3, 1, 2, 6, 5],  legend_label='Trend',
            color = "red") 
    st.bokeh_chart(p3a, use_container_width=True)

    p4 = figure(title='simple vbar example', 
                x_axis_label='x', y_axis_label='y', plot_width = 300, plot_height = 300) 
    p4.vbar([1, 2, 3, 4, 5], top=[3, 1, 2, 6, 5], width=.5, color = "blue") 
    st.bokeh_chart(p4, use_container_width=True)

    p5 = figure(title='line + bar example', 
                x_axis_label='x', y_axis_label='y', plot_width = 300, plot_height = 300) 
    p5.vbar([1, 2, 3, 4, 5], top=[3, 1, 2, 6, 5], width=.5, color = "blue") 
    p5.line([1, 2, 3, 4, 5], [3, 1, 2, 6, 5],  line_width = 2, color = "red") 
    st.bokeh_chart(p5, use_container_width=True)

    x_column = "x"; y_column1 = "y1"; y_column2 = "y2"
    df = pd.DataFrame()
    df[x_column] = range(0, 100)
    df[y_column1] = np.linspace(100, 1000, 100)
    df[y_column2] = np.linspace(1, 2, 100)
    p = figure()
    p.line(df[x_column], df[y_column1], legend_label=y_column1, line_width=1, color="blue")
    p.extra_y_ranges['Second Axis'] = Range1d(start=1, end=2)
    p.add_layout(LinearAxis(y_range_name='Second Axis'), "right")
    p.line(df[x_column], df[y_column2], legend_label=y_column2, line_width=1, 
           y_range_name="Second Axis", color="green")
    st.bokeh_chart(p, use_container_width=True)

with Apptab8:
    if st.session_state.DEBUGON == True and st.session_state.AppState != "Initial": 
        st.write("DS" , st.session_state.DS)
        st.write("CP[]" , st.session_state.CP1, st.session_state.CP2)
        st.write("VALID_DV" , st.session_state.VALID_DV)    
        st.write("DS_SPLITTER" , st.session_state.DS_SPLITTER)
        st.write("DV" , st.session_state.DV)
        st.write("NUMVARS N C" , st.session_state.NUMVARS, 
                 st.session_state.NUMVARS_N, st.session_state.NUMVARS_C)
        st.write("VLIST N C" , st.session_state.VLIST, 
                 st.session_state.VLIST_N, st.session_state.VLIST_C)    
        st.write("DEV" , st.session_state.DEV)
        st.write("OOT" , st.session_state.OOT)
        st.write("N_DEV NG NB" , st.session_state.N_DEV, 
                 st.session_state.NG_DEV, st.session_state.NB_DEV)
        st.write("N_OOT NG NB" , st.session_state.N_OOT, 
                 st.session_state.NG_OOT, st.session_state.NB_OOT)
        st.write("SUMMARY_N" , st.session_state.SUMMARY_N)
        st.write("SUMMARY_C" , st.session_state.SUMMARY_C)
        st.write("NLEVELS_TBL" , st.session_state.NLEVELS_TBL)
        st.write("NLEVELS_C_TBL" , st.session_state.NLEVELS_C_TBL)
        st.write("VLIST_FAIL" , st.session_state.VLIST_FAIL)
        st.write("RANKPLOT_C_TBL" , st.session_state.RANKPLOT_C_TBL)
        #st.write("RANKPLOTMAP_TBL" , st.session_state.RANKPLOTMAP_TBL)
        st.write("VLISTF_N VLISTF_C" , 
                 st.session_state.VLISTF_N, st.session_state.VLISTF_C)
        st.write("IV_N" , st.session_state.IV_N)
        st.write("IV_C" , st.session_state.IV_C)
        st.write("C_WOE" , st.session_state.C_WOE)
        st.write("N_WOE" , st.session_state.N_WOE)
        st.write("wDEV" , st.session_state.wDEV)
        st.write("wOOT" , st.session_state.wOOT)
        st.write("PSI" , st.session_state.PSI)
        st.write("CORRMAT" , st.session_state.CORRMAT)
        st.write("CORRIV" , st.session_state.CORRIV)
        st.write("CLUSTER" , st.session_state.CLUSTER)
        st.write("ALT_MDLS" , st.session_state.ALT_MDLS)
        #st.write("LR_RESULTS" , st.session_state.LR_RESULTS)
        #st.write("iLR_RESULTS" , st.session_state.iLR_RESULTS)   
        st.write("IMAP" , st.session_state.IMAP)
        #st.write("DT_RESULTS" , st.session_state.DT_RESULTS)
        #st.write("RF_RESULTS" , st.session_state.RF_RESULTS)
        #st.write("GBM_RESULTS" , st.session_state.GBM_RESULTS)
        #st.write("XGB_RESULTS" , st.session_state.XGB_RESULTS)
        #st.write("MLP_RESULTS" , st.session_state.MLP_RESULTS)
        #st.write("LGB_RESULTS" , st.session_state.LGB_RESULTS)


# TODO
# Deploy
# Add Interaction func to create top interac vars
# Consider HalvingRandomSearchCV
#y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

#params = {
#    "colsample_bytree": uniform(0.7, 0.3),
#    "gamma": uniform(0, 0.5),
#    "learning_rate": uniform(0.03, 0.3), # default 0.1 
#    "max_depth": randint(2, 6), # default 3
#    "n_estimators": randint(100, 150), # default 100
#    "subsample": uniform(0.6, 0.4) }
#search = RandomizedSearchCV(xgb_model, param_distributions=params, 
#        random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=1, 
#        return_train_score=True)


# ds['spam'] = ds['yesno'].map({'y': 1, 'n': 0})


# name = st.text_input('Name', placeholder="Input name to continue")
# if not name: st.stop()


# DB
#conn = st.connection("my_database")
#df = conn.query("select * from my_table")
#st.dataframe(df)
