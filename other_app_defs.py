# cd C:\temp\vscode\temp_proj
# python -m venv .venv
# .venv\Scripts\activate.bat

# pip install streamlit
# python -m streamlit hello
# python -m streamlit run app.py OR
# python -m streamlit run ./app.py
# Cntl+C to stop

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, os
import altair as alt
from vega_datasets import data
from bokeh.plotting import figure
from functions import *


def display_Text_Examples():
    with st.expander("Using the expander"):
        st.title("Hello _Jon_ :blue[world]") 
        st.header("A header", divider='grey') #red,blue,green,orange,grey,rainbow
        st.subheader("A subheader", divider='green')
        st.caption("A caption")
        st.code("A code block, x=123")
        st.latex(r''' a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} = 
                 \sum_{k=0}^{n-1} ar^k = a \left(\frac{1-r^{n}}{1-r}\right) ''')
        st.text("A text")
        st.text_area("Outside textarea", "Inside textarea", key="txtarea0")
        st.markdown("Markdown: *1star italic*, **2star bold**, ***3star bold italic***")
        st.markdown(''':red[Streamlit] :orange[can] :green[write] :blue[text] 
                    :violet[in] :gray[pretty] :rainbow[colors].''')
        multi = '''If you end a line with two spaces,  
        a soft return is used for the next line.
    
        Two (or more) newline characters in a row will result in a hard return'''
        st.markdown(multi)



def display_ProgressBar_Examples():
    progress_msg = st.empty()
    bar = st.progress(0)
    for i in range(1,101):
        time.sleep(0.005) # execute func
        bar.progress(i) 
        progress_msg.text(f'Pct Complete {i}/100') 
        # OR progress_msg.text("%i%% Complete" % i)
    bar.empty()
    progress_msg.success('Done!', icon="✅")
    # st.balloons()
    # st.snow()



##################################################################################
##################################################################################
##################################################################################
def display_StatusMessaging():    
    add_status = st.status("Testing", expanded=False)
    time.sleep(.1)
    add_status.update(label="1: EDA")
    time.sleep(.1)
    add_status.update(label="2: Variable Selection")
    time.sleep(.1)
    add_status.update(label="3: Clustering")
    time.sleep(.1)
    add_status.update(label="Processing complete!", state="complete")

    c = st.container(border=True)
    c.info('This is a purely informational message', icon="ℹ️")
    c.success('This is a success message!', icon="✅")
    c.warning('This is a warning', icon="⚠️")
    c.error('This is an error', icon="🚨")
    e = RuntimeError('This is an exception of type RuntimeError')
    c.exception(e)



##################################################################################
##################################################################################
##################################################################################
def display_SessionStatus():
    # Use session state based randomization to not be refreshed
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(np.random.randn(20,2),columns=["x","y"])
    c = st.container(border=True)
    c.caption("Choose a datapoint color")
    color = c.color_picker("Color", "#FF0000", label_visibility='hidden', key="cpick1")
    st.write('The current color is', color)   
    c.scatter_chart(st.session_state.df, x="x", y="y", color=color)



##################################################################################
##################################################################################
##################################################################################
def display_DF():
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["DF: df & write & table","highlights","cache & stretchWidth",
         "data_editor (DE)","DE w Cntls on Params","Local csv","online csv"])
    
    df1 = np.random.randn(3, 5)
    df1b = pd.DataFrame({'col1': [1,2,3,4], 'col2': [10,20,30,40]})
    with tab1:
        st.caption("randn 3x5 DF - using dataframe(df)")
        st.dataframe(df1)
        st.caption("randn 3x5 DF - using write(df)")
        st.write(df1)
        st.caption("randn 3x5 DF - using table(df)")
        st.table(df1)

    df2 = pd.DataFrame(np.random.randn(5,10),
        columns=('col %d' % i for i in range(10)))
    with tab2:
        st.caption("randn 5x10 DF - max of col highlighted, using dataframe()")
        st.dataframe(df2.style.highlight_max(axis=0))

    with tab3:
        c = st.container(border=True)
        c.caption("defined 2x4 DF, checkbox controls width")
        c.checkbox("Use container width", value=False, key="use_container_width")
        c.dataframe(df1b, use_container_width=st.session_state.use_container_width)

    df4 = pd.DataFrame([
        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        {"command": "st.balloons", "rating": 5, "is_widget": False},
        {"command": "st.time_input", "rating": 3, "is_widget": True},])
    tab4.caption("Using data_editor on DF")
    edited_df = tab4.data_editor(df4, key="deditor1")
    selection = edited_df.loc[edited_df["rating"].idxmax()]["command"]
    tab4.markdown(f"Command with max rating is **{selection}**")

    param_df = pd.DataFrame({
        "parameter": ["iv_corr", "dv_corr", "rsqr", "gini"],
        "value": [.4,.7,.7,.5],})
    with tab5:
        c = st.container(border=True)
        c.caption("Using data_editor on DF again with range controls")
        c.data_editor(param_df,
            column_config={
                "parameter": st.column_config.TextColumn(disabled=True),
                "value": st.column_config.NumberColumn(
                    min_value=0, max_value=1, format="%f",)
                    }, hide_index=True, key="deditor2")

    with tab6:
        with st.form("form1"):
            st.write("Pull data from local dir after submit")
            submitted = st.form_submit_button("Submit")
            if submitted: st.write("Submitted")

    URL1 = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
    with tab7:
        if st.checkbox('Show raw data', key="cb1"):
            data_load_state = st.text('Loading data...')
            testdata = load_data_from_web2(URL1,1000)
            data_load_state.text("Done! (using st.cache_data)")
            st.subheader('Raw data')
            st.write(testdata)


##################################################################################
##################################################################################
##################################################################################
def display_Charts():
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Line & Dynamic","Bar & Expand ","Scatter","Hist - numpy & pyplot",
         "Altair","Bokeh"])
    tab1.caption("line chart")
    xy = pd.DataFrame(np.random.randn(20,3), columns=['a','b','c'])
    tab1.line_chart(xy) # OR with.tab1: st.line_chart(xy)
    with tab1:
        st.caption("dynamic chart")
        status_text = st.empty()
        last_rows = np.random.randn(1,1)
        chart = st.line_chart(last_rows)
        for i in range(1, 101):
            new_rows = last_rows[-1,:] + np.random.randn(5,1).cumsum(axis=0)
            status_text.text("%i%% Complete" % i)
            chart.add_rows(new_rows)
            last_rows = new_rows
            time.sleep(0.005)
        st.button("Re-run", key="btn1")

    xy2 = pd.DataFrame(
        {"col1": list(range(20)), 
        "col2": np.random.random(20), 
        "col3": np.random.random(20)})
    with tab2:
        st.caption("bar chart")
        st.bar_chart(xy2, x="col1", y="col2")
        with st.expander("bar chart - with expander"):
            st.bar_chart(xy2, x="col1", y=["col2","col3"], color=["#FF0000","#0000FF"])
        
    xy3 = pd.DataFrame(np.random.randn(20,3), columns=["a","b","c"])
    with tab3:
        st.caption("scatter chart")
        st.scatter_chart(xy3)

    with tab4:
        st.caption("Histogram using numpy")
        slide_select = st.slider('bin', 2, 20, 10, key="slider1")
        randvec = np.random.normal(1,1,size=500)
        hist_values = np.histogram(randvec, bins=slide_select)[0]
        st.bar_chart(hist_values)
        st.caption("Histogram using pyplot")
        fig, ax = plt.subplots()
        ax.hist(randvec, bins=slide_select)
        st.pyplot(fig) 

    tab5.caption("using altair")
    source = data.cars()
    chart = alt.Chart(source).mark_circle().encode(
        x='Horsepower', y='Miles_per_Gallon', 
        color='Origin',).interactive()
    tab5.altair_chart(chart, theme="streamlit", use_container_width=True)
    # OR tab5.altair_chart(chart, theme=None, use_container_width=True)

    tab6.caption("using bokeh")
    x = [1, 2, 3, 4, 5]; y = [6, 7, 2, 4, 5]
    p = figure(title='simple line example',
        x_axis_label='x', y_axis_label='y')
    p.line(x, y, legend_label='Trend', line_width=2)
    tab6.bokeh_chart(p, use_container_width=True)



##################################################################################
##################################################################################
##################################################################################
def display_Widgets():
    # Regular button
    if 'clicked_btn1' not in st.session_state: st.session_state.clicked_btn1 = False
    if 'disable_btn1' not in st.session_state: st.session_state.disable_btn1 = False
    if 'txtarea1' not in st.session_state: st.session_state.txtarea1 = "Not clicked Btn1"
    st.text_area("txtarea1", " ", key="txtarea1")
    def click_btn1(x,y,z): 
        st.session_state.clicked_btn1 = x
        st.session_state.disable_btn1 = y
        st.session_state.txtarea1 = z
    Btn1 = st.button("Btn1", on_click=click_btn1, 
                     disabled=st.session_state.disable_btn1, 
                     args=[True,True,"Clicked Btn1"], key="btn3")
    if st.session_state.clicked_btn1:
        st.caption(f"Btn1 SessionState: {st.session_state.clicked_btn1}")
    st.divider()
    # upload button
    uploaded_file = st.file_uploader("Choose a file", key="fileuploader1")
    if uploaded_file is not None: dataframe=pd.read_csv(uploaded_file)
    st.divider()
    # text input
    st.text_input("Your name", key="name")
    st.write(st.session_state.name)
    st.divider()
    # checkbox
    checked = st.checkbox('Show dataframe', key="cb2")
    if checked: 
        param_df = pd.DataFrame({
            "parameter": ["iv_corr", "dv_corr", "rsqr", "gini"],
            "value": [.4,.7,.7,.5],})
        st.dataframe(param_df)
    st.divider()
    # toggle
    on = st.toggle('Activate feature', key="toggle1")
    if on: st.write('Feature activated!')
    st.divider()    
    # slider
    x = st.slider('value_name', key="slider2") 
    st.write(x, 'squared is', x * x)
    st.divider()
    # selectbox
    df2 = pd.DataFrame({'col1': [1,2,3,4], 'col2': [10,20,30,40]})
    option = st.selectbox('Chooose num', df2['col1'], key="selectbox1")
    st.write('You selected: ', option)
    st.divider()
    # radio
    yn = st.radio("Choose Y/N", ("Yes","No"), key="radio1")
    st.write(f"You selected {yn}.")
    # Double column
    # col1, col2 = st.columns(2), 50/50 size cols
    col1, col2 = st.columns([2, 3]) # 2/3 ratio cols
    with col1:
        if "disable_key" not in st.session_state: st.session_state.disabled = False
        c = st.container(border=True)
        c.checkbox("Disable radio widget", key="disabled")
        c.radio("_",("A","B"), key="disable_key",
            disabled=st.session_state.disabled, horizontal=True)
    with col2:
        c2 = st.container(border=True)
        chosen = c2.radio('Sorting hat',
            ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"), key="radio2")
        c2.write(f"You are in {chosen} house!")
        if chosen=="Gryffindor": c2.write("yay")
        else: c2.write("not yay")

