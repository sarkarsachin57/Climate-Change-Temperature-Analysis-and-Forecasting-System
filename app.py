# Import Libraries

import pandas as pd # for data manipulation
import numpy as np # For numerical ops
import streamlit as st # For Web development
import plotly.express as px # For Visualization
from neuralprophet import NeuralProphet # For forecasting
import datetime, json # For some other ops
from statsmodels.tsa.stattools import acf # For acf plotting



# Loading and initializing all datasets


df_global = pd.read_csv('datasets/GlobalTemperatures.csv',parse_dates=['dt'],index_col='dt')
df_global = df_global[['LandAverageTemperature']]
df_global.columns = ['AverageTemperature']

df_countries = pd.read_csv('datasets/GlobalLandTemperaturesByCountry.csv',parse_dates=['dt'],index_col='dt')
df_countries = df_countries[['AverageTemperature','Country']]

df_states = pd.read_csv('datasets/GlobalLandTemperaturesByState.csv',parse_dates=['dt'],index_col='dt')
df_states = df_states[['Country','State','AverageTemperature']]

month = {1:'Janauary',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',
         8:'August',9:'September',10:'October',11:'November',12:'December'}




# This function will take input on Country and State using a Dropdown selectbox from the user and return data on that particular selected country and state.


def choose_location():
    country = st.selectbox('Select Country :',['All'] + list(df_countries.Country.unique()))
      
    state = st.selectbox('Select State :',['All'] + list(df_states[df_states['Country']==country]['State'].unique()))
    if country == 'All':
        df = df_global
    elif state == 'All':
        df = df_countries[df_countries['Country']==country][['AverageTemperature']]
    else :
        df = df_states[df_states['Country']==country]
        df = df[df['State']==state][['AverageTemperature']]
    
    return df, country, state



# This Function will impute missing Values with the monthly mean temperatures


def missing_value_imputer(df):
    monthly_mean = df['AverageTemperature'].groupby(df.index.month).mean().to_dict()
    df['monthly_mean'] = df.index.month.map(monthly_mean)
    df['AverageTemperature'].fillna(df['monthly_mean'],inplace=True)
    return df[['AverageTemperature']]




# This Function create 2 slider input on frontend to take lower year and upper year from user and filter data based on that year range and return it.


def time_ranger(df):
    
    min_value = int(df.index.year[0])
    max_value = int(df.index.year[-1])
    
    lower_year = st.slider('Lower Year Range :',min_value=min_value,max_value=max_value,value=min_value)
    upper_year = st.slider('Upper Year Range :',min_value=min_value,max_value=max_value,value=max_value)
    df = df.loc[str(lower_year):str(upper_year)]
    
    return df, lower_year, upper_year





# This function will show the data and its statistical description on the frontend as well as a download button to download the data as csv


def show_data(df,country,state,lower_year,upper_year):
    
    str_show = '##### Average Temperature Per Month Data of Country : ' + country + ' and State : ' + state + ' from ' + str(lower_year) + ' to ' + str(upper_year)
    st.markdown(str_show,unsafe_allow_html=True)
    
    df['year'] = df.index.year
    df['month'] = df.index.month.map(month)
    df['Month'] = df['year'].astype(str) + ' , ' + df['month'] 
    
    a,b = st.columns([5.5,5])
    a.markdown('**Average Temperatures Data :**')
    b.markdown('**Statistical Description :**')
    a,b = st.columns([5.5,5])
    a.dataframe(df.set_index('Month')[['AverageTemperature']])
    df = df[['AverageTemperature']]
    b.dataframe(df.describe())
    file = 'monthly_aver_temp_' + country + '_' + state + '_' + str(lower_year) + '_to_' + str(upper_year) + '.csv'
    st.download_button('Download this Data', df.to_csv(),file_name=(file))





# This function will used to plot an interactive linear chart to visualize the trend based on the rolling window selected. 
# The rolling window selected from a slider input on frontend by user and based on the selected value, it will apply rolling mean or moving averages on the given data and plot it.  


def trend_plot(df,country,state,lower_year,upper_year):
    
    text = '''### Trend Plot :
        A trend Graph is a graph that is used to show the trends data over
        a period of time. It describes a functional representation of 
        two variables (x , y). In which the x is the time-dependent 
        variable whereas y is the collected data.
        
        Moving Averages : In statistics, a moving average is a calculation 
        to analyze data points by creating a series of averages of different 
        subsets of the full data set. It is also called a moving mean or 
        rolling mean and is a type of finite impulse response filter. '''
    
    st.markdown(text)
    
    win = st.slider('Rolling Window [ Rolling Window of 1 is Equal to Real Data ] :',min_value=1,max_value=20*12,value=1)
    st.warning('Note : The below Trend Plot depend on the above rolling window. Set rolling window = 1, to get plot on Real Data as per above. Set rolling window > 1 [ 60, 120, 180, 240 Recommended ] to view the trend.')

    str_show='##### Trend Plot on Average Temperature per Month with Moving Average of ' + str(win) + ' Months of Country : ' + country + ' and State : ' + state + ' from ' + str(lower_year) + ' to ' + str(upper_year)
    st.markdown(str_show,unsafe_allow_html=True)
    
    data = df['AverageTemperature'].rolling(window=win).mean()
    fig = px.line(data,title='Trend Plot', width=750, height=550,labels={'dt':'Date','ds':'Date','value':'Average Temperature'})
    fig.update_layout(title={'font_size':25,'x':0.5})
    st.plotly_chart(fig)





# This function will used to plot an interactive bar chart to visualize the seasonality or monthly mean temperature from january to december on the given data. 


def seasonal_plot(df,country,state,lower_year,upper_year):
    text = '''### Seasonal Bar Plot :
        
        What Is Seasonality?
        Seasonality is a characteristic of a time series in which the data
        experiences regular and predictable changes that recur every 
        calendar year. Any predictable fluctuation or pattern that recurs 
        or repeats over a one-year period is said to be seasonal. 
        
        A Seasonal Bar Plot is a bar chart similar to a time plot except 
        that the data are plotted against the individual “seasons” or
        "months" in which the data were observed. This plot visualize 
        the seasonality of the given Time Series.
''' 
    st.markdown(text)
    
    str_show='##### Seasonal Bar Plot on Average Temperature per Month of Country : ' + country + ' and State : ' + state + ' from ' + str(lower_year) + ' to ' + str(upper_year)
    st.markdown(str_show,unsafe_allow_html=True)
   
    temp = df['AverageTemperature'].groupby(df['AverageTemperature'].index.month).mean()
    temp.index = temp.index.map(month)
    fig = px.bar(temp,color=temp,title='Seasonal Bar Plot', width=750, height=550, labels={'dt':'Date','ds':'Date','value':'Average Temperature'})
    fig.update_layout(title={'font_size':25,'x':0.5})
    st.plotly_chart(fig)





# This function will used to plot an interactive linear chart to visualize autocorrelation upto 100 lags on the given data 


def autocorrelation_plot(df,country,state,lower_year,upper_year):
    text = '''### ACF or Autocorrelation plots : 

    Autocorrelation plots are a commonly used tool for checking randomness
    in a data set. This randomness is ascertained by computing 
    autocorrelations for data values at varying time lags. 
    It measures a set of current values against a set of past values and 
    finds whether they correlate.
    It is the correlation of one-time series data to another time series
    data which has a time lag.
    It varies from +1 to -1.
    An autocorrelation of +1 indicates that if time series one increases in
    value the time series 2 also increases in proportion to the change in 
    time series 1.
    An autocorrelation of -1 indicates that if time series one increases in
    value the time series 2 decreases in proportion to the change 
    in time series 1.'''
    
    st.markdown(text)
    str_show='##### Autocorrelation Plot on Average Temperature per Month of Country : ' + country + ' and State : ' + state + ' from ' + str(lower_year) + ' to ' + str(upper_year)
    st.markdown(str_show,unsafe_allow_html=True)
    
    acf_df = pd.DataFrame({'ACF':acf(df['AverageTemperature'],nlags=100,fft=False)})
    acf_df['Upper Level'] = 1.96 / 1.96 / (df.shape[0] ** 0.5 )
    acf_df['Lower Level'] = - 1.96 / 1.96 / (df.shape[0] ** 0.5 )
    fig = px.line(acf_df,title='Autocorrelation Plot',labels={'index':'Lags','value':'Correlation'})
    fig.update_layout(title={'font_size':25,'x':0.5})
    st.plotly_chart(fig)
    




# This function will train the neural prophet model on time the based on the selected historical data and return forcast for next 20 years. This will may take 10 to 15 seconds while training.


def training_and_forecasting(df):
    
    df = df.reset_index()
    df.columns = ['ds','y']
    

    prophet = NeuralProphet()
    
    st.info('Training. It takes only 10 to 15 seconds. Please Wait....')
    progress = st.progress(0)
    progress.progress(10)
    prophet.fit(df,freq='1m',epochs=10)
    
    progress.progress(100)
    st.success('Training Completed. See results..')
    
    today = str(datetime.datetime.today()).split()[0]
    dates = pd.date_range(today, periods=12*20, freq='1m')
    dates_df = pd.DataFrame({'ds':dates})
    dates_df['y'] = np.nan
    
    def add_dates(x):
        return x + datetime.timedelta(days=1)

    dates_df['ds'] = dates_df['ds'].apply(add_dates)
    dates_pred = prophet.predict(dates_df)
    df = dates_pred.set_index('ds')[['yhat1']]
    df.columns = ['AverageTemperature']
    
    return df





# This function will show user an abstract about the application.


def show_doc():
    
    st.header('Welcome to this Global Temperature Climate Change Analysis and Forecasting Application')
    st.markdown('<br>',True)
    def usa(x):
        if x == 'United States':
            return 'United States of America'
        else:
            return x
    
    df_countries['Country'] = df_countries['Country'].apply(usa)
    geo = json.load(open('datasets/countries.geo.json','r'))
    country_id = {}
    for feature in geo['features']:
        country_id[feature['properties']['name']] = feature['id']
    df_countries['country_id'] = df_countries.Country.map(country_id)
    
    locations = df_countries.groupby('country_id')['AverageTemperature'].mean().to_dict().keys()
    data = df_countries.groupby('country_id')['AverageTemperature'].mean().to_dict().values()
    data = np.around(list(data),2)
    hover_name = df_countries.groupby('country_id')['Country'].unique()
    hover_data = {'Average Temperature':data}
    title = 'Chloropleth Map of Average Temperature by Countries'
    fig = px.choropleth_mapbox(geojson=geo,locations=locations,color=data,hover_name=hover_name,hover_data=hover_data,
                           mapbox_style='carto-positron',zoom=0.35,opacity=0.5,title=title)
    fig.update_layout(title={'font_family':'Georgia','font_size':23,'x':0.5})
    st.plotly_chart(fig)
    
    
    
    text = '''
      
Climate change is undoubtedly one of the biggest problems in the 21st century. Artificial Intelligence methods have recently contributed in the advancement of accurate prediction tools for the estimation and assessment of extreme environmental events and investigation of the climate change time series. The recent advancement in Artificial Intelligence including the novel machine learning and deep learning algorithms as well as soft computing applications have greatly empowered prediction methods. Through this project, we have explore, analyze the global Climatic trend and pattern on temperature component and forecast the future temperature trends using a state of art time series deep learning model. After the research, exploration and analysis on the historical data and modelling, we build and deploy this end to end web solution on the frontend to view and explore historical data as well as future forecasts generated through the deep learning model.
<br><br>
##### <center> Get Historical Data and Analysis </center>

>Select **"Historical Data and Plotting"** from the menu inside the sidebar. Then you will - <br><br>
> 1) Get Historical Average Temperatures Per Month upto past 200 years as per the choice of Country, State and Time Range selected.<br><br>
> 2) Able to download the same data as filtered. <br><br>
> 3) View the interactive plots to analyze and conclude the Historical Temperature trend, Seasonality and Autocorrelation on the filtered Country, State and Time Range. <br><br>

##### <center> Get Future Forecasts and Analysis </center>

>Select **"Future Data and Plotting"** from the menu inside the sidebar. Then you will - <br><br>
> 1) Get Predicted Future Average Temperatures Per Month upto next 20 years as per the choice of Country, State and Time Range selected. <br><br>
> 2) Able to download the same data as filtered. <br><br>
> 3) View the interactive plots to analyze and conclude the Predicted Future Temperature trend, Seasonality and Autocorrelation on the filtered Country, State and Time Range. <br><br>

##### <center> Want to send Message or Feedback Us </center>
<center>Select <strong>"Feedback Us"</strong> from the menu inside the sidebar.</center> <br><br>

##### <center> Want to known about Us </center>
<center>Select <strong>"About Us"</strong> from the menu inside the sidebar.</center> <br><br>

'''
    st.markdown(text,unsafe_allow_html=True)
    





# This will create a page where user can view Historical data, description, plots based on selected country, state and time range


def show_historical():
    
    st.header('Historical Average Temperature Per Month Data, Description and Visualization')
    
    df, country, state = choose_location()
    
    df = missing_value_imputer(df)
    
    df, lower_year, upper_year = time_ranger(df)
    
    show_data(df,country,state,lower_year,upper_year)
    
    trend_plot(df,country,state,lower_year,upper_year)
   
    seasonal_plot(df,country,state,lower_year,upper_year)
    
    autocorrelation_plot(df,country,state,lower_year,upper_year)
    





# This will create a page where user can view future data, description, plots based on selected country, state and time range    
    
    
def show_future():
    
    st.header('Future Average Temperature Per Month Data, Description and Visualization')
    
    df, country, state = choose_location()
    df = missing_value_imputer(df)
    
    df = training_and_forecasting(df)
    
    df, lower_year, upper_year = time_ranger(df) 
    
    
    show_data(df,country,state,lower_year,upper_year)
    
    trend_plot(df,country,state,lower_year,upper_year)
   
    seasonal_plot(df,country,state,lower_year,upper_year)
    
    autocorrelation_plot(df,country,state,lower_year,upper_year)
    
    
   
    
    
# This will create another page where user can send feedback or message to the team.
    
 
def show_feedback():
    st.title('Feedback or Message Us :')
    name = st.text_input('Your Name :')
    email = st.text_input('your Email :')
    msg = st.text_area('Feedback or Message :')
    clicked = st.button('SUBMIT')
    if clicked == True:
        sep = '   |   '
        data = [name,sep,email,sep,msg,'\n','--------------------------------------------------------------------------','\n']
        with open('datasets/feedbacks.txt','a',encoding="utf-8") as f:
            f.writelines(data)
            f.close()
        st.success('Thanks for your feedback!')
        st.balloons()
        
        
 
        
         
         
# This will used to create button using html and css to hyperlink
        
        
def create_btn(text,link):
         
    btn_css =  '''
    <style>
    a.link-btn {
        color: #fff;
        background: #337ab7;
        display:inline-block;
        border: 1px solid #2e6da4;        
        font: bold 14px Arial, sans-serif;
        text-decoration: none;
        border-radius: 2px;
        padding: 6px 20px;
    }
    a.link-btn:hover {
        background-color: #245582;
        border-color: #1a3e5b;
    }
</style>'''
    
    btn_html = '<a href={} class="link-btn" target="_blank">{}</a>'.format(link,text)

    return btn_css + btn_html
        






 # This will create another page where user can see about iNeuron and the developer.
        
        
def show_about():
         
    st.header('🏢 About iNeuron ')
    
    st.markdown('iNeuron is a product-driven organisation working on state of the art projects for our domestic and international clients carrying a lot of expertise in product development in the area of Computer vision, Deep learning, NLP, Auto ML and Machine learning with industry expertise in warehousing, Security, Surveillance, Healthcare and Inventory management. We also have our training academy where we are providing affordable AI education in Deep learning, Machine Learning and NLP.')
    st.markdown('iNeuron started its journey from being a product development team that caters to domestic as well as international clients, and we continue to develop state of the art products for our prestigious clients all over the world. However, we realised the shortcomings in the field Ai education and realised that data science education was not only very expensive but also lacked practical exposure via live-projects.')
    st.markdown('Our mission is to provide quality education throughout all sets of the economy which is why we did not want the price to be a factor for which individuals would hesitate in attaining education in the respective domains.We believe in growing together which is why we also have our learning community where students can raise discussion related to technical questions and find the solutions to their problems.')

    a,b,c = st.columns([1.53,1.5,7])
    
    a.markdown(create_btn('LinkedIn', 'https://www.linkedin.com/company/ineuron-ai/'),True)
    b.markdown(create_btn('Website', 'https://ineuron.ai/'),True)
    
    st.markdown('<br>',True)
    st.header('👨‍💻 About The Developer')
    
    st.markdown('Hi, I am Sachin Sarkar, Intern at iNeuron Intelligence Pvt. Ltd. and Developer of this Global Temperature Climate Change Time Series Analysis and Forecasting Application. I am a Data Science and AI enthusiast and Practitioner. Currently I am doing Bachelors in Data Science from MAKAUT, WB. I am an active contributor to Kaggle Community with holding the status of "Notebook Expert" with ranking less than 2k out of 200k (approx). From the begining of 2021, I am continuously doing multiple Internships and Projects related to Data Science, Machine Learning, Computer Vision, Natural Language Processing, Time Series, Business Intelligence, EDA etc and learning new things each and every day. As per now ( December 2021 ) I am Well experienced in working with Python and Python libraries like pandas, numpy , matplotlib, seaborn, plotly, sklearn, tensorflow, keras, cv2, nltk, flask, streamlit and many more. I am a quick learner and attentive worker. ')
    a,b,c = st.columns([1.47,1.30,7])
    
    
    a.markdown(create_btn('LinkedIn', 'https://www.linkedin.com/in/sachin-sarkar-aba74420b/'),True)
    b.markdown(create_btn('Github', 'https://github.com/sarkarsachin57'),True)
    c.markdown(create_btn('Kaggle', 'https://www.kaggle.com/sachinsarkar'),True)







# sidebar design 
a,b = st.sidebar.columns([1,8])
b.image('https://d24cdstip7q8pz.cloudfront.net/t/ineuron1/content/common/images/final%20logo.png',width=230,caption='iNeuron Intelligence Pvt Ltd')
st.sidebar.title('Temperature Change Analysis and Forecasting')


# this will create a sidebar dropdown menu to controls page switching.

res = st.sidebar.radio('You Want : ', ['Documentation','Historical Data and Plotting' ,'Future Data and Plotting','Feedback Us','About Us'])

if res == 'Documentation':
    show_doc()
elif res == 'Historical Data and Plotting' :
    show_historical()
elif res == 'Future Data and Plotting' :
    show_future()
elif res == 'Feedback Us' :
    show_feedback()
elif res == 'About Us' :
    show_about()
    
    
# To hide streamlit default menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 




























 
