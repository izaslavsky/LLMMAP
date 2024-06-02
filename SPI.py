from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
import pandas as pd
import streamlit as st
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
from langchain_groq import ChatGroq
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import matplotlib

# Load governorate GeoJSON file
gdf = gpd.read_file('dataset/Jordan Purchasing Power/governorate.geojson')
gdf = gdf.iloc[:, 43:45]

def get_governorate(df, gdf):
    points = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
    df_points = gpd.GeoDataFrame(df, geometry=points)
    merged = gpd.sjoin(df_points, gdf, how='left', op='within')
    return merged

def get_df_code(llm, question):
    prompt = PromptTemplate(
        template="""system
        We have a dataframe df with the following columns:
            Code	
            Station_ID	
            JMD_code	
            Station_Name	
            Altitude_m	
            Latitude	
            Longitude	
            Time	
            SPI

        The following is the request from a user:    
        {question}

        Generate the python code for the request as one statement st.session_state.df = ... only without any explanation.

        Answer:assistant
        """,
        input_variables=["question"],
    )

    df_code_chain = prompt | llm | StrOutputParser()
    return df_code_chain.invoke({"question": question})

def handle_request(question):
        result = get_df_code(llm, question)
        exec(result)

def is_geojson(data):
    # Check if data is a GeoJSON dictionary
    return isinstance(data, dict) and 'type' in data and data['type'] == 'FeatureCollection'

title = "Jordan Standardized Precipitation Index"
st.set_page_config(layout="wide", page_title=title)
st.markdown(f"### {title}")

# Set up LLM
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key="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")

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Create a Kepler map
map1 = KeplerGl(height=400)

config = {
    "version": "v1",
    "config": {
        "mapState": {
            "bearing": 0,
            "latitude": 32.24,
            "longitude": 35.35,
            "pitch": 0,
            "zoom": 6,
        },
        "visState": {
          'layerBlending': "additive",
        }
    },
}
map1.config = config

# Load CSV file
df = pd.read_csv('dataset/SPI/Jordan Standardized Precipitation Index.csv')

# Add governorate information to df
df_with_governorate = get_governorate(df, gdf)

if "df" in st.session_state:
    if isinstance(st.session_state.df, pd.DataFrame) or is_geojson(st.session_state.df) or isinstance(st.session_state.df, str):
        map1.add_data(data=st.session_state.df, name=title)
else:
    map1.add_data(data=df_with_governorate, name=title)

# Set up two columns for the map and chat interface
col1, col2 = st.columns([3, 2])

with col1:
    keplergl_static(map1)

# Set up the chat interface
with col2:
    chat_container = st.container(height=355)

    for message in st.session_state.chat:
        with chat_container:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    user_input = st.chat_input("What can I help you with?")
    if user_input:
        with chat_container:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("We are in the process of your request"):
                    try:
                        handle_request(user_input)
                        response = f"Your request was processed."
                    except:
                        response = "We are not able to process your request. Please refine your request and try again."
                    st.session_state.chat.append({"role": "assistant", "content": response})
                    st.rerun()

if "df" in st.session_state:
    showdf = st.session_state.df
    if isinstance(st.session_state.df, pd.DataFrame):
        st.dataframe(showdf)
    elif isinstance(st.session_state.df, matplotlib.axes.Axes):
        fig = showdf.get_figure()
        st.pyplot(fig)

else:
    showdf = df
    st.dataframe(showdf)