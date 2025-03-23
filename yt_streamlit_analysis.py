import streamlit as st
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, inspect
import altair as alt
from datetime import datetime
from rank_bm25 import BM25Okapi
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_tags import st_tags  # Import st_tags
import requests
import yaml

# Set the page layout to wide
st.set_page_config(
    layout="wide",
    page_title="YT財經類頻道與影片資料分析"
)


def load_config():
    with open("db_config.yaml", "r") as file:
        return yaml.safe_load(file)


config = load_config()

DB_USER = config["database"]["user"]
DB_PASSWORD = config["database"]["password"]
DB_NAME = config["database"]["dbname"]
DB_HOST = config["database"]["host"]
DB_PORT = config["database"]["port"]

# Database connection function


def connect_to_db():
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={
                               'connect_timeout': 10})
        return engine
    except Exception as e:
        st.error("❌ 無法連接到 PostgreSQL，請檢查連線設定")
        st.error(str(e))
        return None

# Function to fetch table names from the schema


def fetch_table_names(engine):
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'yt_data' AND table_name != 'classification_results';
    """
    return pd.read_sql(query, engine)['table_name'].tolist()


# Define table name mapping
table_name_map = {
    'cmoney_etf_playlist': 'CMoney理財寶-ETF錢滾錢',
    'ebcmoneyshow_channel': '理財達人秀 EBCmoneyshow',
    'tvbs_money_playlist': '金臨天下-財經鈔能力',
    'tvbs_money_essence_playlist': '金臨天下-財經鈔能力-精華版',
    'shin_li_hall_of_fame_playlist': 'SHIN LI-理財名人堂',
    'zrbros_playlist': '柴鼠兄弟-台股國民ETF人氣榜',
    'cts_featured_channel': '鈔錢部署–華視優選',
}

# Define fixed colors for each channel
fixed_color_map = {
    'CMoney理財寶-ETF錢滾錢': '#A6FFA6',
    '理財達人秀 EBCmoneyshow': '#FF7575',
    '金臨天下-財經鈔能力': '#FFFFAA',
    '金臨天下-財經鈔能力-精華版': '#FFBB77',
    'SHIN LI-理財名人堂': '#D3A4FF',
    '柴鼠兄弟-台股國民ETF人氣榜': '#A3D1D1',
    '鈔錢部署–華視優選': '#97CBFF'
}

# Function to fetch video data from a specific table


def fetch_video_data(engine, table_name):
    inspector = inspect(engine)
    if not inspector.has_table(table_name, schema='yt_data'):
        raise ValueError(
            f"Table {table_name} does not exist in schema yt_data.")

    query = f"""
    SELECT
        TO_CHAR(upload_date, 'YYYY-MM-DD') AS 上傳日期,
        views AS 觀看數,
        '{table_name_map.get(table_name, table_name)}' AS 頻道,
        title AS 標題,
        tags AS 標籤,
        description AS 描述,
        href AS url
    FROM yt_data.{table_name};
    """
    return pd.read_sql(query, engine)

# Function to fetch the earliest upload date from all tables in the schema


def fetch_earliest_upload_date(engine):
    query = """
    SELECT MIN(upload_date) AS earliest_date
    FROM (
        SELECT upload_date FROM yt_data.cmoney_etf_playlist
        UNION ALL
        SELECT upload_date FROM yt_data.ebcmoneyshow_channel
        UNION ALL
        SELECT upload_date FROM yt_data.tvbs_money_playlist
        UNION ALL
        SELECT upload_date FROM yt_data.tvbs_money_essence_playlist
        UNION ALL
        SELECT upload_date FROM yt_data.shin_li_hall_of_fame_playlist
        UNION ALL
        SELECT upload_date FROM yt_data.zrbros_playlist
        UNION ALL
        SELECT upload_date FROM yt_data.cts_featured_channel
    ) AS all_upload_dates;
    """
    result = pd.read_sql(query, engine)
    return result.iloc[0, 0]

# Function to generate time series graph


def plot_time_series(data, title, date_range):
    data['上傳日期'] = pd.to_datetime(data['上傳日期']).dt.date
    date_range = [pd.to_datetime(date).date() for date in date_range]
    data = data[(data['上傳日期'] >= date_range[0]) &
                (data['上傳日期'] <= date_range[1])]
    time_series_data = data.groupby('上傳日期')['觀看數'].sum().reset_index()
    time_series_data.columns = ['上傳日期', '觀看數']

    chart = alt.Chart(time_series_data).mark_line(point=True).encode(
        x=alt.X('上傳日期:T', title='上傳日期',
                axis=alt.Axis(format='%Y-%m-%d')),
        y=alt.Y('觀看數:Q', title='觀看數'),
        tooltip=['上傳日期:T', '觀看數:Q']
    ).properties(
        title=title,
        width=800,
        height=400
    ).interactive()

    return chart

# Function to generate combined time series graph for all channels


def plot_combined_time_series(engine, selected_tables, table_name_map, date_range):
    combined_data = pd.DataFrame()
    for table_name in selected_tables:
        data = fetch_video_data(engine, table_name)
        data['channel'] = table_name_map.get(table_name, table_name)
        combined_data = pd.concat([combined_data, data])

    if combined_data.empty:
        return None

    combined_data['上傳日期'] = pd.to_datetime(combined_data['上傳日期']).dt.date
    date_range = [pd.to_datetime(date).date() for date in date_range]
    combined_data = combined_data[(combined_data['上傳日期'] >= date_range[0]) & (
        combined_data['上傳日期'] <= date_range[1])]

    # 保留每個影片的觀看數，不按日期加總
    combined_data['影片標題'] = combined_data['標題']

    # Define fixed colors for each channel
    fixed_color_map = {
        'CMoney理財寶-ETF錢滾錢': '#A6FFA6',
        '理財達人秀 EBCmoneyshow': '#FF7575',
        '金臨天下-財經鈔能力': '#FFFFAA',
        '金臨天下-財經鈔能力-精華版': '#FFBB77',
        'SHIN LI-理財名人堂': '#D3A4FF',
        '柴鼠兄弟-台股國民ETF人氣榜': '#A3D1D1',
        '鈔錢部署–華視優選': '#97CBFF'
    }
    selected_channels = [table_name_map[table] for table in selected_tables]
    color_scale = alt.Scale(domain=selected_channels, range=[
                            fixed_color_map[channel] for channel in selected_channels])

    chart = alt.Chart(combined_data).mark_line(point=True).encode(
        x=alt.X('上傳日期:T', title='上傳日期',
                axis=alt.Axis(format='%Y-%m-%d', tickCount='day', labelAngle=-45)),
        y=alt.Y('觀看數:Q', title='觀看數'),
        color=alt.Color('channel:N', scale=color_scale,
                        legend=alt.Legend(labelLimit=200, labelPadding=5)),
        tooltip=['上傳日期:T', 'channel:N', '影片標題:N', '觀看數:Q']
    ).properties(
        title='',
        width=800,
        height=400
    ).configure_legend(
        labelFontSize=12,
        labelAlign='left',
        labelBaseline='middle'
    ).interactive()

    return chart


# Main function to run the Streamlit app


def main():
    st.title('YT財經類頻道與影片資料分析')

    engine = connect_to_db()
    table_names = fetch_table_names(engine)

    table_names_display = [table_name_map.get(
        name.lower(), name) for name in table_names if name in table_name_map]

    st.sidebar.title('選擇頻道')
    selected_tables_display = st.sidebar.multiselect(
        '選擇頻道', table_names_display, default=table_names_display, label_visibility='collapsed')

    selected_tables = [list(table_name_map.keys())[list(
        table_name_map.values()).index(name)] for name in selected_tables_display]

    min_date = fetch_earliest_upload_date(engine)
    max_date = datetime.today().date()
    default_start_date = max_date - pd.Timedelta(days=14)

    st.sidebar.markdown("<hr style='height:3px;border-width:0;color:gray;background-color:white'>",
                        unsafe_allow_html=True)

    # Add date input widgets in the same row
    st.sidebar.title('選擇資料日期範圍')
    st.sidebar.write('(預設顯示最近 14 天的資料)')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.sidebar.date_input(
            '開始日期', value=default_start_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.sidebar.date_input('結束日期', value=max_date,
                                         min_value=min_date, max_value=max_date)
    date_range = (pd.to_datetime(start_date), pd.to_datetime(
        end_date))  # Convert to datetime

    st.markdown("<hr style='height:3px;border-width:0;color:gray;background-color:white'>",
                unsafe_allow_html=True)

    st.subheader('影片發布的觀看數-上傳日期序列圖')
    combined_chart = plot_combined_time_series(
        engine, selected_tables, table_name_map, date_range)
    if combined_chart:
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.write("請選擇至少一個頻道來顯示日期序列圖。")

    st.markdown("<hr style='height:3px;border-width:0;color:gray;background-color:white'>",
                unsafe_allow_html=True)
    # Add search functionality
    st.subheader('搜尋影片資料')
    search_columns = st.multiselect('選擇搜尋欄位', ['標題', '標籤', '描述'])
    search_texts = st_tags(
        label='輸入搜尋文字 (按Enter鍵新增標籤)',
        text='按Enter鍵新增',
        value=[],
        suggestions=[],
        maxtags=-1,
        key='1'
    )
    search_logic = st.radio('多重文字的搜尋邏輯', ['AND', 'OR'])

    try:
        data = pd.DataFrame()
        for table in selected_tables:
            table_data = fetch_video_data(engine, table)
            data = pd.concat([data, table_data])

        if not data.empty:
            data['上傳日期'] = pd.to_datetime(data['上傳日期'])
            data = data[(data['上傳日期'] >= date_range[0]) &
                        (data['上傳日期'] <= date_range[1])]
            data = data.sort_values(by='觀看數', ascending=False)  # Sort by views

            # Apply search filter
            if search_texts and search_columns:
                column_filter = data[search_columns].apply(lambda x: x.str.contains(
                    '|'.join(search_texts), case=False, na=False)).any(axis=1)
                if search_logic == 'AND':
                    for search_text in search_texts:
                        column_filter &= data[search_columns].apply(lambda x: x.str.contains(
                            search_text, case=False, na=False)).any(axis=1)
                data = data[column_filter]
            st.markdown("<hr style='height:3px;border-width:0;color:gray;background-color:white'>",
                        unsafe_allow_html=True)
            st.subheader('影片資料總覽')
            # Display only year, month, and day
            data['上傳日期'] = data['上傳日期'].dt.date
            st.write(data)

            # Add bar chart to display current table content
            color_scale = alt.Scale(domain=list(
                fixed_color_map.keys()), range=list(fixed_color_map.values()))
            bar_chart = alt.Chart(data).mark_bar().encode(
                # Change axis title and remove labels
                x=alt.X('標題:N', title='影片', axis=None, sort='-y'),
                y=alt.Y('觀看數:Q', title='觀看數'),
                color=alt.Color('頻道:N', scale=color_scale,
                                legend=alt.Legend(title="頻道", labelLimit=300)),
                tooltip=['標題:N', '觀看數:Q', '頻道:N']
            ).properties(
                width=800,
                height=400
            ).interactive()

            st.altair_chart(bar_chart, use_container_width=True)

            # Add bar chart to display the count of videos for each brand
            brand_count_data = data['頻道'].value_counts().reset_index()
            brand_count_data.columns = ['頻道', '影片數']

            brand_count_chart = alt.Chart(brand_count_data).mark_bar().encode(
                x=alt.X('頻道:N', title='頻道', axis=None, sort='-y'),
                y=alt.Y('影片數:Q', title='影片數'),
                color=alt.Color('頻道:N', scale=color_scale,
                                legend=alt.Legend(title="頻道", labelLimit=300)),
                tooltip=['頻道:N', '影片數:Q']
            ).properties(
                width=800,
                height=400
            ).interactive()

            st.altair_chart(brand_count_chart, use_container_width=True)
        else:
            st.write("沒有選擇任何頻道或沒有符合條件的數據。")

    except ValueError as e:
        st.error(e)


if __name__ == '__main__':
    main()
