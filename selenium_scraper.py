import json
import os
import yt_dlp
import psycopg2
from psycopg2 import sql
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def get_video_urls_from_channel(channel_url):
    """
    使用 Selenium 抓取頻道中的所有影片 URL。

    Args:
        channel_url: 頻道的 URL

    Returns:
        video_urls: 包含所有影片 URL 的清單
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(
        ChromeDriverManager().install()), options=options)
    driver.get(channel_url)

    video_urls = []
    while True:
        videos = driver.find_elements(By.XPATH, '//*[@id="video-title"]')
        for video in videos:
            video_url = video.get_attribute('href')
            if video_url and video_url not in video_urls:
                video_urls.append(video_url)

        # 滾動到頁面底部以加載更多影片
        driver.execute_script(
            "window.scrollTo(0, document.documentElement.scrollHeight);")
        # 等待加載
        driver.implicitly_wait(3)

        # 檢查是否已經加載完所有影片
        new_videos = driver.find_elements(By.XPATH, '//*[@id="video-title"]')
        if len(new_videos) == len(videos):
            break

    driver.quit()
    return video_urls


def get_video_details_from_url(url, is_playlist=True, cookies_path=None):
    """
    使用 yt-dlp 抓取播放清單或頻道中的影片詳細信息。

    Args:
        url: 播放清單或頻道的 URL
        is_playlist: 是否為播放清單的 URL
        cookies_path: cookies 文件的路徑

    Returns:
        videos: 包含影片詳細信息的清單
    """
    # yt-dlp 提取器選項設定
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,    # 僅提取影片列表，不下載影片
        'force_generic_extractor': True,
        'retries': 20,           # 增加重試次數
        'sleep_interval': 10,    # 增加延遲時間
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    if cookies_path:
        ydl_opts['cookies'] = cookies_path

    # 若不是播放清單，則視為頻道，強制補上 /videos 以提取影片列表
    if not is_playlist:
        if not url.rstrip("/").endswith("videos"):
            url = url.rstrip("/") + "/videos"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get('entries', [])
        except yt_dlp.utils.DownloadError as e:
            print(f"Error extracting channel info {url}: {e}")
            return []
    else:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                entries = info.get('entries', [])
        except yt_dlp.utils.DownloadError as e:
            print(f"Error downloading {url}: {e}")
            return []

    videos = []

    def init_video():
        return {
            'video_id': '',
            'title': '',
            'href': '',
            'tags': [],
            'description': '',
            'views': '',
            'upload_date': ''
        }

    # 逐一處理每個 entry 取得影片詳細資訊
    for entry in entries:
        video_url = entry.get('url')
        if not video_url:
            print(f"Skipping entry with no URL: {entry}")
            continue

        video = init_video()

        try:
            # 注意：每次提取單支影片資訊時可用不同的 yt-dlp 選項
            # 此處僅設定 quiet 與 cookies（若有的話）
            with yt_dlp.YoutubeDL({'quiet': True, 'cookies': cookies_path}) as ydl:
                video_info = ydl.extract_info(video_url, download=False)

            video['video_id'] = video_info.get('id', '')
            video['title'] = video_info.get('title', '')
            video['href'] = video_url
            video['tags'] = video_info.get('tags', [])
            video['description'] = video_info.get('description', '')
            video['views'] = video_info.get('view_count', 0)
            video['upload_date'] = video_info.get('upload_date', '')

            videos.append(video)
        except yt_dlp.utils.DownloadError as e:
            print(f"Error downloading video {video_url}: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    return videos


def save_videos_to_json(videos, output_path):
    """
    將影片資訊儲存為 JSON 檔案
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        print('{} does not exist, creating folder'.format(output_path))
    except Exception as e:
        print(f"Error creating directory {output_path}: {e}")

    for video in videos:
        video_id = video['video_id']
        entity_fname = os.path.join(output_path, f"{video_id}.json")
        with open(entity_fname, "w", encoding='utf-8') as f:
            json.dump(video, f, ensure_ascii=False, indent=4)


def save_videos_to_postgresql(videos, db_config, table_name):
    """
    將影片資料儲存到 PostgreSQL 資料庫中。

    Args:
        videos: 包含影片詳細信息的清單
        db_config: 資料庫配置字典，包含 host, dbname, user, password
        table_name: 要插入資料的表名
    """
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    # 建立 schema 與 table
    try:
        # 先建立 schema
        cur.execute("CREATE SCHEMA IF NOT EXISTS yt_data;")
        conn.commit()
        # 再建立 table
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS yt_data.{} (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                href TEXT,
                tags TEXT[],
                description TEXT,
                views INTEGER,
                upload_date DATE
            );
        """).format(sql.Identifier(table_name)))
        conn.commit()
    except Exception as e:
        print(f"Error creating schema/table: {e}")
        conn.rollback()

    # 插入資料
    for video in videos:
        try:
            cur.execute(sql.SQL("""
                INSERT INTO yt_data.{} (video_id, title, href, tags, description, views, upload_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id) DO NOTHING;
            """).format(sql.Identifier(table_name)), (
                video['video_id'],
                video['title'],
                video['href'],
                video['tags'],
                video['description'],
                video['views'],
                video['upload_date']
            ))
        except Exception as e:
            print(f"Error inserting video {video['video_id']}: {e}")
    conn.commit()
    cur.close()
    conn.close()


def get_channel_subscriber_count(url, cookies_path=None):
    """
    使用 yt-dlp 抓取頻道的訂閱數。

    Args:
        url: 頻道的 URL
        cookies_path: cookies 文件的路徑

    Returns:
        subscriber_count: 頻道的訂閱數
    """
    ydl_opts = {
        'quiet': True,
        'force_generic_extractor': True,
        'retries': 20,
        'sleep_interval': 10,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    if cookies_path:
        ydl_opts['cookies'] = cookies_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subscriber_count = info.get('subscriber_count', 0)
    except yt_dlp.utils.DownloadError as e:
        print(f"Error extracting channel info {url}: {e}")
        return 0

    return subscriber_count


def main(config):
    url = config['url']
    output_path = config['output_path']
    db_config = config['db_config']
    table_name = config['table_name']
    is_playlist = config.get('is_playlist', True)
    cookies_path = config.get('cookies_path')

    print('url: {}'.format(url))

    # 抓取播放清單或頻道中的影片資料
    videos = get_video_details_from_url(url, is_playlist, cookies_path)
    # 抓取頻道訂閱數
    if not is_playlist:
        subscriber_count = get_channel_subscriber_count(url, cookies_path)
        print(f"頻道訂閱數: {subscriber_count}")

    # 輸出抓到影片資訊的結果
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video['title']}: {video['href']}")

    # 保存影片資料到 JSON 文件
    save_videos_to_json(videos, output_path)

    # 保存影片資料到 PostgreSQL
    save_videos_to_postgresql(videos, db_config, table_name)


if __name__ == "__main__":
    channels = {
        'daisyhiu': {
            'url': 'https://www.youtube.com/@daisychiu0101',  # 頻道 URL（非播放清單）
            'output_path': './Subtitles/daisyhiu',
            'table_name': 'daisyhiu_channel',
            'is_playlist': False,
            'cookies_path': './cookies.txt'
        },
        'ebcmoneyshow': {
            'url': 'https://www.youtube.com/@EBCmoneyshow',  # 頻道 URL
            'output_path': './Subtitles/eBcmoneyshow',
            'table_name': 'eBcmoneyshow_channel',
            'is_playlist': False,
            'cookies_path': './cookies.txt'
        },
        'finance_frontline': {
            'url': 'https://www.youtube.com/@yuantachannel',  # 頻道 URL
            'output_path': './Subtitles/finance_frontline',
            'table_name': 'finance_frontline_channel',
            'is_playlist': False,
            'cookies_path': './cookies.txt'
        }
    }

    db_config = {
        'host': 'localhost',
        'dbname': 'postgres',
        'user': 'tvbs',
        'password': '10030805'
    }

    # 逐一處理各頻道
    for channel_name, config in channels.items():
        config['db_config'] = db_config
        main(config)
