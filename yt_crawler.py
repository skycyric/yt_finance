import json
import os
import yt_dlp
import psycopg2
from psycopg2 import sql
from selenium_scraper import get_video_urls_from_channel
import yaml


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
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,  # 修改這裡以獲取詳細信息
        'force_generic_extractor': True,
        'retries': 20,  # 增加重試次數
        'sleep_interval': 10,  # 增加延遲時間
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    if cookies_path:
        ydl_opts['cookies'] = cookies_path

    if not is_playlist:
        video_urls = get_video_urls_from_channel(url)
        entries = [{'url': video_url} for video_url in video_urls]
    else:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
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

    for entry in entries:
        video_url = entry.get('url')
        if not video_url:
            print(f"Skipping entry with no URL: {entry}")
            continue

        video = init_video()

        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'cookies': cookies_path}) as ydl:
                video_info = ydl.extract_info(video_url, download=False)

            video['video_id'] = video_info['id']
            video['title'] = video_info['title']
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
    try:
        os.makedirs(output_path, exist_ok=True)
        print('{} does not exist, creating folder'.format(output_path))
    except Exception as e:
        print(f"Error creating directory {output_path}: {e}")

    for video in videos:
        video_id = video['video_id']
        if not video_id:
            print(f"Skipping video with no ID: {video}")
            continue
        # Sanitize video_id to ensure it's a valid filename
        video_id = "".join([c if c.isalnum() else "_" for c in video_id])
        entity_fname = os.path.join(output_path, f"{video_id}.json")
        try:
            with open(entity_fname, "w", encoding='utf-8') as f:
                json.dump(video, f, ensure_ascii=False, indent=4)
        except OSError as e:
            print(f"Error writing file {entity_fname}: {e}")


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

    # 建立 schema 和 table
    cur.execute(sql.SQL("""
        CREATE SCHEMA IF NOT EXISTS yt_data;
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

    # 插入資料
    for video in videos:
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
    conn.commit()
    cur.close()
    conn.close()


def test_api(url, is_playlist=True, cookies_path=None):
    """
    測試 API 是否能抓取完整的資料。

    Args:
        url: 播放清單或頻道的 URL
        is_playlist: 是否為播放清單的 URL
        cookies_path: cookies 文件的路徑

    Returns:
        bool: 如果資料完整則返回 True，否則返回 False
    """
    videos = get_video_details_from_url(url, is_playlist, cookies_path)
    if not videos:
        print("No videos found.")
        return False

    for video in videos:
        if not all([video['video_id'], video['title'], video['href']]):
            print(f"Incomplete data for video: {video}")
            return False

    print("API test passed. All videos have complete data.")
    return True


def main(config):
    url = config['url']
    output_path = config['output_path']
    db_config = config['db_config']
    table_name = config['table_name']
    is_playlist = config.get('is_playlist', True)
    cookies_path = config.get('cookies_path')

    print('url: {}'.format(url))

    # 測試 API 是否能抓取完整的資料
    if not test_api(url, is_playlist, cookies_path):
        print("API test failed. Exiting.")
        return

    # 抓取播放清單或頻道中的影片資料
    videos = get_video_details_from_url(url, is_playlist, cookies_path)
    # 輸出抓到影片資訊的結果
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video['title']}: {video['href']}")

    # 保存影片資料到 JSON 文件
    save_videos_to_json(videos, output_path)

    # 保存影片資料到 PostgreSQL
    save_videos_to_postgresql(videos, db_config, table_name)


def load_db_config(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    channels = {
        'tvbs_money': {
            'url': 'https://www.youtube.com/playlist?list=PLmVlUjKuEG7RJfiRNH6_I9x6E-aXdbyGT',
            'output_path': './Subtitles/tvbs_money',
            'table_name': 'tvbs_money_playlist',
            'is_playlist': True,
            'cookies_path': './cookies.txt'
        },
        'tvbs_money_essence': {
            'url': 'https://www.youtube.com/playlist?list=PLmVlUjKuEG7QbNC1XcFyhpiEZRZDUBaUM',
            'output_path': './Subtitles/tvbs_money_essence',
            'table_name': 'tvbs_money_essence_playlist',
            'is_playlist': True,
            'cookies_path': './cookies.txt'
        },
        'zrbros': {
            'url': 'https://www.youtube.com/playlist?list=PLrZrfGLGySzc8YRDxD3HAODPY6McwJ6_0',
            'output_path': './Subtitles/zrbros',
            'table_name': 'zrbros_playlist',
            'is_playlist': True,
            'cookies_path': './cookies.txt'
        },
        'shin_li_hall_of_fame': {
            'url': 'https://www.youtube.com/playlist?list=PLdk_5UmCoVwpNyEq6_HbIu5KgHgjEjCbs',
            'output_path': './Subtitles/shin_li_hall_of_fame',
            'table_name': 'shin_li_hall_of_fame_playlist',
            'is_playlist': True,
            'cookies_path': './cookies.txt'
        },
        'cmoney_etf': {
            'url': 'https://www.youtube.com/playlist?list=PL8JLWRfy17gKB6Z3NJrjAtpHTMtPuQJPK',
            'output_path': './Subtitles/cmoney_etf',
            'table_name': 'cmoney_etf_playlist',
            'is_playlist': True,
            'cookies_path': './cookies.txt'
        },
        'ebcmoneyshow': {
            'url': 'https://www.youtube.com/@EBCmoneyshow',
            'output_path': './Subtitles/ebcmoneyshow',
            'table_name': 'ebcmoneyshow_channel',
            'is_playlist': False,
            'cookies_path': './cookies.txt'
        },
        'cts_featured': {
            'url': 'https://www.youtube.com/@ctsfeatured',
            'output_path': './Subtitles/cts_featured',
            'table_name': 'cts_featured_channel',
            'is_playlist': False,
            'cookies_path': './cookies.txt'
        }
    }

    db_config_path = 'db_config.yaml'  # Path to the YAML file
    db_config = load_db_config(db_config_path)

    for channel_name, config in channels.items():
        config['db_config'] = db_config
        main(config)
