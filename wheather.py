import requests
import os
from datetime import datetime

def get_location():
    """IPアドレスから現在地（緯度・経度）を取得"""
    try:
        res = requests.get("http://ip-api.com/json/")
        data = res.json()
        lat, lon = data["lat"], data["lon"]
        return lat, lon
    except:
        return None, None

def get_weather_by_location(target_datetime=None):
    """
    現在地の天気を取得
    target_datetime: datetimeオブジェクト。Noneなら現在時刻
    """
    lat, lon = get_location()
    if lat is None or lon is None:
        return "不明"

    api_key = os.getenv("WEATHER_APIKEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&lang=ja&units=metric"

    res = requests.get(url)
    if res.status_code != 200:
        return "不明"

    weather_desc = res.json()["weather"][0]["description"]

    # 日時を指定されていれば文字列にして付加
    if target_datetime is None:
        target_datetime = datetime.now()
        print("target_datetime", target_datetime)
    datetime_str = target_datetime.strftime("%Y-%m-%d %H:%M:%S")
    print("datetime_str", datetime_str)

    return f"{weather_desc}"

if __name__ == "__main__":
    # 現在時刻で取得
    weather_now = get_weather_by_location()
    print("weather_now:", weather_now)


