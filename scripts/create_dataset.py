import hashlib
import io
import os

import fire
import numpy as np
import PIL.Image
import requests
import pathlib
import pandas as pd
import sklearn.metrics

KEY=os.environ.get('KEY')
assert KEY is not None

def get_image(latitude, longitude, heading=0):
    url =f'https://maps.googleapis.com/maps/api/streetview?location={latitude},{longitude}&size=456x456&key={KEY}&heading={heading}'
    d = requests.get(url)
    return PIL.Image.open(io.BytesIO(d.content))

def uniform_random_noise(n):
    for _ in range(n):
        length = np.sqrt(np.random.uniform(0, 1))
        angle = np.pi * np.random.uniform(0, 2)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        yield (x, y)

def get_town(district, area):
    d = requests.get(f'https://geolonia.github.io/japanese-addresses/api/ja/{district}/{area}.json')
    df = pd.DataFrame(d.json())
    return df

AREAS = [
    ('東京都', '中央区'),
    ('東京都', '港区'),
]

def main():
    basedir = pathlib.Path(__file__).parent
    for district, area in AREAS:
        towns = get_town(district, area)

        darray = sklearn.metrics.pairwise_distances(towns[['lat', 'lng']], towns[['lat', 'lng']])
        r = darray[darray > 0.0001].min()

        for _, row in towns.iterrows():
            print(row.town)
            seed = int(hashlib.sha1(row.town.encode("UTF-8")).hexdigest(), 16) % 2**32
            np.random.seed(seed)
            target_dir = basedir / (district+area) / row.town
            target_dir.mkdir(exist_ok=True, parents=True)
            center_lat, center_lng = row.lat, row.lng
            for x, y in uniform_random_noise(10):
                lat = center_lat + x * r
                lng = center_lng + y * r
                heading = np.random.randint(180)
                path: pathlib.Path = target_dir / f'{lat}{lng}{heading}.png'
                if not path.exists():
                    img = get_image(lat, lng, heading=heading)
                    img.save(path)


if __name__ == '__main__':
    fire.Fire(main)
