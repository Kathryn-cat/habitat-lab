import gzip
import json

with gzip.open("data/datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz", 'r') as f:
		json_bytes = f.read()
		json_str = json_bytes.decode('utf-8')
		data = json.loads(json_str)
		episodes = data['episodes']
		episode_1 = episodes[1]
		print(episode_1)

