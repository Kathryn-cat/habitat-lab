import gzip
import json

with gzip.open("data/datasets/rearrange_pick/replica_cad/v0/train/train_counter_L_analysis_5000_500.json.gz", 'r') as f:
		json_bytes = f.read()
		json_str = json_bytes.decode('utf-8')
		data = json.loads(json_str)
		episodes = data['episodes']
		episode_0 = episodes[0]
		print(episode_0)

