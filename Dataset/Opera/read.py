#! /usr/bin/python

import json
import yaml

#json_file = "Friends_s2.json"
json_file = "TheBigBangTheory_s1.json"
json_data = open(json_file)
#data = yaml.safe_load(json_data)
data = json.load(json_data)
json_data.close()
	
#output_file = open("Friends_s2.only_statement.no_no_speaker_utterance.txt", "w")
output_file = open("TheBigBangTheory_s1.all.conversations.txt", "w")
insert = "***+***"

for episodes in data['episodes']: # Episode number
	print ('---')
	print "episodes"
	print episodes
	for episode in data['episodes'][episodes]: #['episode_id', 'scenes']
		print '--'
		print "episode"
		print episode
		episode_id = data['episodes'][episodes]['episode_id']
		scenes = data['episodes'][episodes]['scenes']
		for scene in scenes: #['scene_id', 'utterances']
			print '-'
			print 'scene'
			print scene 
			scene_id = scenes[scene]['scene_id']
			print scene_id
			utterances = scenes[scene]['utterances']
			print type(utterances)
			for utterance in utterances:
				utterance_id = utterance['utterance_id']
				speaker = utterance.get('speaker')
				if speaker is None:
					speaker = "None"
				utterance_raw = utterance['utterance_raw']
				statement = utterance['statment_raw']

				output_file.write(str(episode_id))
				output_file.write(insert)
				output_file.write(str(scene_id))
				output_file.write(insert)
				output_file.write(str(utterance_id))
				output_file.write(insert)
				output_file.write(speaker.encode('ascii', 'ignore').decode('ascii'))
				output_file.write(insert)
				output_file.write(utterance_raw.encode('ascii', 'ignore').decode('ascii'))
				output_file.write(insert)
				output_file.write(statement.encode('ascii', 'ignore').decode('ascii'))
				output_file.write('\n')
