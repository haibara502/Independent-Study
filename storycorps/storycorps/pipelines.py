# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import codecs
import json


class StorycorpsPipeline(object):

	def __init__(self):
		self.ids_seen = set()
	
	def process_item(self, item, spider):
		path = item['name']
		path = "listen-" + path

		if path in self.ids_seen:
			raise DropItem("Duplicate item found: %s" % path)
		else:
			self.ids_seen.add(path)

		isExist = os.path.exists(path)
		if not isExist :
			os.mkdir(path)
		with open("".join(path + '/' + path + '-mp3.txt'), "wb") as f:
			f.write("".join(item['audio']))
		with open("".join(path + '/' + path + '-script.txt'), "wb") as f:
			f.write("".join(item['script']))

		del item['audio']

		return item

class StorycorpsJsonPipeline(object):

	def __init__(self):
		self.file = codecs.open('listen_logs.json', 'w', encoding='utf-8')

	def process_item(self, item, spider):
		line = json.dumps(dict(item), ensure_ascii = True) + "\n"
		self.file.write(line)
		return item

	def spider_closed(self, spider):
		self.file.close()
