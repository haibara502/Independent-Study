# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import codecs
import json

class TheracecardprojectPipeline(object):
	def __init__(self):
		self.ids_seen = set()
		self.file = codecs.open('theracecardprojet_log.json', 'w', encoding='utf-8')
		
	def process_item(self, item, spider):
		path = item['name']
		
		if path in self.ids_seen:
			raise DropItem("Duplicate item found: %s" % path)
		else:
			self.ids_seen.add(path)

		line = json.dumps(dict(item), ensure_ascii = True) + "\n"
		self.file.write(line)
		return item

	def spider_closed(self, spider):
		self.file.close()

		
