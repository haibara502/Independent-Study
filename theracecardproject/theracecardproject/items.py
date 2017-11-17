# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field

class TheracecardprojectItem(scrapy.Item):
	name = scrapy.Field()
	title = scrapy.Field()
	link = scrapy.Field()
	author = scrapy.Field()
	location = scrapy.Field()
	content = scrapy.Field()
	comments = scrapy.Field()
	tag = scrapy.Field()
